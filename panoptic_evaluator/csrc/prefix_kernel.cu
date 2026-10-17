#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <climits>
#include <vector>

std::vector<torch::Tensor> compute_prefix_cuda(torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor);

namespace {
// Aggregate equal keys within a warp, including signed histogram transfers.
__device__ void add(int64_t* bins, int64_t key, int sign = 1) {
#if __CUDA_ARCH__ >= 700
  unsigned peers = __match_any_sync(__activemask(), key);
  if ((threadIdx.x & 31) != __ffs(peers)-1) return;
  int64_t value = static_cast<int64_t>(__popc(peers)) * sign;
#else
  int64_t value = sign;
#endif
  atomicAdd(reinterpret_cast<unsigned long long*>(bins + key),
            static_cast<unsigned long long>(value));
}

__global__ void initialize(const int* gt, const int* gc, const bool* crowd,
    const bool* things, int* canvas, int64_t* ga, int64_t* pa, int64_t* pairs,
    int64_t* positives, int* error, int64_t pixels, int G, int P, int C) {
  int64_t x = static_cast<int64_t>(blockIdx.x)*blockDim.x + threadIdx.x;
  if (x == 0) pa[0] = pixels;
  for (int64_t i=x; i<pixels; i+=static_cast<int64_t>(gridDim.x)*blockDim.x) {
    canvas[i] = 0;
    int g = gt[i];
    if (g >= 0 && g < G) { add(ga, g); add(pairs, static_cast<int64_t>(g)*P); }
    else atomicOr(error, 1);
  }
  for (int g=x; g<G; g+=gridDim.x*blockDim.x) {
    int c=gc[g];
    if (c>=0 && c<C && things[c] && !crowd[g]) add(positives, c);
  }
}

// One dense mask pass updates the visible canvas, segment areas/intersections,
// and original-mask intersections needed by independent instance matching.
__global__ void draw(const int* gt, const bool* mask, int* canvas,
    int64_t* pairs, int64_t* pa, int64_t* raw, int64_t pixels, int G, int P, int p) {
  for (int64_t x=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
       x<pixels; x+=static_cast<int64_t>(gridDim.x)*blockDim.x) {
    if (!mask[x]) continue;
    int g=gt[x];
    if (g<0 || g>=G) continue;
    int old=canvas[x];
    add(pairs, static_cast<int64_t>(g)*P+old, -1);
    add(pairs, static_cast<int64_t>(g)*P+p);
    add(pa, old, -1); add(pa, p); add(raw, g);
    canvas[x]=p;
  }
}

// A single cooperative block consumes the completed histogram. Each threshold
// owns its matching chain, so earlier instance decisions are never recomputed.
// Canvas scoring only runs at requested checkpoints. No host read per step.
__global__ void score(const int* gc, const int* pc, const bool* crowd,
    const bool* things, const double* thresholds, const int64_t* ga,
    const int64_t* pa, const int64_t* pairs, const int64_t* raw,
    bool* used, int64_t* itp, int64_t* ifp, const int64_t* positives,
    int* matched, int64_t* tp, int64_t* fp, int64_t* fn, double* sums,
    int64_t* confusion, int64_t* out_itp, int64_t* out_ifp, int64_t* out_ifn,
    double* precision, double* recall, int* error,
    int G, int P, int C, int T, int p, int q, bool validate) {
  int t=threadIdx.x;
  if (p>0) {
    int c=pc[p-1];
    int64_t area=pa[p];
    if (t==0 && validate && area==0) atomicOr(error, 2);
    if (c>=0 && c<C && things[c]) {
      for (int k=t; k<T; k+=blockDim.x) {
        int best=-1; double best_iou=-1; bool ignored=false;
        for (int g=0; g<G; ++g) {
          if (gc[g]!=c) continue;
          int64_t intersection=raw[g];
          int64_t u=crowd[g] ? area : area+ga[g]-intersection;
          double iou=static_cast<double>(intersection)/max(int64_t(1),u);
          if (iou<thresholds[k]) continue;
          if (crowd[g]) ignored=true;
          else if (!used[static_cast<int64_t>(k)*G+g] && iou>best_iou) {
            best=g; best_iou=iou;
          }
        }
        if (best>=0) {
          used[static_cast<int64_t>(k)*G+best]=true;
          ++itp[static_cast<int64_t>(k)*C+c];
        } else if (!ignored) ++ifp[static_cast<int64_t>(k)*C+c];
      }
    }
  }
  if (q<0) return;
  tp+=static_cast<int64_t>(q)*C; fp+=static_cast<int64_t>(q)*C;
  fn+=static_cast<int64_t>(q)*C; sums+=static_cast<int64_t>(q)*C;
  confusion+=static_cast<int64_t>(q)*C*(C+1);
  for (int g=t; g<G; g+=blockDim.x) matched[g]=0;
  __syncthreads();
  for (int j=t; j<=p; j+=blockDim.x) {
    int c=j ? pc[j-1] : -1;
    bool valid=j>0 && c>=0 && c<C && pa[j]>0;
    bool pm=false; double sum=0;
    int64_t ignored=pairs[j];
    for (int g=0; g<G; ++g) {
      int target=gc[g];
      int64_t intersection=pairs[static_cast<int64_t>(g)*P+j];
      if (target>=0 && target<C && !crowd[g] && intersection)
        atomicAdd(reinterpret_cast<unsigned long long*>(confusion+
          static_cast<int64_t>(target)*(C+1)+(valid ? c : C)),
          static_cast<unsigned long long>(intersection));
      if (!valid || target!=c) continue;
      if (crowd[g]) ignored=pairs[j]+intersection;
      else if (intersection) {
        int64_t u=ga[g]+pa[j]-intersection-pairs[j];
        if (intersection>u/2) {
          pm=true; sum+=static_cast<double>(intersection)/u;
          atomicExch(matched+g,1);
        }
      }
    }
    if (valid) {
      if (pm) { add(tp,c); atomicAdd(sums+c,sum); }
      else if (static_cast<double>(ignored)<=0.5*pa[j]) add(fp,c);
    }
  }
  __syncthreads();
  for (int g=t; g<G; g+=blockDim.x)
    if (gc[g]>=0 && gc[g]<C && !crowd[g] && !matched[g]) add(fn,gc[g]);
  for (int k=t; k<T*C; k+=blockDim.x) {
    int64_t offset=static_cast<int64_t>(q)*T*C+k;
    int64_t a=itp[k], b=ifp[k], n=positives[k%C];
    out_itp[offset]=a; out_ifp[offset]=b; out_ifn[offset]=n-a;
    precision[offset]=static_cast<double>(a)/max(int64_t(1),a+b);
    recall[offset]=static_cast<double>(a)/max(int64_t(1),n);
  }
}
}  // namespace

std::vector<torch::Tensor> prefix_cuda(torch::Tensor gt, torch::Tensor masks,
    torch::Tensor pc, torch::Tensor gc, torch::Tensor crowd, torch::Tensor things,
    torch::Tensor thresholds, std::vector<int64_t> points, bool validate) {
  TORCH_CHECK(gt.is_cuda(), "prefix inputs must be CUDA tensors");
  c10::cuda::CUDAGuard guard(gt.device());
  for (const auto& t : {gt,masks,pc,gc,crowd,things,thresholds})
    TORCH_CHECK(t.device()==gt.device() && t.is_contiguous(), "incorrect device or layout");
  TORCH_CHECK(gt.scalar_type()==torch::kInt32 && masks.scalar_type()==torch::kBool &&
    pc.scalar_type()==torch::kInt32 && gc.scalar_type()==torch::kInt32 &&
    crowd.scalar_type()==torch::kBool && things.scalar_type()==torch::kBool &&
    thresholds.scalar_type()==torch::kFloat64, "incorrect prefix input dtypes");
  TORCH_CHECK(gt.dim()==2 && masks.dim()==3 && pc.dim()==1 && gc.dim()==1 &&
    crowd.sizes()==gc.sizes() && things.dim()==1 && thresholds.dim()==1 &&
    masks.size(0)==pc.numel() && masks.size(1)==gt.size(0) && masks.size(2)==gt.size(1),
    "incorrect prefix input shapes");
  int64_t N=pc.numel(), G=gc.numel(), C=things.numel(), T=thresholds.numel(), K=points.size();
  TORCH_CHECK(N<INT_MAX && G>0 && G<INT_MAX && C>0 && C<INT_MAX && T>0 && T*C<INT_MAX &&
    gt.numel()>0 && K<INT_MAX, "prefix dimensions outside supported range");
  int64_t last=-1;
  for (auto q: points) { TORCH_CHECK(q>last && q>=0 && q<=N, "invalid checkpoints"); last=q; }
  auto longs=gt.options().dtype(torch::kInt64), doubles=gt.options().dtype(torch::kFloat64);
  auto tp=torch::zeros({K,C},longs), fp=torch::zeros_like(tp), fn=torch::zeros_like(tp);
  auto sums=torch::zeros({K,C},doubles), confusion=torch::zeros({K,C,C+1},longs);
  auto oitp=torch::empty({K,T,C},longs), oifp=torch::empty_like(oitp), oifn=torch::empty_like(oitp);
  auto precision=torch::empty({K,T,C},doubles), recall=torch::empty_like(precision);
  auto ga=torch::zeros({G},longs), pa=torch::zeros({N+1},longs);
  auto pairs=torch::zeros({G,N+1},longs), raw=torch::zeros({N,G},longs);
  auto positives=torch::zeros({C},longs), itp=torch::zeros({T,C},longs), ifp=torch::zeros_like(itp);
  auto used=torch::zeros({T,G},gt.options().dtype(torch::kBool));
  auto matched=torch::empty({G},gt.options()), canvas=torch::empty_like(gt);
  auto error=torch::zeros({},gt.options());
  auto stream=at::cuda::getCurrentCUDAStream();
  int blocks=std::min<int64_t>(4096,(std::max(gt.numel(),G)+255)/256);
  initialize<<<blocks,256,0,stream>>>(gt.data_ptr<int>(),gc.data_ptr<int>(),crowd.data_ptr<bool>(),
    things.data_ptr<bool>(),canvas.data_ptr<int>(),ga.data_ptr<int64_t>(),pa.data_ptr<int64_t>(),
    pairs.data_ptr<int64_t>(),positives.data_ptr<int64_t>(),error.data_ptr<int>(),gt.numel(),G,N+1,C);
  int q=0;
  for (int p=0; p<=N; ++p) {
    if (p) draw<<<blocks,256,0,stream>>>(gt.data_ptr<int>(),
      masks.data_ptr<bool>()+static_cast<int64_t>(p-1)*gt.numel(),canvas.data_ptr<int>(),
      pairs.data_ptr<int64_t>(),pa.data_ptr<int64_t>(),raw.data_ptr<int64_t>()+static_cast<int64_t>(p-1)*G,
      gt.numel(),G,N+1,p);
    int out=q<K && points[q]==p ? q++ : -1;
    if (p || out>=0) score<<<1,256,0,stream>>>(gc.data_ptr<int>(),pc.data_ptr<int>(),crowd.data_ptr<bool>(),
      things.data_ptr<bool>(),thresholds.data_ptr<double>(),ga.data_ptr<int64_t>(),pa.data_ptr<int64_t>(),
      pairs.data_ptr<int64_t>(),p ? raw.data_ptr<int64_t>()+static_cast<int64_t>(p-1)*G : nullptr,
      used.data_ptr<bool>(),itp.data_ptr<int64_t>(),ifp.data_ptr<int64_t>(),positives.data_ptr<int64_t>(),
      matched.data_ptr<int>(),tp.data_ptr<int64_t>(),fp.data_ptr<int64_t>(),fn.data_ptr<int64_t>(),
      sums.data_ptr<double>(),confusion.data_ptr<int64_t>(),oitp.data_ptr<int64_t>(),oifp.data_ptr<int64_t>(),
      oifn.data_ptr<int64_t>(),precision.data_ptr<double>(),recall.data_ptr<double>(),error.data_ptr<int>(),
      G,N+1,C,T,p,out,validate);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  auto result=compute_prefix_cuda(tp,fp,fn,sums,confusion,things);
  result.insert(result.end(),{oitp,oifp,oifn,precision,recall});
  result.push_back(error);
  return result;
}
