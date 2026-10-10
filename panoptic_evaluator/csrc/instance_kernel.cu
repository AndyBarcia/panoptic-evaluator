#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cfloat>
#include <climits>
#include <vector>

namespace {
__global__ void pack_masks(const bool* masks, int32_t* packed, int N, int64_t pixels, int64_t words) {
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= N * words) return;
  int64_t n = i / words, base = (i % words) * 32;
  unsigned bits = 0;
  for (int b = 0; b < 32 && base + b < pixels; ++b)
    bits |= static_cast<unsigned>(masks[n * pixels + base + b]) << b;
  packed[i] = static_cast<int32_t>(bits);
}

__global__ void overlap(const int32_t* p, const int32_t* g, const int32_t* pc,
    const int32_t* gc, const double* pa, const double* ga, const bool* crowd,
    double* iou, int D, int G, int64_t words) {
  int d = blockIdx.x / G, gt = blockIdx.x % G;
  __shared__ unsigned long long sums[256];
  unsigned long long value = 0;
  if (pc[d] == gc[gt])
    for (int64_t w = threadIdx.x; w < words; w += blockDim.x)
      value += __popc(static_cast<unsigned>(p[d * words + w] & g[gt * words + w]));
  sums[threadIdx.x] = value;
  __syncthreads();
  for (int s = 128; s; s >>= 1) {
    if (threadIdx.x < s) sums[threadIdx.x] += sums[threadIdx.x + s];
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    double denominator = crowd[gt] ? pa[d] : pa[d] + ga[gt] - sums[0];
    iou[static_cast<int64_t>(d) * G + gt] = denominator > 0 ? sums[0] / denominator : 0;
  }
}

__device__ void area_range(int a, double& lo, double& hi) {
  lo = a == 2 ? 1024. : a == 3 ? 9216. : 0.;
  hi = a == 1 ? 1024. : a == 2 ? 9216. : 1e10;
}

// Build stable category lists once, instead of filtering all slots in every
// one of the 40 matching chains. Inputs are already stably sorted by score.
__global__ void group_candidates(const int32_t* pc, const int32_t* gc,
    const bool* crowd, const double* areas, int32_t* detections, int32_t* targets,
    int32_t* sizes, int32_t* ranks, int64_t* counts, int D, int G, int C) {
  int lane = threadIdx.x & 31;
  int c = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  if (c >= C) return;
  int nd = 0, ng = 0;
  int64_t positive[4] = {};
  unsigned earlier = (1u << lane) - 1;
  for (int64_t base = 0; base < D; base += 32) {
    int64_t d = base + lane;
    bool selected = d < D && pc[d] == c;
    unsigned peers = __ballot_sync(0xffffffff, selected);
    if (selected) {
      int rank = nd + __popc(peers & earlier) + 1;
      ranks[d] = rank;
      if (rank <= 100) detections[c * 100 + rank - 1] = d;
    }
    nd += __popc(peers);
  }
  for (int64_t base = 0; base < G; base += 32) {
    int64_t g = base + lane;
    bool selected = g < G && gc[g] == c;
    unsigned peers = __ballot_sync(0xffffffff, selected);
    if (selected)
      targets[static_cast<int64_t>(c) * G + ng + __popc(peers & earlier)] = g;
    ng += __popc(peers);
    bool eligible = selected && !crowd[g];
    double area = eligible ? areas[g] : 0.;
    for (int a = 0; a < 4; ++a) {
      double lo, hi; area_range(a, lo, hi);
      positive[a] += __popc(__ballot_sync(0xffffffff, eligible && area >= lo && area <= hi));
    }
  }
  if (lane == 0) {
    sizes[c * 2] = min(nd, 100);
    sizes[c * 2 + 1] = ng;
    for (int a = 0; a < 4; ++a) counts[c * 4 + a] = positive[a];
  }
}

__device__ bool better_match(int ignore, double value, int g,
                            int best_ignore, double best_value, int best_g) {
  // Eligible ground truth wins over ignored ground truth, then greatest IoU,
  // then the later metadata entry on equal IoU, as in COCO's stable scan.
  return g >= 0 && (ignore < best_ignore ||
      (ignore == best_ignore && (value > best_value ||
                               (value == best_value && g > best_g))));
}

// A warp owns one category/area/IoU chain. Detections remain sequential, while
// lanes search same-category ground truths together and reduce the best match.
__global__ void match(const double* iou, const int32_t* detections,
    const int32_t* targets, const int32_t* sizes, const double* pa,
    const bool* crowd, const double* areas, int8_t* flags, bool* used,
    int D, int G, int C) {
  int lane = threadIdx.x & 31;
  int job = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
  if (job >= C * 40) return;
  int c = job / 40, a = (job % 40) / 10, t = job % 10;
  double lo, hi; area_range(a, lo, hi);
  const double thresholds[10] = {.5, .55, .6, .65, .7, .75, .8, .85, .8999999999999999, .95};
  double threshold = thresholds[t];
  bool* taken = used + static_cast<int64_t>(job) * G;
  int nd = sizes[c * 2], ng = sizes[c * 2 + 1];
  for (int rank = 0; rank < nd; ++rank) {
    int d = detections[c * 100 + rank];
    int best_g = -1, best_j = -1, best_ignore = 2;
    double best_value = threshold;
    for (int j = lane; j < ng; j += 32) {
      int g = targets[static_cast<int64_t>(c) * G + j];
      if (taken[j] && !crowd[g]) continue;
      double value = iou[static_cast<int64_t>(d) * G + g];
      int ignore = crowd[g] || areas[g] < lo || areas[g] > hi;
      if (value >= threshold && better_match(ignore, value, g, best_ignore, best_value, best_g)) {
        best_g = g;
        best_j = j;
        best_ignore = ignore;
        best_value = value;
      }
    }
    for (int offset = 16; offset; offset >>= 1) {
      int other_g = __shfl_down_sync(0xffffffff, best_g, offset);
      int other_j = __shfl_down_sync(0xffffffff, best_j, offset);
      int other_ignore = __shfl_down_sync(0xffffffff, best_ignore, offset);
      double other_value = __shfl_down_sync(0xffffffff, best_value, offset);
      if (better_match(other_ignore, other_value, other_g, best_ignore, best_value, best_g)) {
        best_g = other_g;
        best_j = other_j;
        best_ignore = other_ignore;
        best_value = other_value;
      }
    }
    if (lane == 0) {
      int8_t flag = -1;
      if (best_g >= 0) {
        taken[best_j] = true;
        flag = best_ignore ? -1 : 1;
      } else if (pa[d] >= lo && pa[d] <= hi) flag = 0;
      flags[(a * 10 + t) * static_cast<int64_t>(D) + d] = flag;
    }
    __syncwarp();  // Publish this detection's match before considering the next.
  }
}

std::vector<torch::Tensor> match_instances(torch::Tensor ious, torch::Tensor pc,
    torch::Tensor gc, torch::Tensor pa, torch::Tensor crowd, torch::Tensor areas, int C) {
  int D = pc.numel(), G = gc.numel();
  auto flags = torch::full({40, D}, -1, pc.options().dtype(torch::kInt8));
  auto ranks = torch::zeros({D}, pc.options());
  auto counts = torch::zeros({C, 4}, pc.options().dtype(torch::kInt64));
  auto detections = torch::empty({C, 100}, pc.options());
  auto targets = torch::empty({C, G}, gc.options());
  auto sizes = torch::empty({C, 2}, pc.options());
  auto used = torch::zeros({C * 40, G}, crowd.options());
  auto stream = at::cuda::getCurrentCUDAStream();
  group_candidates<<<(C + 3) / 4, 128, 0, stream>>>(pc.data_ptr<int32_t>(),
      gc.data_ptr<int32_t>(), crowd.data_ptr<bool>(), areas.data_ptr<double>(),
      detections.data_ptr<int32_t>(), targets.data_ptr<int32_t>(), sizes.data_ptr<int32_t>(),
      ranks.data_ptr<int32_t>(), counts.data_ptr<int64_t>(), D, G, C);
  match<<<(C * 40 + 3) / 4, 128, 0, stream>>>(ious.data_ptr<double>(),
      detections.data_ptr<int32_t>(), targets.data_ptr<int32_t>(), sizes.data_ptr<int32_t>(),
      pa.data_ptr<double>(), crowd.data_ptr<bool>(), areas.data_ptr<double>(),
      flags.data_ptr<int8_t>(), used.data_ptr<bool>(), D, G, C);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {flags, ranks, counts};
}

// Global score order is supplied by a stable device sort. Each thread reduces
// one category/area/threshold/maxDets setting into 101 recall buckets.
__global__ void accumulate(const int32_t* classes, const int32_t* ranks,
    const int8_t* flags, const int64_t* counts, double* precision, double* recall,
    int64_t D, int C) {
  int job = blockIdx.x * blockDim.x + threadIdx.x;
  if (job >= C * 120) return;
  int m = job % 3, a = (job / 3) % 4, c = (job / 12) % C, t = job / (12 * C);
  int64_t positives = counts[c * 4 + a];
  if (!positives) return;
  double buckets[101] = {};
  int64_t tp = 0, fp = 0;
  int limit = m == 0 ? 1 : m == 1 ? 10 : 100;
  for (int64_t d = 0; d < D; ++d) {
    if (classes[d] != c || ranks[d] < 1 || ranks[d] > limit) continue;
    int flag = flags[(a * 10 + t) * D + d];
    if (flag == 0) ++fp;
    if (flag != 1) continue;
    ++tp;
    double rc = static_cast<double>(tp) / positives;
    int b = min(100, static_cast<int>(rc * 100));
    // Compare against the actual recall grid to avoid boundary rounding errors.
    while (b < 100 && rc >= (b + 1) * .01) ++b;
    while (b > 0 && rc < b * .01) --b;
    double pr = static_cast<double>(tp) / (static_cast<double>(tp + fp) + DBL_EPSILON);
    buckets[b] = fmax(buckets[b], pr);
  }
  double envelope = 0;
  for (int r = 100; r >= 0; --r) {
    envelope = fmax(envelope, buckets[r]);
    precision[((((static_cast<int64_t>(t) * 101 + r) * C + c) * 4 + a) * 3 + m)] = envelope;
  }
  recall[((t * C + c) * 4 + a) * 3 + m] = static_cast<double>(tp) / positives;
}

// One pixel pass per detection against the shared ground-truth slot map.
// Every covered pixel contributes, including void and stuff, so row sums give
// full detection area even though only thing columns participate in AP.
template <bool Shared>
__global__ void detection_histogram(const bool* masks, const int32_t* target,
    const int64_t* order, int64_t* pairs, int64_t pixels, int S, int chunks) {
  extern __shared__ unsigned bins[];
  int d = blockIdx.x / chunks, chunk = blockIdx.x % chunks;
  if (Shared) {
    for (int s = threadIdx.x; s < S; s += blockDim.x) bins[s] = 0;
    __syncthreads();
  }
  int64_t begin = static_cast<int64_t>(chunk) * 8192;
  int64_t end = min(begin + 8192, pixels);
  const bool* mask = masks + order[d] * pixels;
  for (int64_t p = begin + threadIdx.x; p < end; p += blockDim.x) {
    if (!mask[p]) continue;
    int slot = target[p];
    if (slot < 0 || slot >= S) continue;  // PQ validation reports bad map values.
#if __CUDA_ARCH__ >= 700
    unsigned peers = __match_any_sync(__activemask(), slot);
    if ((threadIdx.x & 31) != __ffs(peers) - 1) continue;
    unsigned value = __popc(peers);
#else
    unsigned value = 1;
#endif
    if (Shared) atomicAdd(bins + slot, value);
    else atomicAdd(reinterpret_cast<unsigned long long*>(pairs) + static_cast<int64_t>(d) * S + slot,
                   static_cast<unsigned long long>(value));
  }
  if (Shared) {
    __syncthreads();
    for (int s = threadIdx.x; s < S; s += blockDim.x)
      if (bins[s]) atomicAdd(reinterpret_cast<unsigned long long*>(pairs) + static_cast<int64_t>(d) * S + s,
                            static_cast<unsigned long long>(bins[s]));
  }
}

__global__ void histogram_iou(const int64_t* pairs, const double* pa,
    const double* ga, const bool* crowd, double* iou, int64_t total, int S) {
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= total) return;
  int d = i / S, g = i % S;
  double intersection = static_cast<double>(pairs[i]);
  double denominator = crowd[g] ? pa[d] : pa[d] + ga[g] - intersection;
  iou[i] = denominator > 0 ? intersection / denominator : 0.;
}

void check(const torch::Tensor& x, const torch::Tensor& reference, at::ScalarType type) {
  TORCH_CHECK(x.is_cuda() && x.device() == reference.device() && x.is_contiguous() &&
              x.scalar_type() == type, "invalid instance tensor device, layout or dtype");
}

void check_metadata(const torch::Tensor& reference, torch::Tensor pc, torch::Tensor gc,
    torch::Tensor pa, torch::Tensor ga, torch::Tensor crowd, torch::Tensor areas,
    int64_t D, int64_t G) {
  for (const auto& tensor : {pc, gc}) check(tensor, reference, torch::kInt32);
  for (const auto& tensor : {pa, ga, areas}) check(tensor, reference, torch::kFloat64);
  check(crowd, reference, torch::kBool);
  TORCH_CHECK(D <= INT_MAX && G <= INT_MAX && D * G <= INT_MAX,
              "too many instance pairs");
  for (const auto& tensor : {pc, pa})
    TORCH_CHECK(tensor.dim() == 1 && tensor.numel() == D,
                "prediction metadata shape mismatch");
  for (const auto& tensor : {gc, ga, crowd, areas})
    TORCH_CHECK(tensor.dim() == 1 && tensor.numel() == G,
                "target metadata shape mismatch");
}

}  // namespace

std::vector<torch::Tensor> instance_update_cuda(torch::Tensor pm, torch::Tensor gm,
    torch::Tensor pc, torch::Tensor gc, torch::Tensor pa, torch::Tensor ga,
    torch::Tensor crowd, torch::Tensor areas, int64_t C) {
  TORCH_CHECK(pm.is_cuda() && pm.dim() == 3 && gm.dim() == 3 && C > 0 && C <= INT_MAX / 120,
              "invalid instance masks or category count");
  c10::cuda::CUDAGuard guard(pm.device());
  check(pm, pm, torch::kBool);
  check(gm, pm, torch::kBool);
  TORCH_CHECK(pm.size(1) == gm.size(1) && pm.size(2) == gm.size(2) &&
              pm.size(1) > 0 && pm.size(2) > 0, "instance mask dimensions must match");
  check_metadata(pm, pc, gc, pa, ga, crowd, areas, pm.size(0), gm.size(0));
  int D = pm.size(0), G = gm.size(0);
  auto ious = torch::empty({D, G}, pa.options());
  int64_t pixels = pm.size(1) * pm.size(2), words = (pixels + 31) / 32;
  auto pp = torch::empty({D, words}, pc.options()), gp = torch::empty({G, words}, pc.options());
  auto stream = at::cuda::getCurrentCUDAStream();
  if (D)
    pack_masks<<<(D * words + 255) / 256, 256, 0, stream>>>(
        pm.data_ptr<bool>(), pp.data_ptr<int32_t>(), D, pixels, words);
  if (G)
    pack_masks<<<(G * words + 255) / 256, 256, 0, stream>>>(
        gm.data_ptr<bool>(), gp.data_ptr<int32_t>(), G, pixels, words);
  if (D && G)
    overlap<<<D * G, 256, 0, stream>>>(pp.data_ptr<int32_t>(), gp.data_ptr<int32_t>(),
        pc.data_ptr<int32_t>(), gc.data_ptr<int32_t>(), pa.data_ptr<double>(),
        ga.data_ptr<double>(), crowd.data_ptr<bool>(), ious.data_ptr<double>(), D, G, words);
  return match_instances(ious, pc, gc, pa, crowd, areas, C);
}

std::vector<torch::Tensor> instance_compute_cuda(torch::Tensor classes, torch::Tensor ranks,
    torch::Tensor flags, torch::Tensor counts) {
  TORCH_CHECK(classes.is_cuda(), "instance accumulation requires CUDA");
  c10::cuda::CUDAGuard guard(classes.device());
  check(classes, classes, torch::kInt32);
  check(ranks, classes, torch::kInt32);
  check(flags, classes, torch::kInt8);
  check(counts, classes, torch::kInt64);
  TORCH_CHECK(classes.dim() == 1 && ranks.sizes() == classes.sizes() &&
              flags.dim() == 2 && flags.size(0) == 40 && flags.size(1) == classes.numel() &&
              counts.dim() == 2 && counts.size(1) == 4 && counts.size(0) > 0 &&
              counts.size(0) <= INT_MAX / 120, "invalid instance state shapes");
  int C = counts.size(0);
  auto options = counts.options().dtype(torch::kFloat64);
  auto precision = torch::full({10, 101, C, 4, 3}, -1., options);
  auto recall = torch::full({10, C, 4, 3}, -1., options);
  accumulate<<<(C * 120 + 63) / 64, 64, 0, at::cuda::getCurrentCUDAStream()>>>(
      classes.data_ptr<int32_t>(), ranks.data_ptr<int32_t>(), flags.data_ptr<int8_t>(),
      counts.data_ptr<int64_t>(), precision.data_ptr<double>(), recall.data_ptr<double>(),
      classes.numel(), C);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {precision, recall};
}

torch::Tensor instance_histogram_cuda(torch::Tensor masks, torch::Tensor target,
    torch::Tensor order, int64_t S) {
  TORCH_CHECK(masks.is_cuda() && masks.dim() == 3 && target.dim() == 2 &&
              S > 0 && S < INT_MAX, "invalid shared ground-truth inputs");
  c10::cuda::CUDAGuard guard(masks.device());
  check(masks, masks, torch::kBool);
  check(target, masks, torch::kInt32);
  check(order, masks, torch::kInt64);
  TORCH_CHECK(target.size(0) == masks.size(1) && target.size(1) == masks.size(2) &&
              target.numel() > 0 && order.dim() == 1 && order.numel() == masks.size(0),
              "shared target and detection dimensions must match");
  int64_t pixels = target.numel(), D = masks.size(0), chunks = (pixels + 8191) / 8192;
  TORCH_CHECK(D <= INT_MAX && D * chunks <= INT_MAX && D * S <= INT_MAX,
              "too many shared detection histogram bins or blocks");
  auto pairs = torch::zeros({D, S}, masks.options().dtype(torch::kInt64));
  if (D) {
    auto stream = at::cuda::getCurrentCUDAStream();
    if (S <= 12288)
      detection_histogram<true><<<D * chunks, 256, S * sizeof(unsigned), stream>>>(
          masks.data_ptr<bool>(), target.data_ptr<int32_t>(), order.data_ptr<int64_t>(),
          pairs.data_ptr<int64_t>(), pixels, S, chunks);
    else
      detection_histogram<false><<<D * chunks, 256, 0, stream>>>(
          masks.data_ptr<bool>(), target.data_ptr<int32_t>(), order.data_ptr<int64_t>(),
          pairs.data_ptr<int64_t>(), pixels, S, chunks);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return pairs;
}

std::vector<torch::Tensor> instance_match_histogram_cuda(torch::Tensor pairs,
    torch::Tensor pc, torch::Tensor gc, torch::Tensor pa, torch::Tensor ga,
    torch::Tensor crowd, torch::Tensor areas, int64_t C) {
  TORCH_CHECK(pairs.is_cuda() && pairs.dim() == 2 && C > 0 && C <= INT_MAX / 120,
              "invalid shared instance intersections or category count");
  c10::cuda::CUDAGuard guard(pairs.device());
  check(pairs, pairs, torch::kInt64);
  int64_t D = pairs.size(0), G = pairs.size(1);
  check_metadata(pairs, pc, gc, pa, ga, crowd, areas, D, G);
  auto ious = torch::empty({D, G}, pa.options());
  auto stream = at::cuda::getCurrentCUDAStream();
  if (D && G)
    histogram_iou<<<(D * G + 255) / 256, 256, 0, stream>>>(pairs.data_ptr<int64_t>(),
        pa.data_ptr<double>(), ga.data_ptr<double>(), crowd.data_ptr<bool>(),
        ious.data_ptr<double>(), D * G, G);
  return match_instances(ious, pc, gc, pa, crowd, areas, C);
}
