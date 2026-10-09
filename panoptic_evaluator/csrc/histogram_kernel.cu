#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <vector>
#include <cstdlib>
#include <string>

namespace {
using Count = unsigned long long;

// Combine identical histogram keys within each warp before global atomics.
template <typename T>
__device__ void count(T* output, int64_t key) {
#if __CUDA_ARCH__ >= 700
  unsigned peers = __match_any_sync(__activemask(), key);
  if ((threadIdx.x & 31) == __ffs(peers) - 1)
    atomicAdd(output + key, static_cast<T>(__popc(peers)));
#else
  atomicAdd(output + key, static_cast<T>(1));
#endif
}

// Lookup tables are prepared once on the host and shared across all pixel blocks.
__device__ int resolve(int64_t key, const int64_t* ids, const int* slots, int S) {
  if (!ids) return key >= 0 && key < S ? static_cast<int>(key) : -1;
  if (key == 0) return 0;
  if (key < 0 || key == INT64_MAX) return -1;
  int lo = 1, hi = S;
  while (lo < hi) {
    int mid = lo + (hi - lo) / 2;
    if (ids[mid] < key) lo = mid + 1;
    else hi = mid;
  }
  if (lo == S || ids[lo] != key) return -1;
  int slot = slots[lo];
  return slot > 0 && slot < S ? slot : -1;
}

template <typename GT, typename Pred, bool Lookup>
__global__ void histogram(const GT* gt, const Pred* pred, Count* pairs,
    int64_t pixels, int G, int P, int64_t total,
    const int64_t* gi, const int* gs, const int64_t* pi, const int* ps, int* error) {
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < total; i += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    int b = i / pixels;
    int g = Lookup ? resolve(gt[i], gi ? gi + static_cast<int64_t>(b) * G : nullptr,
                            gs ? gs + static_cast<int64_t>(b) * G : nullptr, G)
                   : resolve(gt[i], nullptr, nullptr, G);
    int p = Lookup ? resolve(pred[i], pi ? pi + static_cast<int64_t>(b) * P : nullptr,
                            ps ? ps + static_cast<int64_t>(b) * P : nullptr, P)
                   : resolve(pred[i], nullptr, nullptr, P);
    if (g >= 0 && p >= 0) count(pairs, (static_cast<int64_t>(b) * G + g) * P + p);
    else if (error) atomicOr(error, 1);
  }
}

// Each block processes at most 8192 pixels from one image, so local uint32
// counts cannot overflow. Dataset and global pair counts remain uint64.
template <typename GT, typename Pred>
__global__ void histogram_shared(const GT* gt, const Pred* pred, Count* pairs,
    int64_t pixels, int G, int P, const int64_t* gi, const int* gs,
    const int64_t* pi, const int* ps, int* error, int chunks) {
  extern __shared__ unsigned bins[];
  int b = blockIdx.x / chunks;
  int chunk = blockIdx.x % chunks;
  int size = G * P;
  for (int k = threadIdx.x; k < size; k += blockDim.x) bins[k] = 0;
  __syncthreads();
  int64_t begin = static_cast<int64_t>(chunk) * 8192;
  int64_t end = min(begin + 8192, pixels);
  for (int64_t x = begin + threadIdx.x; x < end; x += blockDim.x) {
    int64_t i = static_cast<int64_t>(b) * pixels + x;
    int g = resolve(gt[i], gi ? gi + static_cast<int64_t>(b) * G : nullptr,
                    gs ? gs + static_cast<int64_t>(b) * G : nullptr, G);
    int p = resolve(pred[i], pi ? pi + static_cast<int64_t>(b) * P : nullptr,
                    ps ? ps + static_cast<int64_t>(b) * P : nullptr, P);
    if (g >= 0 && p >= 0) count(bins, g * P + p);
    else if (error) atomicOr(error, 1);
  }
  __syncthreads();
  pairs += static_cast<int64_t>(b) * size;
  for (int k = threadIdx.x; k < size; k += blockDim.x)
    if (bins[k]) atomicAdd(pairs + k, static_cast<Count>(bins[k]));
}

template <typename Map>
__global__ void pack(const Map* maps, const int64_t* ids, const int* slots,
    int* output, int* error, int64_t pixels, int S, int64_t total) {
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < total; i += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    int64_t offset = (i / pixels) * S;
    int slot = resolve(maps[i], ids + offset, slots + offset, S);
    output[i] = slot >= 0 ? slot : 0;
    if (slot < 0) atomicOr(error, 1);
  }
}

// Area checks share the exact reductions needed by scoring. No extra pixel pass.
__device__ void prepare_image(const Count* pairs, const int* gc, const int* pc,
    const bool* crowd, Count* areas, int* matched, int G, int P, int C, int* error, bool reward = false) {
  int lane = threadIdx.x & 31;
  int warp = threadIdx.x / 32;
  // Small rows retain the cheaper thread-per-row reduction.
  bool cooperative = P >= 128;
  for (int g = cooperative ? warp : threadIdx.x; g < G;
       g += cooperative ? blockDim.x / 32 : blockDim.x) {
    Count sum = 0;
    for (int p = cooperative ? lane : 0; p < P; p += cooperative ? 32 : 1)
      sum += pairs[static_cast<int64_t>(g) * P + p];
    if (cooperative) {
      for (int offset = 16; offset; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
      if (lane != 0) continue;
    }
    areas[g] = sum;
    matched[g] = 0;
    if (error) {
      if (gc[g] < -1 || gc[g] >= C || (g == 0 && (gc[g] != -1 || crowd[g]))) atomicOr(error, 2);
      if (g > 0 && ((sum > 0) != (gc[g] >= 0))) atomicOr(error, 4);
    }
  }
  for (int p = threadIdx.x; p < P; p += blockDim.x) {
    Count sum = 0;
    for (int g = 0; g < G; ++g) sum += pairs[static_cast<int64_t>(g) * P + p];
    areas[G + p] = sum;
    if (error) {
      if (pc[p] < -1 || pc[p] >= C || (p == 0 && pc[p] != -1)) atomicOr(error, 2);
      if (p > 0 && (reward ? (sum > 0 && pc[p] < 0) : ((sum > 0) != (pc[p] >= 0)))) atomicOr(error, 4);
    }
  }
  __syncthreads();
}

__global__ void prepare(const Count* pairs, const int* gc, const int* pc,
    const bool* crowd, Count* areas, int* matched, int G, int P, int C, int* error, bool reward = false) {
  int64_t b = blockIdx.x;
  prepare_image(pairs + b * G * P, gc + b * G, pc + b * P, crowd + b * G,
                areas + b * (G + P), matched + b * G, G, P, C, error, reward);
}

// One block owns an image, so barriers safely connect areas, matches and FN.
// Only linear scratch is needed; no union, IoU or match matrices are materialized.
template <bool Multi = false>
__global__ void accumulate(const Count* pairs, const int* gc, const int* pc,
    const bool* crowd, Count* areas, int* matched, Count* tp, Count* fp,
    Count* fn, double* iou_sum, Count* confusion, int G, int P, int C, bool areas_ready, bool reward = false) {
  int chunks = (P + blockDim.x - 1) / blockDim.x;
  int b = Multi ? blockIdx.x / chunks : blockIdx.x, t = threadIdx.x;
  pairs += static_cast<int64_t>(b) * G * P;
  gc += static_cast<int64_t>(b) * G;
  pc += static_cast<int64_t>(b) * P;
  crowd += static_cast<int64_t>(b) * G;
  areas += static_cast<int64_t>(b) * (G + P);
  matched += static_cast<int64_t>(b) * G;
  if (reward) {
    tp += static_cast<int64_t>(b) * C; fp += static_cast<int64_t>(b) * C;
    fn += static_cast<int64_t>(b) * C; iou_sum += static_cast<int64_t>(b) * C;
  }
  if (!areas_ready) prepare_image(pairs, gc, pc, crowd, areas, matched, G, P, C, nullptr);
  for (int p = Multi ? (blockIdx.x % chunks) * blockDim.x + t : t;
       p < P; p += Multi ? P : blockDim.x) {
    int prediction = pc[p];
    bool valid_prediction = prediction >= 0 && prediction < C;
    bool pm = false;
    double sum_iou = 0;
    Count ignored = pairs[p];  // void overlap
    for (int g = 0; g < G; ++g) {
      int target = gc[g];
      Count intersection = pairs[static_cast<int64_t>(g) * P + p];
      bool valid_target = target >= 0 && target < C;
      if (!reward && valid_target && !crowd[g] && intersection) {
        int label = valid_prediction ? prediction : C;
        atomicAdd(confusion + static_cast<int64_t>(target) * (C + 1) + label, intersection);
      }
      if (!valid_target || !valid_prediction || target != prediction) continue;
      if (crowd[g]) {
        // Overwrite to preserve the official API's last-crowd metadata rule.
        ignored = pairs[p] + intersection;
      } else if (intersection) {
        Count union_area = areas[g] + areas[G + p] - intersection - pairs[p];
        // Strict half threshold without doubling counts (which could overflow).
        if (intersection > union_area / 2) {
          double iou = static_cast<double>(intersection) / static_cast<double>(union_area);
          pm = true;
          sum_iou += iou;
          atomicExch(matched + g, 1);
        }
      }
    }
    if (valid_prediction && (!reward || areas[G + p] > 0)) {
      if (pm) {
        atomicAdd(tp + prediction, 1ULL);
        atomicAdd(iou_sum + prediction, sum_iou);
      } else if (static_cast<double>(ignored) <= 0.5 * areas[G + p]) {
        atomicAdd(fp + prediction, 1ULL);
      }
    }
  }
  if (Multi) return;
  __syncthreads();
  for (int g = t; g < G; g += blockDim.x) {
    if (gc[g] >= 0 && gc[g] < C && !crowd[g] && !matched[g])
      atomicAdd(fn + gc[g], 1ULL);
  }
}

__global__ void false_negatives(const int* gc, const bool* crowd,
    const int* matched, Count* fn, int64_t total, int C) {
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < total; i += static_cast<int64_t>(gridDim.x) * blockDim.x)
    if (gc[i] >= 0 && gc[i] < C && !crowd[i] && !matched[i])
      atomicAdd(fn + gc[i], 1ULL);
}

void require_tensor(const torch::Tensor& t, const torch::Tensor& reference,
                    torch::ScalarType dtype) {
  TORCH_CHECK(t.device() == reference.device() && t.is_contiguous() &&
              t.scalar_type() == dtype, "incorrect tensor device, layout or dtype");
}
}  // namespace

namespace {
const int64_t* lookup_ids(const c10::optional<torch::Tensor>& ids,
    const c10::optional<torch::Tensor>& slots, const torch::Tensor& classes,
    const torch::Tensor& reference) {
  TORCH_CHECK(ids.has_value() == slots.has_value(), "lookup IDs and slots must be supplied together");
  if (!ids) return nullptr;
  require_tensor(*ids, reference, torch::kInt64);
  require_tensor(*slots, reference, torch::kInt32);
  TORCH_CHECK(ids->sizes() == classes.sizes() && slots->sizes() == classes.sizes(), "incorrect lookup shape");
  return ids->data_ptr<int64_t>();
}

template <typename GT, typename Pred>
void launch_histogram(torch::Tensor gt, torch::Tensor pred, Count* pairs,
    int G, int P, const int64_t* gi, const int* gs, const int64_t* pi,
    const int* ps, int* error, cudaStream_t stream) {
  int blocks = std::min<int64_t>((gt.numel() + 255) / 256, 4096);
  const char* mode = std::getenv("PANOPTIC_HISTOGRAM");
  if (mode && std::string(mode) == "shared" && static_cast<int64_t>(G) * P <= 12288) {
    int64_t pixels = gt.size(1) * gt.size(2);
    int chunks = (pixels + 8191) / 8192;
    histogram_shared<GT, Pred><<<gt.size(0) * chunks, 256, G * P * sizeof(unsigned), stream>>>(
        gt.data_ptr<GT>(), pred.data_ptr<Pred>(), pairs, pixels, G, P, gi, gs, pi, ps, error, chunks);
    return;
  }
  if (gi || pi)
    histogram<GT, Pred, true><<<blocks, 256, 0, stream>>>(gt.data_ptr<GT>(), pred.data_ptr<Pred>(), pairs,
        gt.size(1) * gt.size(2), G, P, gt.numel(), gi, gs, pi, ps, error);
  else
    histogram<GT, Pred, false><<<blocks, 256, 0, stream>>>(gt.data_ptr<GT>(), pred.data_ptr<Pred>(), pairs,
        gt.size(1) * gt.size(2), G, P, gt.numel(), nullptr, nullptr, nullptr, nullptr, error);
}
}  // namespace

int update_cuda(torch::Tensor gt, torch::Tensor pred, torch::Tensor gc,
    torch::Tensor pc, torch::Tensor crowd, torch::Tensor pairs,
    torch::Tensor areas, torch::Tensor matched, torch::Tensor tp,
    torch::Tensor fp, torch::Tensor fn, torch::Tensor iou_sum,
    torch::Tensor confusion, c10::optional<torch::Tensor> gt_ids,
    c10::optional<torch::Tensor> gt_slots, c10::optional<torch::Tensor> pred_ids,
    c10::optional<torch::Tensor> pred_slots, torch::Tensor error, bool validate, bool reward) {
  TORCH_CHECK(gt.is_cuda() && gt.dim() == 3, "gt must be CUDA B x H x W");
  c10::cuda::CUDAGuard guard(gt.device());
  for (const auto& t : {gt, pred}) {
    TORCH_CHECK(t.device() == gt.device() && t.is_contiguous() &&
                (t.scalar_type() == torch::kInt32 || t.scalar_type() == torch::kInt64),
                "maps must be contiguous int32 or int64 on the same CUDA device");
  }
  for (const auto& t : {gc, pc, matched, error}) require_tensor(t, gt, torch::kInt32);
  for (const auto& t : {pairs, areas, tp, fp, fn, confusion}) require_tensor(t, gt, torch::kInt64);
  require_tensor(crowd, gt, torch::kBool);
  require_tensor(iou_sum, gt, torch::kFloat64);
  TORCH_CHECK(pred.sizes() == gt.sizes() && gc.dim() == 2 && pc.dim() == 2 &&
              gc.size(0) == gt.size(0) && pc.size(0) == gt.size(0) &&
              crowd.sizes() == gc.sizes(), "incorrect input shapes");
  int64_t B = gt.size(0), G = gc.size(1), P = pc.size(1), C = reward ? tp.size(1) : tp.numel();
  TORCH_CHECK(C > 0 && C < INT_MAX && G > 0 && G < INT_MAX && P > 0 && P < INT_MAX,
              "invalid capacities");
  TORCH_CHECK(pairs.dim() == 3 && pairs.size(0) == B && pairs.size(1) == G && pairs.size(2) == P &&
              areas.dim() == 2 && areas.size(0) == B && areas.size(1) == G + P &&
              matched.sizes() == gc.sizes() && error.numel() == 1, "incorrect scratch shapes");
  TORCH_CHECK((reward ? (tp.dim() == 2 && tp.size(0) == B) : tp.dim() == 1) && fp.sizes() == tp.sizes() && fn.sizes() == tp.sizes() &&
              iou_sum.sizes() == tp.sizes() && (reward || (confusion.dim() == 2 &&
              confusion.size(0) == C && confusion.size(1) == C + 1)), "incorrect state shapes");
  const int64_t* gi = lookup_ids(gt_ids, gt_slots, gc, gt);
  const int64_t* pi = lookup_ids(pred_ids, pred_slots, pc, gt);
  const int* gs = gt_slots ? gt_slots->data_ptr<int>() : nullptr;
  const int* ps = pred_slots ? pred_slots->data_ptr<int>() : nullptr;
  auto stream = at::cuda::getCurrentCUDAStream();
  Count* pair_data = reinterpret_cast<Count*>(pairs.data_ptr<int64_t>());
  Count* area_data = reinterpret_cast<Count*>(areas.data_ptr<int64_t>());
  if (pairs.numel()) C10_CUDA_CHECK(cudaMemsetAsync(pairs.data_ptr(), 0, pairs.nbytes(), stream));
  if (validate) C10_CUDA_CHECK(cudaMemsetAsync(error.data_ptr(), 0, error.nbytes(), stream));
  if (gt.numel()) {
    int* error_data = validate ? error.data_ptr<int>() : nullptr;
    if (gt.scalar_type() == torch::kInt32) {
      if (pred.scalar_type() == torch::kInt32)
        launch_histogram<int, int>(gt, pred, pair_data, G, P, gi, gs, pi, ps, error_data, stream);
      else launch_histogram<int, int64_t>(gt, pred, pair_data, G, P, gi, gs, pi, ps, error_data, stream);
    } else {
      if (pred.scalar_type() == torch::kInt32)
        launch_histogram<int64_t, int>(gt, pred, pair_data, G, P, gi, gs, pi, ps, error_data, stream);
      else launch_histogram<int64_t, int64_t>(gt, pred, pair_data, G, P, gi, gs, pi, ps, error_data, stream);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  const char* scoring = std::getenv("PANOPTIC_SCORING");
  bool multi = !reward && scoring && std::string(scoring) == "multi";
  if (validate || multi) {
    if (B) {
      prepare<<<B, 256, 0, stream>>>(pair_data, gc.data_ptr<int>(), pc.data_ptr<int>(), crowd.data_ptr<bool>(),
          area_data, matched.data_ptr<int>(), G, P, C, validate ? error.data_ptr<int>() : nullptr, reward);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    // Exactly one scalar transfer/synchronization; errors cannot mutate totals.
    if (validate) {
      int code = error.cpu().item<int>();
      if (code) return code;
    }
  }
  if (B) {
    if (multi) {
      accumulate<true><<<B * ((P + 63) / 64), 64, 0, stream>>>(pair_data, gc.data_ptr<int>(), pc.data_ptr<int>(), crowd.data_ptr<bool>(),
          area_data, matched.data_ptr<int>(), reinterpret_cast<Count*>(tp.data_ptr<int64_t>()),
          reinterpret_cast<Count*>(fp.data_ptr<int64_t>()), reinterpret_cast<Count*>(fn.data_ptr<int64_t>()),
          iou_sum.data_ptr<double>(), reinterpret_cast<Count*>(confusion.data_ptr<int64_t>()), G, P, C, true);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      false_negatives<<<std::min<int64_t>((B * G + 255) / 256, 4096), 256, 0, stream>>>(gc.data_ptr<int>(),
          crowd.data_ptr<bool>(), matched.data_ptr<int>(), reinterpret_cast<Count*>(fn.data_ptr<int64_t>()), B * G, C);
    } else {
    accumulate<false><<<B, 256, 0, stream>>>(pair_data, gc.data_ptr<int>(), pc.data_ptr<int>(), crowd.data_ptr<bool>(),
        area_data, matched.data_ptr<int>(), reinterpret_cast<Count*>(tp.data_ptr<int64_t>()),
        reinterpret_cast<Count*>(fp.data_ptr<int64_t>()), reinterpret_cast<Count*>(fn.data_ptr<int64_t>()),
        iou_sum.data_ptr<double>(), reinterpret_cast<Count*>(confusion.data_ptr<int64_t>()), G, P, C, validate, reward);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return 0;
}

std::vector<torch::Tensor> pack_cuda(torch::Tensor maps, torch::Tensor ids, torch::Tensor slots) {
  TORCH_CHECK(maps.is_cuda() && maps.dim() == 3 && maps.is_contiguous() &&
              (maps.scalar_type() == torch::kInt32 || maps.scalar_type() == torch::kInt64),
              "maps must be contiguous CUDA int32 or int64 B x H x W");
  c10::cuda::CUDAGuard guard(maps.device());
  require_tensor(ids, maps, torch::kInt64);
  require_tensor(slots, maps, torch::kInt32);
  TORCH_CHECK(ids.dim() == 2 && slots.sizes() == ids.sizes() && ids.size(0) == maps.size(0) &&
              ids.size(1) > 0 && ids.size(1) < INT_MAX, "incorrect lookup shapes");
  auto output = torch::empty_like(maps, maps.options().dtype(torch::kInt32));
  auto error = torch::zeros({1}, maps.options().dtype(torch::kInt32));
  if (maps.numel()) {
    int blocks = std::min<int64_t>((maps.numel() + 255) / 256, 4096);
    auto stream = at::cuda::getCurrentCUDAStream();
    int64_t pixels = maps.size(1) * maps.size(2);
    if (maps.scalar_type() == torch::kInt32)
      pack<<<blocks, 256, 0, stream>>>(maps.data_ptr<int>(), ids.data_ptr<int64_t>(), slots.data_ptr<int>(),
          output.data_ptr<int>(), error.data_ptr<int>(), pixels, ids.size(1), maps.numel());
    else pack<<<blocks, 256, 0, stream>>>(maps.data_ptr<int64_t>(), ids.data_ptr<int64_t>(), slots.data_ptr<int>(),
          output.data_ptr<int>(), error.data_ptr<int>(), pixels, ids.size(1), maps.numel());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {output, error};
}
