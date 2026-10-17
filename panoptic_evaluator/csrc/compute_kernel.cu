#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <vector>

namespace {
// Final summaries and independent result snapshots in one launch.
__global__ void compute(const int64_t* tp, const int64_t* fp, const int64_t* fn,
    const double* sums, const int64_t* confusion, const bool* things,
    double* per_class, double* summaries, int64_t* counts, double* miou,
    int64_t* out_tp, int64_t* out_fp, int64_t* out_fn, double* out_sums,
    int64_t* out_confusion, int C) {
  int64_t b = blockIdx.x;
  tp += b*C; fp += b*C; fn += b*C; sums += b*C;
  confusion += b*C*(C+1); per_class += b*4*C;
  summaries += b*9; counts += b*3; miou += b;
  out_tp += b*C; out_fp += b*C; out_fn += b*C; out_sums += b*C;
  out_confusion += b*C*(C+1);
  __shared__ double reduction[14][256];
  int t = threadIdx.x;
  double local[14] = {};
  for (int c = t; c < C; c += blockDim.x) {
    out_tp[c] = tp[c]; out_fp[c] = fp[c]; out_fn[c] = fn[c]; out_sums[c] = sums[c];
    double denominator = tp[c] + 0.5 * (static_cast<double>(fp[c]) + fn[c]);
    double pq = denominator > 0 ? sums[c] / denominator : 0;
    double sq = tp[c] ? sums[c] / tp[c] : 0;
    double rq = denominator > 0 ? tp[c] / denominator : 0;
    per_class[c] = pq;
    per_class[C + c] = sq;
    per_class[2 * C + c] = rq;
    int64_t row = 0, col = 0, diagonal = confusion[static_cast<int64_t>(c) * (C + 1) + c];
    for (int k = 0; k < C + 1; ++k) row += confusion[static_cast<int64_t>(c) * (C + 1) + k];
    for (int k = 0; k < C; ++k) col += confusion[static_cast<int64_t>(k) * (C + 1) + c];
    int64_t union_area = row + col - diagonal;
    double iou = union_area ? static_cast<double>(diagonal) / union_area : nan("");
    per_class[3 * C + c] = iou;
    if (union_area) { local[12] += iou; local[13] += 1; }
    if (denominator > 0) {
      for (int group = 0; group < 3; ++group) {
        if (group != 0 && group != (things[c] ? 1 : 2)) continue;
        local[group * 4] += pq; local[group * 4 + 1] += sq;
        local[group * 4 + 2] += rq; local[group * 4 + 3] += 1;
      }
    }
  }
  for (int64_t i = t; i < static_cast<int64_t>(C) * (C + 1); i += blockDim.x)
    out_confusion[i] = confusion[i];
  for (int k = 0; k < 14; ++k) reduction[k][t] = local[k];
  __syncthreads();
  for (int stride = 128; stride; stride >>= 1) {
    if (t < stride)
      for (int k = 0; k < 14; ++k) reduction[k][t] += reduction[k][t + stride];
    __syncthreads();
  }
  if (t < 3) {
    double n = reduction[t * 4 + 3][0];
    counts[t] = static_cast<int64_t>(n);
    for (int k = 0; k < 3; ++k)
      summaries[t * 3 + k] = n ? reduction[t * 4 + k][0] / n : 0;
  }
  if (t == 0) *miou = reduction[13][0] ? reduction[12][0] / reduction[13][0] : 0;
}
}  // namespace

std::vector<torch::Tensor> compute_cuda(torch::Tensor tp, torch::Tensor fp,
    torch::Tensor fn, torch::Tensor sums, torch::Tensor confusion, torch::Tensor things) {
  TORCH_CHECK(tp.is_cuda() && tp.dim() == 1 && tp.numel() > 0 && tp.numel() < INT_MAX,
              "tp must be nonempty CUDA category counts");
  c10::cuda::CUDAGuard guard(tp.device());
  for (const auto& t : {tp, fp, fn, sums, confusion, things})
    TORCH_CHECK(t.device() == tp.device() && t.is_contiguous(), "incorrect device or layout");
  for (const auto& t : {tp, fp, fn, confusion})
    TORCH_CHECK(t.scalar_type() == torch::kInt64, "counts must be int64");
  int64_t C = tp.numel();
  TORCH_CHECK(fp.sizes() == tp.sizes() && fn.sizes() == tp.sizes() && sums.sizes() == tp.sizes() &&
              things.sizes() == tp.sizes() && sums.scalar_type() == torch::kFloat64 &&
              things.scalar_type() == torch::kBool && confusion.dim() == 2 &&
              confusion.size(0) == C && confusion.size(1) == C + 1, "incorrect state shapes or dtypes");
  auto doubles = tp.options().dtype(torch::kFloat64);
  auto per_class = torch::empty({4, C}, doubles);
  auto summaries = torch::empty({3, 3}, doubles);
  auto counts = torch::empty({3}, tp.options());
  auto miou = torch::empty({}, doubles);
  auto out_tp = torch::empty_like(tp), out_fp = torch::empty_like(fp), out_fn = torch::empty_like(fn);
  auto out_sums = torch::empty_like(sums), out_confusion = torch::empty_like(confusion);
  compute<<<1, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      tp.data_ptr<int64_t>(), fp.data_ptr<int64_t>(), fn.data_ptr<int64_t>(), sums.data_ptr<double>(),
      confusion.data_ptr<int64_t>(), things.data_ptr<bool>(), per_class.data_ptr<double>(),
      summaries.data_ptr<double>(), counts.data_ptr<int64_t>(), miou.data_ptr<double>(),
      out_tp.data_ptr<int64_t>(), out_fp.data_ptr<int64_t>(), out_fn.data_ptr<int64_t>(),
      out_sums.data_ptr<double>(), out_confusion.data_ptr<int64_t>(), C);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {per_class, summaries, counts, miou, out_tp, out_fp, out_fn, out_sums, out_confusion};
}

// Prefix sufficient statistics are independent rows; summarize them in one launch.
std::vector<torch::Tensor> compute_prefix_cuda(torch::Tensor tp, torch::Tensor fp,
    torch::Tensor fn, torch::Tensor sums, torch::Tensor confusion, torch::Tensor things) {
  int64_t K = tp.size(0), C = tp.size(1);
  auto doubles = sums.options();
  auto values = torch::empty({K, 4, C}, doubles);
  auto summaries = torch::empty({K, 3, 3}, doubles);
  auto counts = torch::empty({K, 3}, tp.options());
  auto miou = torch::empty({K}, doubles);
  auto ot = torch::empty_like(tp), of = torch::empty_like(fp), on = torch::empty_like(fn);
  auto os = torch::empty_like(sums), oc = torch::empty_like(confusion);
  if (K) {
    compute<<<K, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      tp.data_ptr<int64_t>(), fp.data_ptr<int64_t>(), fn.data_ptr<int64_t>(), sums.data_ptr<double>(),
      confusion.data_ptr<int64_t>(), things.data_ptr<bool>(), values.data_ptr<double>(),
      summaries.data_ptr<double>(), counts.data_ptr<int64_t>(), miou.data_ptr<double>(),
      ot.data_ptr<int64_t>(), of.data_ptr<int64_t>(), on.data_ptr<int64_t>(), os.data_ptr<double>(),
      oc.data_ptr<int64_t>(), C);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {values, summaries, counts, miou, ot, of, on, os, oc};
}
