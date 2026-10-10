#include <torch/extension.h>
#include <vector>

int update_cuda(torch::Tensor gt, torch::Tensor pred, torch::Tensor gc,
    torch::Tensor pc, torch::Tensor crowd, torch::Tensor pairs, torch::Tensor areas,
    torch::Tensor matched, torch::Tensor tp, torch::Tensor fp, torch::Tensor fn,
    torch::Tensor iou_sum, torch::Tensor confusion, c10::optional<torch::Tensor> gt_ids,
    c10::optional<torch::Tensor> gt_slots, c10::optional<torch::Tensor> pred_ids,
    c10::optional<torch::Tensor> pred_slots, torch::Tensor error, bool validate, bool reward);
std::vector<torch::Tensor> compute_cuda(torch::Tensor tp, torch::Tensor fp,
    torch::Tensor fn, torch::Tensor sums, torch::Tensor confusion, torch::Tensor things);
std::vector<torch::Tensor> pack_cuda(torch::Tensor maps, torch::Tensor ids, torch::Tensor slots);
std::vector<torch::Tensor> instance_update_cuda(torch::Tensor pm, torch::Tensor gm,
    torch::Tensor pc, torch::Tensor gc, torch::Tensor pa, torch::Tensor ga,
    torch::Tensor crowd, torch::Tensor areas, int64_t C);
std::vector<torch::Tensor> instance_compute_cuda(torch::Tensor classes, torch::Tensor ranks,
    torch::Tensor flags, torch::Tensor counts);
torch::Tensor instance_histogram_cuda(torch::Tensor masks, torch::Tensor target,
    torch::Tensor order, int64_t S);
std::vector<torch::Tensor> instance_match_histogram_cuda(torch::Tensor pairs,
    torch::Tensor pc, torch::Tensor gc, torch::Tensor pa, torch::Tensor ga,
    torch::Tensor crowd, torch::Tensor areas, int64_t C);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("update", &update_cuda, "Count intersections, validate and accumulate metrics",
    pybind11::arg("gt"),
    pybind11::arg("pred"),
    pybind11::arg("gc"),
    pybind11::arg("pc"),
    pybind11::arg("crowd"),
    pybind11::arg("pairs"),
    pybind11::arg("areas"),
    pybind11::arg("matched"),
    pybind11::arg("tp"),
    pybind11::arg("fp"),
    pybind11::arg("fn"),
    pybind11::arg("iou_sum"),
    pybind11::arg("confusion"),
    pybind11::arg("gt_ids"),
    pybind11::arg("gt_slots"),
    pybind11::arg("pred_ids"),
    pybind11::arg("pred_slots"),
    pybind11::arg("error"),
    pybind11::arg("validate"),
    pybind11::arg("reward") = false);
  m.def("compute", &compute_cuda, "Compute metric summaries and independent snapshots");
  m.def("pack", &pack_cuda, "Materialize compact slots with one pixel kernel");
  m.def("instance_update", &instance_update_cuda, "Mask IoU and COCO greedy matching");
  m.def("instance_compute", &instance_compute_cuda, "COCO AP and AR accumulation");
  m.def("instance_histogram", &instance_histogram_cuda, "Count detection overlaps with shared target slots");
  m.def("instance_match_histogram", &instance_match_histogram_cuda, "COCO matching from shared target intersections");
}
