# panoptic-evaluator

A PyTorch evaluator for batched panoptic segmentation and semantic segmentation.
It computes panoptic quality (PQ), segmentation quality (SQ), recognition quality
(RQ), and semantic mean intersection over union (mIoU), accumulating results
across batches. Evaluation supports CUDA and CPU tensors.

## Build and test

CUDA evaluation requires PyTorch and a matching CUDA toolkit. From this directory,
build the extension and run the tests in the provided container:

```bash
apptainer exec --nv \
  --bind "$PWD:/workspace/panoptic-evaluator" \
  --pwd /workspace/panoptic-evaluator \
  /data/Andy/fcclip-torch27-cu126.sif \
  python setup.py build_ext --inplace

apptainer exec --nv \
  --bind "$PWD:/workspace/panoptic-evaluator" \
  --pwd /workspace/panoptic-evaluator \
  /data/Andy/fcclip-torch27-cu126.sif \
  python -m unittest discover -s tests -v
```

CPU evaluation works without compiling the extension. Tests require NumPy and
Pillow. To enable comparisons against the official COCO evaluator, install
`panopticapi` in the test environment:

```bash
pip install git+https://github.com/cocodataset/panopticapi.git
```

## Quick start

```python
import torch
from panoptic_evaluator import PanopticBatch, PanopticEvaluator

# Segment slots are local to each image; slot 0 is void.
maps = torch.tensor([[[0, 1, 1, 2]]], device="cuda", dtype=torch.int32)
classes = torch.tensor([[-1, 0, 1]], device="cuda", dtype=torch.int32)
target = PanopticBatch(maps, classes)
prediction = PanopticBatch(maps.clone(), classes.clone())

metrics = PanopticEvaluator(2, isthing=[True, False])
metrics.update(prediction, target)  # Repeat for subsequent batches.
result = metrics.compute()
print(result["All"]["pq"].item(), result["miou"].item())
metrics.reset()
```

For CPU evaluation, create tensors on the CPU and pass `device="cpu"` to the
evaluator.

## API

### `PanopticBatch(segment_map, classes, crowd=None)`

| Input | Shape and type | Meaning |
| --- | --- | --- |
| `segment_map` | `B × H × W`, `torch.int32` | Per-pixel segment slots; 0 is void. |
| `classes` | `B × S`, `torch.int32` | Category for each slot; slot 0 and unused padding use -1. |
| `crowd` | Optional `B × S`, `torch.bool` | Ground-truth crowd flags. |

Categories are contiguous indices from 0 to `num_classes - 1`. Category 0 is a
valid category. Each active segment must have pixels. Prediction and target
maps must have matching image dimensions, but may have different numbers of
segment slots. All inputs must be on the evaluator's device. Group differently
sized images into separate batches, or pad both maps with void.

### `PanopticEvaluator(num_classes, *, isthing=None, device="cuda", validate=True)`

- `update(prediction, target)`: accumulate statistics for a batch.
- `compute()`: return metric tensors on the evaluator's device.
- `reset()`: clear accumulated statistics.

`isthing` identifies thing categories; it defaults to all true. Input validation
is enabled by default. Set `validate=False` for trusted inputs to reduce CUDA
synchronization overhead.

`compute()` returns:

- `All`, `Things`, `Stuff`: summaries containing `pq`, `sq`, `rq`, and `n`
  (the number of scored classes).
- `per_class`: vectors for `pq`, `sq`, `rq`, and semantic `iou`.
- `miou`: semantic mean IoU.
- `tp`, `fp`, `fn`, `iou_sum`, `confusion`: accumulated statistics.

Scores are fractions, not percentages. Results remain valid after subsequent
updates or resets. Accumulate all batches before computing dataset metrics;
averaging batch scores gives different results.

### COCO metadata and arbitrary segment IDs

Use `PanopticBatch.from_coco(maps, segments_info, category_ids, compact=True)`
for maps whose pixel values are original segment IDs:

```python
# Integer B × H × W maps; 0 is void.
# Each metadata entry has id, category_id, and optional iscrowd.
category_ids = [7, 42]
target = PanopticBatch.from_coco(gt_maps, gt_segments_info, category_ids)
prediction = PanopticBatch.from_coco(pred_maps, pred_segments_info, category_ids)

metrics = PanopticEvaluator(2, isthing=[True, False])
metrics.update(prediction, target)
```

`category_ids` defines the category order in results. Maps contain integer IDs,
not RGB pixels; decode COCO PNGs as `R + 256*G + 65536*B`. Segment IDs must be
positive and below `INT64_MAX`. Areas are counted from maps.

Pass `compact=False` to evaluate original-ID maps without first creating compact
slot maps. Use `batch.with_maps(next_maps)` to reuse metadata when the batch size,
device, and active segment set are unchanged, or `batch.compact()` to convert to
compact slots.

## Metric behavior

PQ follows the [official COCO panoptic evaluator](https://github.com/cocodataset/panopticapi/blob/master/panopticapi/evaluation.py),
including same-category matching at IoU strictly greater than 0.5 and its void
and crowd handling. Segments are scored independently; stuff segments are not
merged automatically. PQ summaries exclude classes without TP, FP, or FN.

Semantic mIoU excludes ground-truth void and crowd pixels. Predicted void on
valid ground truth counts as a false negative. Classes with no semantic union
have NaN per-class IoU and are excluded from mIoU. Empty summaries return zero.

## Benchmarks

Run the resident-tensor benchmark in the same container:

```bash
apptainer exec --nv \
  --bind "$PWD:/workspace/panoptic-evaluator" \
  --pwd /workspace/panoptic-evaluator \
  /data/Andy/fcclip-torch27-cu126.sif \
  python -m examples.benchmark --batch-size 8 --segments 64
```

It reports CUDA time, synchronized host time, throughput, and peak allocated
memory. Packing, file I/O, and data generation are excluded. Add `--validate`
to include input value checks. With `panopticapi` installed, run
`python -m examples.compare_official` to compare against the official evaluator.

A recorded A100 80GB PCIe run with eight 512 × 768 images, 64 segments, and
15 classes measured these median times, including reset, update, and compute:

| Input path | ms/batch |
| --- | ---: |
| Resident compact tensors, validation disabled | 0.141 |
| Resident compact tensors, validated | 0.198 |
| Resident original IDs plus metadata preparation, validated | 1.426 |
| CPU original IDs plus transfer and metadata preparation, validated | 5.477 |

These are workload-specific measurements from five rounds of 20 batches.
Performance depends on segment counts, input preparation, validation, and
hardware. The evaluator is intended for hundreds of segments per image;
intersection memory grows with the product of prediction and target capacities.