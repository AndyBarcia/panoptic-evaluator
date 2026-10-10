# panoptic-evaluator

Evaluate panoptic quality (PQ/SQ/RQ), semantic mIoU, and COCO-style instance
segmentation AP/AR with PyTorch tensors on CUDA or CPU. Accumulate batches with
`update()`, get dataset metrics with `compute()`, and clear them with `reset()`.

## Quick start

Compute all three kinds of metrics with separate predictions and shared ground
truth:

```python
import torch
from panoptic_evaluator import InstanceBatch, PanopticBatch, SegmentationEvaluator

device = "cuda"
maps = torch.tensor([[[0, 1, 1, 2]]], dtype=torch.int32, device=device)
classes = torch.tensor([[-1, 0, 1]], dtype=torch.int32, device=device)
target = PanopticBatch(maps, classes)
prediction = PanopticBatch(maps.clone(), classes.clone())
detections = InstanceBatch(
    masks=[torch.tensor([[[False, True, True, False]]], device=device)],
    classes=[torch.tensor([0], device=device)],
    scores=[torch.tensor([0.9], device=device)],
)

metrics = SegmentationEvaluator(2, isthing=[True, False], device=device)
metrics.update(prediction, detections, target)  # Repeat for subsequent batches.
result = metrics.compute()
print(result["All"]["pq"].item(), result["miou"].item(),
      result["instance"]["ap"].item())
metrics.reset()
```

Use `device="cpu"` and CPU tensors for CPU evaluation. All inputs must be on the
evaluator's device. Scores are fractions, not percentages. Accumulate the whole
dataset before computing metrics; averaging batch scores gives different results.
Results remain valid after later updates or resets.

## API

Category indices run from 0 to `num_classes - 1`. Category 0 is valid.
`isthing` identifies thing categories and defaults to all true. Validation is
on by default; use `validate=False` for trusted inputs.

### Panoptic inputs

`PanopticBatch(segment_map, classes, crowd=None)`:

| Input | Shape and dtype | Meaning |
| --- | --- | --- |
| `segment_map` | `B × H × W`, int32 | Segment slots; 0 is void. |
| `classes` | `B × S`, int32 | Category per slot; void and unused slots use -1. |
| `crowd` | Optional `B × S`, bool | Ground-truth crowd flags. |

Each active segment must have pixels. Prediction and target dimensions must
match, but their segment counts may differ. Batch images of the same size, or
pad maps with void.

For original COCO segment IDs, use:

```python
target = PanopticBatch.from_coco(gt_maps, gt_segments_info, category_ids=[7, 42])
```

Each metadata entry needs `id`, `category_id`, and optional `iscrowd`.
`category_ids` defines the category order in results. Maps contain integer IDs;
decode RGB PNGs as `R + 256*G + 65536*B`. IDs must be positive and below
`INT64_MAX`; 0 is void. Use `compact=False` to keep original IDs, `compact()` to
convert to slots, or `with_maps(next_maps)` to reuse metadata with the same
batch size, device, and active segment IDs.

### Instance inputs

`InstanceBatch(masks, classes, scores=None, crowd=None, areas=None)` accepts a
list of tensors for each field, with one tensor per image:

| Input | Shape and dtype per image | Meaning |
| --- | --- | --- |
| `masks` | `N × H × W`, bool | Individual masks; overlaps are allowed. |
| `classes` | `N`, int32/int64 | Category per instance. |
| `scores` | `N`, float | Required prediction confidence scores. |
| `crowd` | Optional `N`, bool | Ground-truth crowd flags. |
| `areas` | Optional `N`, float | Ground-truth annotation areas for AP filtering. |

Instance counts and image sizes may vary across images. Each instance mask must
have pixels. Images without instances still need a `0 × H × W` mask tensor and
empty metadata vectors. Scores must be finite; annotation areas must be finite
and nonnegative. Areas otherwise come from masks.

### Combined evaluation

`SegmentationEvaluator(num_classes, *, isthing=None, device="cuda", validate=True)`:

- `update(prediction, detections, target, *, instance_areas=None)` accepts
  panoptic predictions, scored instance predictions, and one panoptic target.
- `compute()` returns panoptic/semantic results plus an `instance` dictionary.
- `reset()` clears all metrics.

Use this evaluator when instance ground truth is exactly the thing segments of
the panoptic target. Detection masks can differ from panoptic predictions and
may overlap. Detection categories must be things. All three inputs must have
matching batch sizes and image dimensions. Crowd flags come from the target.

Optional `instance_areas` is a floating point `B × S` tensor in target metadata
slot order, used only for AP area filtering. IoU always uses pixel areas. AP
arrays keep the full category order; stuff categories have unavailable values.

### Panoptic and semantic evaluation

`PanopticEvaluator(num_classes, *, isthing=None, device="cuda", validate=True)`:

```python
from panoptic_evaluator import PanopticEvaluator

metrics = PanopticEvaluator(2, isthing=[True, False], device=device)
metrics.update(prediction, target)
result = metrics.compute()
```

The result contains:

- `All`, `Things`, `Stuff`: `pq`, `sq`, `rq`, and `n` (scored category count).
- `per_class`: category vectors for `pq`, `sq`, `rq`, and semantic `iou`.
- `miou`: semantic mean IoU.
- `tp`, `fp`, `fn`, `iou_sum`, `confusion`: accumulated statistics.

PQ follows the [COCO panoptic evaluator](https://github.com/cocodataset/panopticapi/blob/master/panopticapi/evaluation.py),
including void/crowd handling and an IoU threshold strictly greater than 0.5.
Stuff segments are evaluated separately. Semantic mIoU excludes ground-truth
void and crowd pixels; predicted void on valid ground truth counts as an error.
Categories with no semantic union have NaN IoU and are excluded from mIoU.
Empty PQ and mIoU summaries return zero.

### Per-image panoptic quality

`panoptic_quality(prediction, target, num_classes, *, validate=True)` returns
one mean-class PQ score per image as a float64 tensor on the input device:

```python
from panoptic_evaluator import panoptic_quality

scores = panoptic_quality(prediction, target, 2)
```

This stateless API supports prepared metadata reused with `with_maps()` and
ignores prediction slots without pixels, including fully occluded segments.
Only `reduction="none"` is supported. Use `PanopticEvaluator` for dataset PQ
and semantic metrics.

### Instance evaluation

`InstanceEvaluator(num_classes, *, device="cpu", validate=True)`:

```python
from panoptic_evaluator import InstanceBatch, InstanceEvaluator

prediction = InstanceBatch(masks, classes, scores=scores)
target = InstanceBatch(gt_masks, gt_classes, crowd=gt_crowd)
metrics = InstanceEvaluator(num_classes, device=device)
metrics.update(prediction, target)
result = metrics.compute()
print(result["ap"].item(), result["ap50"].item())
```

Use this evaluator for separate instance ground truth, including overlapping
annotations. It follows the default segmentation settings of
[COCOeval](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py):
IoU thresholds 0.50–0.95, 101 recall points, all/small/medium/large area ranges,
and maximum detection counts of 1, 10, and 100 per image/category.

Results include `ap`, `ap50`, `ap75`, `ap_small`, `ap_medium`, `ap_large`, `ar1`,
`ar10`, `ar100`, `ar_small`, `ar_medium`, and `ar_large`. Full `precision`
(`10 × 101 × C × 4 × 3`) and `recall` (`10 × C × 4 × 3`) arrays are also
returned. Unavailable settings have a value of -1.

## Build and test

CPU evaluation requires PyTorch. For CUDA, build with a matching CUDA toolkit.
From this directory, use the provided container:

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

Tests require NumPy and Pillow. Install `pycocotools` and
[`panopticapi`](https://github.com/cocodataset/panopticapi) for optional reference
comparisons; evaluation itself does not require them.

## Benchmarks

Run these commands inside the same container:

```bash
python -m examples.benchmark --batch-size 8 --segments 64
python -m examples.benchmark_instance
python -m examples.benchmark_combined
```

The instance comparison requires pycocotools. Add `--validate` to include input
validation. The combined comparison checks results against separate evaluators.

Recorded combined evaluation times on an A100 80GB PCIe, with eight images,
15 categories, and validation disabled:

| Image size | GT segments / detections per image | Combined ms/batch | Separate ms/batch |
| --- | ---: | ---: | ---: |
| 256 × 384 | 64 / 64 | 3.776 | 3.964 |
| 512 × 768 | 256 / 256 | 5.350 | 28.344 |

Times are medians over seven rounds of reset, update, and compute. Input
preparation is excluded. These synthetic measurements depend on workload and
hardware.
