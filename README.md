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
| `scores` | `N`, float | Required for AP/AR predictions; unused for generation prefixes. |
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

### Generation prefixes

`segmentation_prefixes(prediction, target, num_classes, *, isthing=None,
checkpoints=None, iou_thresholds=(0.5, 0.75), validate=True)` accepts an
`InstanceBatch` in generation order and a `PanopticBatch` target. Scores are
ignored. It returns one list per image, containing that image's metrics at each
requested prefix:

```python
from panoptic_evaluator import InstanceBatch, segmentation_prefixes

# Masks and classes are in generation order; scores are unnecessary.
sequence = InstanceBatch(masks, classes)
curves = segmentation_prefixes(
    sequence, target, num_classes=15, isthing=isthing,
    iou_thresholds=(0.5, 0.75),
    # checkpoints=[0, 1, 5, 10],  # Optional; otherwise evaluate every prefix.
)
for step in curves[0]:  # First image.
    print(step["prefix"], step["All"]["pq"], step["miou"],
          step["instance"]["precision"], step["instance"]["recall"])
```

Each image returns its own list of snapshots, allowing different sequence
lengths. Later masks overwrite earlier masks on the panoptic/semantic canvas;
each mask creates a distinct panoptic segment, including stuff masks. Fully
occluded segments disappear from canvas metrics. Instance metrics use the
original masks of thing classes, even after occlusion, and ignore stuff masks.

Instance matching proceeds in generation order independently at each IoU
threshold, using same-class GT with IoU **at least** the threshold. The best
available noncrowd match is preferred, followed by an ignored crowd match;
crowd targets are reusable and use intersection/detection-area IoU. Ties use
GT slot order. There is no confidence sorting or detection cap. This defines
generation-order precision/recall, rather than COCO AP/AR. It is greedy, not
maximum-cardinality matching: an early mask's match remains fixed at later
prefixes. Duplicate unmatched predictions count as false positives.

`instance` contains `iou_thresholds`, plus `tp`, `fp`, `fn`, `precision`, and
`recall` arrays of shape `thresholds × classes`. Zero denominators return zero.
All outputs are tensors on the target device, except the integer `prefix`.
Snapshots remain valid after processing later steps. Empty sequences return the
prefix-zero snapshot. Explicit checkpoints must be strictly increasing and valid
for every image; an empty checkpoint list returns no snapshots.

Panoptic and semantic results have the same fields as `PanopticEvaluator`.
These are per-image results: for dataset curves, sum counts, IoU sums, and
confusion matrices at each prefix before computing metrics; do not average
per-image ratios. Sum instance counts at each prefix before computing dataset
precision/recall as well.
