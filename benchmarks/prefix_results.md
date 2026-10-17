# Fused incremental prefix evaluation benchmark

The CUDA implementation now uses native incremental kernels. The earlier Python
loop over CUDA tensor operations has been replaced. API outputs and matching
semantics are unchanged.

## Environment and method

NVIDIA A100 80GB PCIe (physical GPU 4, idle before runs), AMD EPYC 7413 CPU,
PyTorch 2.7.0+cu126, and the repository CUDA container. The host is shared with
other jobs. CPU execution uses one PyTorch thread.

Each workload contains one image, five classes (three things and two stuff),
16 or 64 GT segments and equally many predictions. Dense predicted masks are GT
masks dilated with a 9×9 kernel; some are duplicates. Inputs reside on the
measurement device before timing. Generation is excluded. Compositing,
active-slot preparation, evaluation, validation when enabled, and Python result
construction are included.

Three paths are measured:

- **Fused prefix:** `segmentation_prefixes` returns PQ/SQ/RQ, semantic IoU, and
  independent-mask generation-order instance precision/recall at IoU .50 and .75.
- **Standard PR:** draw masks in order, then reset and update
  `SegmentationEvaluator` at each requested checkpoint. Compute panoptic/semantic
  metrics and derive endpoint instance precision/recall from standard matching
  records without AP interpolation. Strictly descending synthetic scores ensure
  the same generation-order matching. There are fewer than 100 detections per
  class, no crowd, and no all-area exclusions, so the instance definitions agree.
- **Standard AP:** the same baseline with full standard `compute()`, including
  COCO AP/AR across default thresholds, ranges, recall levels and detection caps.
  This computes additional metrics and is not the equivalent-metric comparison.

Every checkpoint is checked for panoptic/semantic and instance-count/ratio
parity between Fused prefix and Standard PR before timing. All checks passed.
One warmup per path precedes alternating-order timing rounds. CUDA uses
synchronized wall time. Tables report medians for the **entire sequence**, not
per-step latency. CUDA memory is maximum additional allocated memory above
warmed resident inputs/state, excluding allocator reservations. Returned snapshots
are retained until timing and memory measurement finish.

## CUDA, every prefix, validation disabled

Seven timing rounds.

| Image | Predictions / checkpoints | Fused prefix ms | Standard PR ms | Standard AP ms | Speedup vs PR |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128×192 | 16 / 16 | 1.39 | 19.20 | 41.69 | 13.81× |
| 256×384 | 64 / 64 | 4.30 | 76.61 | 166.76 | 17.80× |
| 512×768 | 64 / 64 | 4.32 | 81.01 | 171.95 | 18.73× |

## CUDA, every prefix, validation enabled

Seven timing rounds, using the API's default validation setting.

| Image | Predictions / checkpoints | Fused prefix ms | Standard PR ms | Standard AP ms | Speedup vs PR |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128×192 | 16 / 16 | 1.92 | 21.61 | 42.28 | 11.24× |
| 256×384 | 64 / 64 | 7.13 | 89.71 | 174.02 | 12.58× |
| 512×768 | 64 / 64 | 7.37 | 95.09 | 177.62 | 12.91× |

## CUDA, every eighth prefix, validation disabled

Seven timing rounds. The final prefix is always included.

| Image | Predictions / checkpoints | Fused prefix ms | Standard PR ms | Standard AP ms | Speedup vs PR |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128×192 | 16 / 2 | 0.54 | 2.51 | 5.12 | 4.68× |
| 256×384 | 64 / 8 | 1.50 | 9.85 | 20.12 | 6.58× |
| 512×768 | 64 / 8 | 1.59 | 10.42 | 21.08 | 6.56× |

## CUDA temporary allocation, every prefix, validation disabled

| Image / predictions | Fused prefix MiB | Standard PR MiB | Standard AP MiB |
| --- | ---: | ---: | ---: |
| 128×192 / 16 | 0.14 | 0.69 | 8.04 |
| 256×384 / 64 | 0.54 | 8.98 | 38.56 |
| 512×768 / 64 | 1.66 | 33.44 | 63.09 |

## CPU reference, every prefix, validation disabled

The CPU implementation is unchanged. Three timing rounds, one PyTorch thread.
The prefix column below is the CPU reference implementation, not a CUDA kernel.

| Image | Predictions / checkpoints | CPU prefix ms | Standard PR ms | Standard AP ms | Speedup vs PR |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128×192 | 16 / 16 | 16.89 | 151.18 | 203.77 | 8.95× |
| 256×384 | 64 / 64 | 97.51 | 4674.03 | 4905.26 | 47.93× |
| 512×768 | 64 / 64 | 168.23 | 5522.94 | 5713.00 | 32.83× |

The standard CPU instance matcher repeats Python loops over thresholds and area
ranges at every prefix. Incremental evaluation matches each new prediction once
at requested thresholds, so these gains reflect less matching as well as reused
intersections.

## Implementation

One native extension call per image dispatches the sequence on the current CUDA
stream. Initialization builds GT areas and a void canvas. For each prediction:

1. A fused draw kernel transfers overwritten pixels between histogram columns,
   updates visible segment areas, and counts original-mask/GT intersections.
2. A fused scoring kernel updates greedy independent instance matches. At
   requested checkpoints it also calculates canvas PQ and semantic statistics.
3. After the sequence, one batched summary launch computes all checkpoint ratios.

There is no per-step Python dispatch or host synchronization. Mask area is
obtained during drawing, avoiding the previous dense int64 area-reduction
intermediate. Validation reads one native error flag per image in addition to
existing input validation. Dense masks still require scanning each prediction;
canvas scoring scans segment statistics, not pixels. Persistent kernel workspace
uses one canvas, a GT×(N+1) visible histogram, N×GT independent intersections,
and threshold×GT instance-match state, plus checkpoint statistics/outputs.

The kernels still launch drawing and scoring separately at each generation step;
stream ordering supplies the required global synchronization. Images are handled
one at a time by the current API. These synthetic measurements do not establish
performance for arbitrary batch sizes, mask densities or segment counts.

## Previous implementation

The earlier PyTorch CUDA orchestration took 32.79, 126.69 and 127.23 ms for the
three every-prefix workloads, versus 1.39, 4.30 and 4.32 ms now. Its largest
additional allocation was 192.01 MiB versus 1.66 MiB now. Those measurements were
separate runs; the equivalent standard baselines are remeasured alongside the
fused implementation in the current tables.

Historical artifacts `prefix_cuda.json`, `prefix_cuda_stride8.json`, and
`prefix_cuda_validate.json` describe that superseded implementation. Current
results use the `prefix_cuda_fused*.json` filenames. All JSON artifacts include
individual timing samples and parity status.

## Reproduce

Use the container invocation from README, then run these commands inside the
container (or set CUDA_VISIBLE_DEVICES=4 before launching Apptainer):

```bash
CUDA_VISIBLE_DEVICES=4 python -m examples.benchmark_prefix --rounds 7 --output benchmarks/prefix_cuda_fused.json
CUDA_VISIBLE_DEVICES=4 python -m examples.benchmark_prefix --rounds 7 --stride 8 --output benchmarks/prefix_cuda_fused_stride8.json
CUDA_VISIBLE_DEVICES=4 python -m examples.benchmark_prefix --rounds 7 --validate --output benchmarks/prefix_cuda_fused_validate.json
python -m examples.benchmark_prefix --device cpu --rounds 3 --output benchmarks/prefix_cpu.json
```
