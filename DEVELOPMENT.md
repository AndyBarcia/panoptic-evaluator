# Development

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

Prefix benchmarking is available with `python -m examples.benchmark_prefix`.
Use `--device cpu`, `--stride 8`, or `--validate` to select CPU execution,
checkpoint spacing, or input validation. The benchmark checks metric parity
against fresh standard evaluation at every checkpoint before timing.
See [recorded results and methodology](benchmarks/prefix_results.md), including
metric parity checks and comparisons with repeated standard evaluation.
