"""Compare an earlier checkout with this implementation on resident CUDA inputs."""
import argparse
import importlib.util
from pathlib import Path
import statistics
import sys
import time

import torch

from panoptic_evaluator import PanopticBatch, PanopticEvaluator


def measure(evaluator, prediction, target, *, full, iterations, graph=False):
    def run_iterations():
        for _ in range(iterations):
            if full:
                evaluator.reset()
            evaluator.update(prediction, target)
            if full:
                evaluator.compute()
    captured = None
    if graph:
        captured = torch.cuda.CUDAGraph()
        with torch.cuda.graph(captured):
            run_iterations()
        captured.replay()
    torch.cuda.synchronize()
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    wall = time.perf_counter()
    start.record()
    if captured is not None:
        captured.replay()
    else:
        run_iterations()
    stop.record()
    stop.synchronize()
    return (time.perf_counter() - wall) * 1000 / iterations, start.elapsed_time(stop) / iterations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", type=Path, required=True,
                        help="Earlier project directory containing its built panoptic_evaluator package")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--segments", type=int, default=64)
    parser.add_argument("--classes", type=int, default=15)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--original-ids", action="store_true")
    parser.add_argument("--fragmented", action="store_true")
    parser.add_argument("--independent-prediction", action="store_true",
                        help="Use independent random prediction slots to exercise dense intersections")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--cuda-graph", action="store_true",
                        help="Time graph replay to isolate device work from Python launch overhead")
    args = parser.parse_args()
    if min(value for value in vars(args).values() if type(value) is int) <= 0:
        parser.error("numeric arguments must be positive")
    if args.cuda_graph and args.validate:
        parser.error("synchronous validation cannot be captured in a CUDA graph")
    package = args.baseline_dir.resolve() / "panoptic_evaluator"
    if not (package / "__init__.py").is_file():
        parser.error("baseline directory must contain panoptic_evaluator/__init__.py")
    spec = importlib.util.spec_from_file_location("baseline_panoptic_evaluator", package / "__init__.py",
                                                submodule_search_locations=[str(package)])
    baseline = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = baseline
    spec.loader.exec_module(baseline)
    x = torch.arange(args.width, device="cuda")[None, :]
    y = torch.arange(args.height, device="cuda")[:, None]
    slots = 1 + ((x // 32 + y // 32 * ((args.width + 31) // 32)) % args.segments)
    maps = slots.int()[None].expand(args.batch_size, -1, -1).contiguous()
    if args.fragmented:
        torch.manual_seed(1234)
        maps = torch.randint(1, args.segments + 1, maps.shape, device="cuda", dtype=torch.int32)
    labels = (torch.arange(args.segments + 1, device="cuda", dtype=torch.int32) % args.classes)
    labels[0] = -1
    labels = labels[None].expand(args.batch_size, -1).contiguous()
    present = torch.stack([torch.bincount(row.flatten().long(), minlength=args.segments + 1) > 0
                           for row in maps])
    labels = torch.where(present, labels, -1).contiguous()
    pred = maps.clone()
    if args.independent_prediction:
        torch.manual_seed(4321)
        pred = torch.randint(1, args.segments + 1, maps.shape, device="cuda", dtype=torch.int32)
    pred[:, ::11, :] = 0
    pred_present = torch.stack([torch.bincount(row.flatten().long(), minlength=args.segments + 1) > 0
                                for row in pred])
    pred_labels = torch.where(pred_present, torch.arange(args.segments + 1, device="cuda", dtype=torch.int32)[None] % args.classes, -1).contiguous()
    pred_labels[:, 0] = -1
    evaluators = (baseline.PanopticEvaluator(args.classes, validate=args.validate),
                  PanopticEvaluator(args.classes, validate=args.validate))
    inputs = ((baseline.PanopticBatch(pred, pred_labels), baseline.PanopticBatch(maps, labels)),
              (PanopticBatch(pred, pred_labels), PanopticBatch(maps, labels)))
    if args.original_ids:
        def original(batch_type, image, classes):
            metadata = [[{"id": slot * 100003 + 2**40, "category_id": label}
                         for slot, label in enumerate(row) if label >= 0]
                        for row in classes.cpu().tolist()]
            ids = torch.where(image > 0, image.long() * 100003 + 2**40, 0)
            return batch_type.from_coco(ids, metadata, list(range(args.classes)), compact=False)
        inputs = tuple((original(batch_type, pred, pred_labels), original(batch_type, maps, labels))
                       for batch_type in (baseline.PanopticBatch, PanopticBatch))
    for evaluator, (prediction, target) in zip(evaluators, inputs):
        for _ in range(20):
            evaluator.update(prediction, target)
        evaluator.reset()
        evaluator.update(prediction, target)
    expected, actual = (evaluator.compute() for evaluator in evaluators)
    for key in ("tp", "fp", "fn", "iou_sum", "confusion", "miou"):
        torch.testing.assert_close(actual[key], expected[key], rtol=1e-12, atol=1e-12)
    print(torch.cuda.get_device_name())
    print(f"{args.batch_size} x {args.height} x {args.width}, {args.segments} segments, "
          f"validation={args.validate}, original_ids={args.original_ids}, fragmented={args.fragmented}, "
          f"independent_prediction={args.independent_prediction}, cuda_graph={args.cuda_graph}")
    for full in (False, True):
        results = [[], []]
        for round_index in range(args.rounds):
            for index in ((0, 1) if round_index % 2 == 0 else (1, 0)):
                results[index].append(measure(evaluators[index], *inputs[index], full=full,
                                              iterations=args.iterations, graph=args.cuda_graph))
        print("Reset/update/compute" if full else "Update only")
        for name, samples in zip(("Baseline", "Current"), results):
            wall = [sample[0] for sample in samples]
            event = [sample[1] for sample in samples]
            print(f"  {name}: wall median {statistics.median(wall):.3f} ms/batch "
                  f"(range {min(wall):.3f}-{max(wall):.3f}); CUDA event median {statistics.median(event):.3f} ms/batch")
        ratio = statistics.median(sample[0] for sample in results[0]) / statistics.median(sample[0] for sample in results[1])
        print(f"  Speedup: {ratio:.2f}x")


if __name__ == "__main__":
    main()
