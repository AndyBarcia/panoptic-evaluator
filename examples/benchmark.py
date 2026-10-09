"""Measure evaluation of resident tensors; excludes data generation and packing."""
import argparse
import time

import torch

from panoptic_evaluator import PanopticBatch, PanopticEvaluator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--segments", type=int, default=64)
    parser.add_argument("--classes", type=int, default=15)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    if min(vars(args)[key] for key in ("batch_size", "height", "width", "segments", "classes", "iterations")) <= 0:
        parser.error("dimensions, classes, segments and iterations must be positive")
    torch.manual_seed(1234)
    # Spatially coherent regions model segmentation maps better than random pixels.
    x = torch.arange(args.width, device="cuda")[None, :]
    y = torch.arange(args.height, device="cuda")[:, None]
    slots = 1 + ((x // 32 + y // 32 * ((args.width + 31) // 32)) % args.segments)
    maps = slots.to(torch.int32)[None].expand(args.batch_size, -1, -1).contiguous()
    labels = (torch.arange(args.segments + 1, device="cuda", dtype=torch.int32) % args.classes)
    labels[0] = -1
    labels = labels[None].expand(args.batch_size, -1).contiguous()
    # Remove metadata for slots not present at small resolutions.
    present = torch.bincount(maps[0].flatten().long(), minlength=args.segments + 1) > 0
    labels = torch.where(present[None], labels, -1).contiguous()
    target = PanopticBatch(maps, labels)
    prediction_map = maps.clone()
    prediction_map[:, ::11, :] = 0
    pred_present = torch.bincount(prediction_map[0].flatten().long(), minlength=args.segments + 1) > 0
    prediction = PanopticBatch(prediction_map, torch.where(pred_present[None], labels, -1).contiguous())
    evaluator = PanopticEvaluator(args.classes, validate=args.validate)
    for _ in range(10):
        evaluator.update(prediction, target)
    evaluator.reset()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    wall = time.perf_counter()
    start.record()
    for _ in range(args.iterations):
        evaluator.update(prediction, target)
    stop.record()
    stop.synchronize()
    elapsed = time.perf_counter() - wall
    milliseconds = start.elapsed_time(stop) / args.iterations
    result = evaluator.compute()
    print(f"{args.batch_size} x {args.height} x {args.width}, {args.segments} segments, validation={args.validate}")
    print(f"CUDA events: {milliseconds:.3f} ms/batch; synchronized wall: {elapsed * 1000 / args.iterations:.3f} ms/batch")
    print(f"Throughput: {args.batch_size * args.iterations / elapsed:.1f} images/s")
    print(f"Peak allocated: {torch.cuda.max_memory_allocated() / 2**20:.1f} MiB")
    print(f"PQ={result['All']['pq'].item():.6f}, mIoU={result['miou'].item():.6f}")


if __name__ == "__main__":
    main()
