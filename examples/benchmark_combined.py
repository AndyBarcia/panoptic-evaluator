"""Compare shared-ground-truth evaluation with separate PQ and instance paths."""

import argparse
import math
import statistics
from time import perf_counter

import torch
import torch.nn.functional as F

from panoptic_evaluator import (
    InstanceBatch,
    InstanceEvaluator,
    PanopticBatch,
    PanopticEvaluator,
    SegmentationEvaluator,
)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--images", type=int, default=8)
    parser.add_argument("--segments", type=int, default=64)
    parser.add_argument("--detections", type=int, default=64)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--classes", type=int, default=15)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    if min(args.images, args.segments, args.detections, args.classes, args.rounds) <= 0:
        parser.error(
            "image, segment, detection, category and round counts must be positive"
        )
    generator = torch.Generator().manual_seed(24)
    device = "cuda"
    maps = torch.zeros((args.images, args.height, args.width), dtype=torch.int32)
    side = math.ceil(math.sqrt(args.segments))
    if min(args.height, args.width) < side:
        parser.error("image dimensions must fit the segment grid")
    for b in range(args.images):
        for slot in range(1, args.segments + 1):
            y, x = divmod(slot - 1, side)
            maps[
                b,
                y * args.height // side : (y + 1) * args.height // side,
                x * args.width // side : (x + 1) * args.width // side,
            ] = slot
    labels = torch.randint(
        args.classes,
        (args.images, args.segments + 1),
        generator=generator,
        dtype=torch.int32,
    )
    labels[:, 0] = -1
    labels[:, 1] = 0  # At least one thing target.
    things = torch.arange(args.classes) < max(1, args.classes * 2 // 3)
    crowd = torch.rand(labels.shape, generator=generator) < 0.1
    crowd[:, 0] = False
    target = PanopticBatch(maps.to(device), labels.to(device), crowd.to(device))
    pred_maps = target.segment_map.clone()
    pred_maps[:, :3] = 0
    prediction = PanopticBatch(pred_maps, target.classes)
    masks, classes, scores, gt_masks, gt_classes, gt_crowd = [], [], [], [], [], []
    for b in range(args.images):
        slots = torch.where((labels[b] >= 0) & things[labels[b].clamp_min(0).long()])[0]
        gt = target.segment_map[b][None] == slots.to(device)[:, None, None]
        selected = torch.randint(len(slots), (args.detections,), generator=generator)
        # Raw detections extend into adjacent ground-truth objects and overlap.
        detections = F.max_pool2d(
            gt[selected].float()[:, None], 5, stride=1, padding=2
        )[:, 0].bool()
        masks.append(detections)
        classes.append(labels[b, slots[selected]].to(device))
        scores.append(torch.rand(args.detections, generator=generator).to(device))
        gt_masks.append(gt)
        gt_classes.append(labels[b, slots].to(device))
        gt_crowd.append(crowd[b, slots].to(device))
    detections = InstanceBatch(masks, classes, scores)
    instance_target = InstanceBatch(gt_masks, gt_classes, crowd=gt_crowd)
    joint = SegmentationEvaluator(
        args.classes, isthing=things.tolist(), device=device, validate=args.validate
    )
    pq = PanopticEvaluator(
        args.classes, isthing=things.tolist(), device=device, validate=args.validate
    )
    ap = InstanceEvaluator(args.classes, device=device, validate=args.validate)

    def combined():
        joint.reset()
        joint.update(prediction, detections, target)
        return joint.compute()

    def separate():
        pq.reset()
        ap.reset()
        pq.update(prediction, target)
        ap.update(detections, instance_target)
        result = pq.compute()
        result["instance"] = ap.compute()
        return result

    def measure(fn):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        initial = torch.cuda.memory_allocated()
        start = perf_counter()
        result = fn()
        torch.cuda.synchronize()
        return (
            result,
            (perf_counter() - start) * 1000,
            torch.cuda.max_memory_allocated() - initial,
        )

    for _ in range(2):
        combined()
        separate()
    joint_times, separate_times, joint_memory, separate_memory = [], [], [], []
    for _ in range(args.rounds):
        actual, elapsed, memory = measure(combined)
        joint_times.append(elapsed)
        joint_memory.append(memory)
        expected, elapsed, memory = measure(separate)
        separate_times.append(elapsed)
        separate_memory.append(memory)
    for key in ("miou", "tp", "fp", "fn", "iou_sum", "confusion"):
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=1e-14)
    for key in expected["instance"]:
        torch.testing.assert_close(
            actual["instance"][key], expected["instance"][key], rtol=0, atol=1e-14
        )
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(
        f"{args.images} images, {args.segments} GT segments, {args.detections} detections/image, {args.height}x{args.width}, {args.classes} classes"
    )
    print(
        f"Combined reset/update/compute: {statistics.median(joint_times):.3f} ms; additional peak {max(joint_memory)/2**20:.2f} MiB"
    )
    print(
        f"Separate reset/update/compute: {statistics.median(separate_times):.3f} ms; additional peak {max(separate_memory)/2**20:.2f} MiB"
    )
    print(
        f"Validation={args.validate}; generation and GT mask materialization excluded from both timings."
    )
    print(
        f"Separate path additionally requires {sum(x.numel()*x.element_size() for x in gt_masks)/2**20:.2f} MiB of resident GT masks."
    )
    print("Metric parity passed.")


if __name__ == "__main__":
    main()
