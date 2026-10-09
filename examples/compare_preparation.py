"""Compare preparation and validation with a built earlier implementation."""
import argparse
import importlib.util
from pathlib import Path
import statistics
import sys
import time

import torch

from panoptic_evaluator import PanopticBatch, PanopticEvaluator


def measure(fn, iterations):
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000 / iterations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()
    if args.rounds <= 0 or args.iterations <= 0:
        parser.error("rounds and iterations must be positive")
    package = args.baseline_dir.resolve() / "panoptic_evaluator"
    if not (package / "__init__.py").is_file():
        parser.error("baseline must contain a built panoptic_evaluator package")
    spec = importlib.util.spec_from_file_location("baseline_panoptic_evaluator", package / "__init__.py",
                                                submodule_search_locations=[str(package)])
    baseline = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = baseline
    spec.loader.exec_module(baseline)
    B, H, W, S, C = 8, 512, 768, 64, 15
    y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    slots = (1 + (x // 32 + y // 32 * ((W + 31) // 32)) % S).int()[None].expand(B, -1, -1).contiguous()
    labels = (torch.arange(S + 1, dtype=torch.int32) % C)[None].expand(B, -1).clone()
    labels[:, 0] = -1
    pred_slots = slots.clone()
    pred_slots[:, ::11, :] = 0
    # Sparse shuffled IDs exercise actual lookup, not accidentally compact labels.
    mapping = torch.cat((torch.zeros(1, dtype=torch.int64),
                         (torch.arange(S, 0, -1, dtype=torch.int64) * 100003 + 2**40)))
    gt_cpu, pred_cpu = mapping[slots.long()], mapping[pred_slots.long()]
    infos = [[{"id": int(mapping[s]), "category_id": int(s % C)} for s in range(1, S + 1)] for _ in range(B)]
    gt_gpu, pred_gpu = gt_cpu.cuda(), pred_cpu.cuda()
    evaluators = (baseline.PanopticEvaluator(C, validate=False), PanopticEvaluator(C, validate=False))
    prepared = ((baseline.PanopticBatch(pred_slots.cuda(), labels.cuda()), baseline.PanopticBatch(slots.cuda(), labels.cuda())),
                (PanopticBatch(pred_slots.cuda(), labels.cuda()), PanopticBatch(slots.cuda(), labels.cuda())))

    def run_prepared(index, validate):
        evaluator = evaluators[index]
        evaluator.validate = validate
        evaluator.reset()
        evaluator.update(*prepared[index])
        return evaluator.compute()

    def run_ids(index, transfer):
        evaluator = evaluators[index]
        # Old packing checked unknown pixel IDs; the new path validates all values.
        evaluator.validate = index == 1
        evaluator.reset()
        maps = (gt_cpu.cuda(), pred_cpu.cuda()) if transfer else (gt_gpu, pred_gpu)
        if index == 0:
            target = baseline.PanopticBatch.from_coco(maps[0], infos, list(range(C)))
            prediction = baseline.PanopticBatch.from_coco(maps[1], infos, list(range(C)))
        else:
            target = PanopticBatch.from_coco(maps[0], infos, list(range(C)), compact=False)
            prediction = PanopticBatch.from_coco(maps[1], infos, list(range(C)), compact=False)
        evaluator.update(prediction, target)
        return evaluator.compute()

    cases = [("Resident compact, no value validation", lambda i: run_prepared(i, False)),
             ("Resident compact, validated", lambda i: run_prepared(i, True)),
             ("Resident original IDs + metadata preparation", lambda i: run_ids(i, False)),
             ("CPU original IDs + transfer + metadata preparation", lambda i: run_ids(i, True))]
    print(torch.cuda.get_device_name())
    print(f"{B} x {H} x {W}, {S} segments, shuffled sparse int64 IDs; reset/update/compute included")
    for name, run in cases:
        expected, actual = run(0), run(1)
        for key in ("tp", "fp", "fn", "iou_sum", "confusion", "miou"):
            torch.testing.assert_close(actual[key], expected[key], rtol=1e-12, atol=1e-12)
        for index in range(2):
            for _ in range(5):
                run(index)
        samples = [[], []]
        for round_index in range(args.rounds):
            for index in ((0, 1) if round_index % 2 == 0 else (1, 0)):
                samples[index].append(measure(lambda: run(index), args.iterations))
        old, new = (statistics.median(values) for values in samples)
        print(f"{name}: previous {old:.3f} ms, current {new:.3f} ms, {old / new:.2f}x")
        print(f"  ranges: previous {min(samples[0]):.3f}-{max(samples[0]):.3f}, current {min(samples[1]):.3f}-{max(samples[1]):.3f} ms")


if __name__ == "__main__":
    main()
