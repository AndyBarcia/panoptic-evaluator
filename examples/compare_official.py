"""Compare the official single-worker evaluator and CUDA on identical maps.

Requires panopticapi, NumPy and Pillow. PNGs are created before timing.
"""
import argparse
import contextlib
import io
import statistics
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image
from panopticapi.evaluation import pq_compute_single_core

from panoptic_evaluator import PanopticBatch, PanopticEvaluator


def measure(fn, iterations, cuda=False):
    fn()
    if cuda:
        torch.cuda.synchronize()
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        if cuda:
            torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
    return statistics.median(times)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--segments", type=int, default=64)
    parser.add_argument("--classes", type=int, default=15)
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()
    if min(vars(args).values()) <= 0:
        parser.error("arguments must be positive")
    y, x = np.indices((args.height, args.width))
    base = (1 + (x // 32 + y // 32 * ((args.width + 31) // 32)) % args.segments).astype(np.int32)
    gt = np.repeat(base[None], args.batch_size, axis=0)
    pred = gt.copy()
    pred[:, ::11, :] = 0
    categories = {c: {"isthing": c % 2} for c in range(args.classes)}
    metadata = []
    annotations = []
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        resident = {}
        for name, maps in (("gt", gt), ("pred", pred)):
            (root / name).mkdir()
            infos, anns = [], []
            for b, image in enumerate(maps):
                ids, areas = np.unique(image, return_counts=True)
                info = [{"id": int(i), "category_id": int(i % args.classes),
                         "area": int(area), "iscrowd": 0} for i, area in zip(ids, areas) if i]
                infos.append(info)
                anns.append({"image_id": b, "file_name": f"{b}.png", "segments_info": info})
                rgb = np.stack((image % 256, image // 256 % 256, image // 65536 % 256), -1).astype(np.uint8)
                path = root / name / f"{b}.png"
                resident[str(path)] = Image.fromarray(rgb)
                resident[str(path)].save(path)
            metadata.append(infos)
            annotations.append(anns)
        pairs = list(zip(*annotations))
        def run_official():
            with contextlib.redirect_stdout(io.StringIO()):
                return pq_compute_single_core(0, pairs, str(root / "gt"), str(root / "pred"), categories)
        png_ms = measure(run_official, args.iterations)
        with patch("panopticapi.evaluation.Image.open", side_effect=lambda path: resident[str(path)]):
            memory_ms = measure(run_official, args.iterations)
        gt_gpu, pred_gpu = torch.from_numpy(gt).cuda(), torch.from_numpy(pred).cuda()
        gi, pi = metadata
        target = PanopticBatch.from_coco(gt_gpu, gi, list(categories))
        prediction = PanopticBatch.from_coco(pred_gpu, pi, list(categories))
        evaluator = PanopticEvaluator(args.classes, validate=False)
        def run_cuda():
            evaluator.reset()
            evaluator.update(prediction, target)
            return evaluator.compute()
        for _ in range(10):
            run_cuda()
        cuda_ms = measure(run_cuda, 50, cuda=True)
        evaluator.validate = True
        validated_ms = measure(run_cuda, 50, cuda=True)
        evaluator.validate = False
        original_target = PanopticBatch.from_coco(gt_gpu, gi, list(categories), compact=False)
        original_prediction = PanopticBatch.from_coco(pred_gpu, pi, list(categories), compact=False)
        def run_original_ids():
            evaluator.reset()
            evaluator.update(original_prediction, original_target)
            return evaluator.compute()
        original_ms = measure(run_original_ids, 50, cuda=True)
        def pack_and_run():
            evaluator.reset()
            target = PanopticBatch.from_coco(gt_gpu, gi, list(categories))
            prediction = PanopticBatch.from_coco(pred_gpu, pi, list(categories))
            evaluator.update(prediction, target)
            return evaluator.compute()
        packed_ms = measure(pack_and_run, 30, cuda=True)
        def fused_prepare_and_run():
            evaluator.reset()
            target = PanopticBatch.from_coco(gt_gpu, gi, list(categories), compact=False)
            prediction = PanopticBatch.from_coco(pred_gpu, pi, list(categories), compact=False)
            evaluator.update(prediction, target)
            return evaluator.compute()
        evaluator.validate = True
        fused_ms = measure(fused_prepare_and_run, 30, cuda=True)
        evaluator.validate = False
        def transfer_pack_and_run():
            evaluator.reset()
            target = PanopticBatch.from_coco(torch.from_numpy(gt).cuda(), gi, list(categories))
            prediction = PanopticBatch.from_coco(torch.from_numpy(pred).cuda(), pi, list(categories))
            evaluator.update(prediction, target)
            return evaluator.compute()
        transfer_ms = measure(transfer_pack_and_run, 30, cuda=True)
        def fused_transfer_and_run():
            evaluator.reset()
            target = PanopticBatch.from_coco(torch.from_numpy(gt).cuda(), gi, list(categories), compact=False)
            prediction = PanopticBatch.from_coco(torch.from_numpy(pred).cuda(), pi, list(categories), compact=False)
            evaluator.update(prediction, target)
            return evaluator.compute()
        evaluator.validate = True
        fused_transfer_ms = measure(fused_transfer_and_run, 30, cuda=True)
        result = run_cuda()
        reference = run_official()
        for c in categories:
            assert result["tp"][c].item() == reference[c].tp
            assert result["fp"][c].item() == reference[c].fp
            assert result["fn"][c].item() == reference[c].fn
            assert abs(result["iou_sum"][c].item() - reference[c].iou) < 1e-9
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"{args.batch_size} x {args.height} x {args.width}, {args.segments} segments, {args.classes} classes")
        print("Median synchronized wall time per batch; CUDA includes reset/update/compute.")
        for label, ms in (("Official single-worker PNG", png_ms),
                          ("Official single-worker resident RGB", memory_ms),
                          ("CUDA resident compact tensors", cuda_ms),
                          ("CUDA resident compact, validated", validated_ms),
                          ("CUDA resident original IDs, prepared lookup", original_ms),
                          ("CUDA resident ID maps + COCO packing", packed_ms),
                          ("CUDA resident original IDs + batched preparation, validated", fused_ms),
                          ("CUDA CPU ID maps + transfer + packing", transfer_ms),
                          ("CUDA CPU original IDs + transfer + batched preparation, validated", fused_transfer_ms)):
            print(f"{label}: {ms:.3f} ms, {args.batch_size * 1000 / ms:.1f} images/s")
        print(f"Resident speedup: {memory_ms / cuda_ms:.1f}x; PNG/compact ratio: {png_ms / cuda_ms:.1f}x")
        print("PQ raw statistics match the official evaluator. PNG creation and process startup are excluded.")


if __name__ == "__main__":
    main()
