"""Compare resident CUDA instance AP with COCOeval on pre-encoded CPU RLEs."""

import argparse
from contextlib import redirect_stdout
from io import StringIO
from time import perf_counter
import statistics

import numpy as np
import torch
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from panoptic_evaluator import InstanceBatch, InstanceEvaluator


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--images", type=int, default=8)
    parser.add_argument("--instances", type=int, default=64)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--classes", type=int, default=15)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    rng = np.random.default_rng(19)
    masks, labels, scores, annotations, detections, images = [], [], [], [], [], []
    for image in range(args.images):
        pixels = np.zeros((args.instances, args.height, args.width), dtype=bool)
        for m in pixels:
            y, x = rng.integers(0, args.height - 16), rng.integers(0, args.width - 16)
            m[y : y + 32, x : x + 32] = True
        cats = rng.integers(0, args.classes, args.instances)
        confidence = rng.random(args.instances)
        masks.append(torch.tensor(pixels, device="cuda"))
        labels.append(torch.tensor(cats, device="cuda"))
        scores.append(torch.tensor(confidence, device="cuda"))
        images.append(dict(id=image + 1, height=args.height, width=args.width))
        for j, m in enumerate(pixels):
            rle = mask_utils.encode(np.asfortranarray(m, dtype=np.uint8))
            entry = dict(
                id=len(annotations) + 1,
                image_id=image + 1,
                category_id=int(cats[j]),
                segmentation=rle,
                area=float(mask_utils.area(rle)),
                iscrowd=0,
            )
            annotations.append(entry)
            detections.append(dict(entry, score=float(confidence[j])))
    prediction = InstanceBatch(masks, labels, scores)
    target = InstanceBatch(masks, labels)
    evaluator = InstanceEvaluator(args.classes, device="cuda", validate=args.validate)
    gt, dt = COCO(), COCO()
    with redirect_stdout(StringIO()):
        for coco, anns in ((gt, annotations), (dt, detections)):
            coco.dataset = dict(
                images=images,
                categories=[dict(id=i) for i in range(args.classes)],
                annotations=anns,
            )
            coco.createIndex()

    def native():
        evaluator.reset()
        evaluator.update(prediction, target)
        return evaluator.compute()

    def official():
        with redirect_stdout(StringIO()):
            e = COCOeval(gt, dt, "segm")
            e.evaluate()
            e.accumulate()
            e.summarize()
        return e

    for _ in range(2):
        native()
    torch.cuda.synchronize()
    gpu_times, cpu_times = [], []
    for _ in range(args.rounds):
        start = perf_counter()
        result = native()
        torch.cuda.synchronize()
        gpu_times.append((perf_counter() - start) * 1000)
        start = perf_counter()
        reference = official()
        cpu_times.append((perf_counter() - start) * 1000)
    np.testing.assert_allclose(
        result["precision"].cpu(), reference.eval["precision"], atol=1e-14, rtol=0
    )
    np.testing.assert_allclose(
        result["recall"].cpu(), reference.eval["recall"], atol=1e-14, rtol=0
    )
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(
        f"{args.images} images, {args.instances} instances/image, {args.height}x{args.width}, {args.classes} classes"
    )
    print(
        f"Native CUDA reset/update/compute: {statistics.median(gpu_times):.3f} ms (validation={args.validate})"
    )
    print(
        f"COCOeval evaluate/accumulate/summarize, pre-encoded RLE: {statistics.median(cpu_times):.3f} ms"
    )
    print(
        "Full precision/recall parity passed; input generation and encoding excluded."
    )


if __name__ == "__main__":
    main()
