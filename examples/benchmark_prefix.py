"""Compare incremental prefixes with fresh standard evaluation at every prefix."""
import argparse
import json
import math
import statistics
from time import perf_counter

import torch
import torch.nn.functional as F

from panoptic_evaluator import (InstanceBatch, PanopticBatch, SegmentationEvaluator,
                                segmentation_prefixes)


def workload(height, width, count, device):
    side = math.ceil(math.sqrt(count))
    y = torch.arange(height)[:, None] * side // height
    x = torch.arange(width)[None] * side // width
    maps = (1 + (y * side + x) % count).int()[None].to(device)
    labels = (torch.arange(count + 1, device=device) % 5).int()[None]
    labels[:, 0] = -1
    target = PanopticBatch(maps, labels)
    # Include duplicate predictions, imperfect boundaries and substantial overlap.
    order = torch.arange(count, device=device)
    order = torch.where(order % 5 == 1, order - 1, order)
    slots = 1 + (order * 7) % count
    masks = maps[0][None] == slots[:, None, None]
    masks = F.max_pool2d(masks.float()[:, None], 9, stride=1, padding=4)[:, 0].bool()
    return InstanceBatch([masks], [labels[0, slots]]), target


def run_case(height, width, count, device, rounds, validate, stride):
    prediction, target = workload(height, width, count, device)
    points = list(range(stride, count + 1, stride))
    if not points or points[-1] != count:
        points.append(count)
    things = [True, True, True, False, False]
    evaluator = SegmentationEvaluator(5, isthing=things, device=device, validate=validate)
    masks, classes = prediction.masks[0], prediction.classes[0]
    thing_flags = torch.tensor(things, device=device)[classes.long()]

    def incremental():
        return segmentation_prefixes(prediction, target, 5, isthing=things,
                                     checkpoints=points, validate=validate)[0]

    def standard(ap=False):
        canvas = torch.zeros_like(target.segment_map)
        labels = torch.cat((classes.new_tensor([-1]), classes))
        results = []
        for i, mask in enumerate(masks):
            canvas[0][mask] = i + 1
            if i + 1 not in points:
                continue
            present = torch.bincount(canvas.flatten().long(), minlength=count + 1) > 0
            active_labels = torch.where(present, labels, -1)
            selected = thing_flags[:i+1]
            # Strictly descending scores reproduce generation-order matching.
            scores = -torch.arange(i + 1, dtype=torch.float64, device=device)[selected]
            detections = InstanceBatch([masks[:i+1][selected]],
                                       [classes[:i+1][selected]], [scores])
            evaluator.reset()
            evaluator.update(PanopticBatch(canvas, active_labels[None]), detections, target)
            if ap:
                results.append(evaluator.compute())
                continue
            result = evaluator.panoptic.compute()
            # Read standard matching output before AP interpolation. This compares
            # the same endpoint precision/recall as the prefix API, without charging
            # the baseline for computing different metrics (COCO AP/AR).
            record = evaluator.instance._records[0]
            flags = record.status[[0, 5]]  # all-area IoU .50 and .75
            tp = torch.zeros((2, 5), dtype=torch.int64, device=device)
            fp = torch.zeros_like(tp)
            indices = record.classes.long()[None].expand(2, -1)
            tp.scatter_add_(1, indices, (flags == 1).long())
            fp.scatter_add_(1, indices, (flags == 0).long())
            positives = evaluator.instance._counts[:, 0][None]
            result['instance'] = dict(tp=tp, fp=fp, fn=positives-tp,
                                      precision=tp.double()/(tp+fp).clamp_min(1),
                                      recall=tp.double()/positives.clamp_min(1))
            results.append(result)
        return results

    actual, expected = incremental(), standard()
    for a, e in zip(actual, expected):
        for key in ('tp', 'fp', 'fn', 'iou_sum', 'confusion', 'miou'):
            torch.testing.assert_close(a[key], e[key], rtol=1e-12, atol=1e-12, equal_nan=True)
        for group in ('All', 'Things', 'Stuff', 'per_class', 'instance'):
            for key in e[group]:
                torch.testing.assert_close(a[group][key], e[group][key],
                                           rtol=1e-12, atol=1e-12, equal_nan=True)
    del actual, expected

    def sync():
        if device == 'cuda':
            torch.cuda.synchronize()

    operations = dict(prefix=incremental, standard_pr=standard,
                      standard_ap=lambda: standard(ap=True))
    times = {name: [] for name in operations}
    peaks = {name: [] for name in operations}
    for operation in operations.values():
        operation()
    # Alternate operation order to reduce systematic warmup/thermal bias.
    for r in range(rounds):
        names = list(operations)
        if r % 2:
            names.reverse()
        for name in names:
            sync()
            if device == 'cuda':
                initial = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
            start = perf_counter()
            result = operations[name]()
            sync()
            times[name].append((perf_counter() - start) * 1000)
            if device == 'cuda':
                peaks[name].append((torch.cuda.max_memory_allocated() - initial) / 2**20)
            del result
    row = dict(device=device, height=height, width=width, predictions=count,
               checkpoints=len(points), validate=validate, parity=True,
               median_ms={name: statistics.median(values) for name, values in times.items()},
               peak_extra_mib={name: max(values) for name, values in peaks.items() if values},
               samples_ms=times)
    row['speedup_vs_standard_pr'] = row['median_ms']['standard_pr'] / row['median_ms']['prefix']
    print(json.dumps(row), flush=True)
    return row


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--validate', action='store_true')
    parser.add_argument('--output')
    args = parser.parse_args()
    if min(args.rounds, args.threads, args.stride) <= 0:
        parser.error('rounds, threads and stride must be positive')
    torch.set_num_threads(args.threads)
    metadata = dict(torch=torch.__version__, threads=args.threads,
                    gpu=torch.cuda.get_device_name() if args.device == 'cuda' else None,
                    rounds=args.rounds, stride=args.stride)
    print(json.dumps(metadata), flush=True)
    rows = [run_case(h, w, n, args.device, args.rounds, args.validate, args.stride)
            for h, w, n in [(128, 192, 16), (256, 384, 64), (512, 768, 64)]]
    if args.output:
        with open(args.output, 'w') as stream:
            json.dump(dict(metadata=metadata, results=rows), stream, indent=2)


if __name__ == '__main__':
    main()
