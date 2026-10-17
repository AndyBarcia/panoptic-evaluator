"""Generation-order evaluation with shared segment intersection statistics."""

import math

import torch

from .evaluator import PanopticEvaluator, _prepare, _require_cuda_extension
from .instance import InstanceBatch, InstanceEvaluator


@torch.no_grad()
def segmentation_prefixes(prediction, target, num_classes, *, isthing=None,
                          checkpoints=None, iou_thresholds=(0.5, 0.75), validate=True):
    """Evaluate independent masks in generation order, later masks drawn on top.

    Returns one list per image of dictionaries with ``prefix``, standard
    panoptic/semantic results, and ``instance`` counts/precision/recall of shape
    [threshold, class]. Instance matching is greedy in generation order, with
    same-class noncrowd GT preferred over reusable crowd GT, IoU >= threshold,
    and no detection cap. Only thing predictions participate in instance metrics.
    Precision/recall with zero denominators are zero. Scores are not consumed.

    Checkpoints default to 1..N (or [0] for an empty sequence). Explicit
    checkpoints must be strictly increasing integers in 0..N for every image.
    Results are per-image, not dataset AP or averaged dataset metrics. Returned
    counts may be summed across images at the same prefix before taking ratios.
    """
    device = target.segment_map.device
    evaluator = PanopticEvaluator(num_classes, isthing=isthing, device=device,
                                  validate=validate)
    thresholds = tuple(iou_thresholds)
    if not thresholds or any(isinstance(t, bool) or not math.isfinite(t) or not 0 < t <= 1
                             for t in thresholds):
        raise ValueError('iou_thresholds must be finite values in (0, 1]')
    thresholds = torch.tensor(thresholds, dtype=torch.float64, device=device)
    requested = None if checkpoints is None else tuple(checkpoints)
    if requested is not None and (any(type(k) is not int or k < 0 for k in requested)
                                  or any(a >= b for a, b in zip(requested, requested[1:]))):
        raise ValueError('checkpoints must be strictly increasing nonnegative integers')
    # Target preparation also supports original COCO IDs.
    target = target.compact()
    maps, gc, crowd = _prepare(target, num_classes, validate)
    images = InstanceEvaluator(num_classes, device=device, validate=validate)._prepare(
        InstanceBatch(prediction.masks, prediction.classes), prediction=False,
        defer_masks=device.type == "cuda")
    if len(images) != len(maps) or any(im.masks.shape[1:] != maps.shape[1:] for im in images):
        raise ValueError('prediction and target must have matching batch and image dimensions')
    if requested and any(requested[-1] > len(im.masks) for im in images):
        raise ValueError('checkpoint exceeds prediction count')
    results = []
    for b, image in enumerate(images):
        n, g = len(image.masks), gc.shape[1]
        points = set(requested if requested is not None else (range(1, n + 1) if n else [0]))
        if device.type == 'cuda':
            ordered_points = sorted(points)
            outputs = _require_cuda_extension().prefix(
                maps[b], image.masks, image.classes, gc[b], crowd[b],
                evaluator.isthing, thresholds, ordered_points, validate)
            if validate and outputs[-1].item():
                raise ValueError('unknown target slot or empty prediction mask')
            results.append(_unpack_cuda(outputs, ordered_points, thresholds))
            continue
        labels = torch.cat((gc.new_full((1,), -1), image.classes))
        canvas = torch.zeros_like(maps[b], dtype=torch.int64)
        gt = maps[b].long()
        ga = torch.bincount(gt.flatten(), minlength=g)
        pairs = torch.zeros((g, n + 1), dtype=torch.int64, device=device)
        pairs[:, 0] = ga
        t = len(thresholds)
        tp = torch.zeros((t, num_classes), dtype=torch.int64, device=device)
        fp = torch.zeros_like(tp)
        used = torch.zeros((t, g), dtype=torch.bool, device=device)
        things = (gc[b] >= 0) & evaluator.isthing[gc[b].clamp_min(0).long()]
        positives = torch.zeros(num_classes, dtype=torch.int64, device=device)
        positives.scatter_add_(0, gc[b].clamp_min(0).long(), (things & ~crowd[b]).long())
        snapshots = []

        def snapshot(k):
            evaluator.reset()
            semantic = torch.zeros_like(evaluator.confusion)
            valid = (gc[b] >= 0) & ~crowd[b]
            active_labels = torch.where(pairs.sum(0) > 0, labels, -1)
            columns = torch.where(active_labels >= 0, active_labels, num_classes).long()
            indices = gc[b, valid, None].long() * (num_classes + 1) + columns[None]
            semantic.view(-1).scatter_add_(0, indices.flatten(), pairs[valid].flatten())
            evaluator._update_histogram(pairs[None], semantic, gc[b:b+1], active_labels[None], crowd[b:b+1])
            result = evaluator.compute()
            result['prefix'] = k
            result['instance'] = dict(
                iou_thresholds=thresholds.clone(), tp=tp.clone(), fp=fp.clone(),
                fn=positives[None] - tp,
                precision=tp.double() / (tp + fp).clamp_min(1),
                recall=tp.double() / positives.clamp_min(1)[None])
            snapshots.append(result)

        if 0 in points:
            snapshot(0)
        for i, mask in enumerate(image.masks):
            # Read the mask once; transfer only its covered pixels between slots.
            covered_gt, previous = gt[mask], canvas[mask]
            intersections = torch.bincount(covered_gt, minlength=g)
            pairs.view(-1).scatter_add_(
                0, covered_gt * (n + 1) + previous, -torch.ones_like(covered_gt))
            pairs[:, i + 1] = intersections
            canvas[mask] = i + 1
            c = image.classes[i].long()
            if bool(evaluator.isthing[c]):
                area = image.pixel_areas[i]
                union = torch.where(crowd[b], area, area + ga - intersections)
                ious = intersections.double() / union.clamp_min(1)
                same = gc[b] == c
                eligible = same[None] & ~crowd[b][None] & ~used & (ious[None] >= thresholds[:, None])
                best_iou, best = torch.where(eligible, ious[None], -1.).max(1)
                matched = best_iou >= 0
                used[torch.arange(t, device=device), best] |= matched
                ignored = (same[None] & crowd[b][None] & (ious[None] >= thresholds[:, None])).any(1)
                tp[:, c] += matched.long()
                fp[:, c] += (~matched & ~ignored).long()
            if i + 1 in points:
                snapshot(i + 1)
        results.append(snapshots)
    return results


def _unpack_cuda(outputs, points, thresholds):
    """Wrap views of device-owned snapshots without launching per-prefix work."""
    values, summaries, counts, miou, tp, fp, fn, sums, confusion = outputs[:9]
    itp, ifp, ifn, precision, recall = outputs[9:14]
    results = []
    for q, k in enumerate(points):
        v, s, n = values[q], summaries[q], counts[q]
        result = {
            name: dict(pq=s[i, 0], sq=s[i, 1], rq=s[i, 2], n=n[i])
            for i, name in enumerate(('All', 'Things', 'Stuff'))}
        result.update(
            prefix=k, miou=miou[q], tp=tp[q], fp=fp[q], fn=fn[q],
            iou_sum=sums[q], confusion=confusion[q],
            per_class={name: v[i] for i, name in enumerate(('pq', 'sq', 'rq', 'iou'))},
            instance=dict(iou_thresholds=thresholds, tp=itp[q], fp=ifp[q], fn=ifn[q],
                          precision=precision[q], recall=recall[q]))
        results.append(result)
    return results
