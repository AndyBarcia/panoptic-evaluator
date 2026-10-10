"""Individual mask inputs and COCO instance segmentation evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, Sequence

import torch

from .evaluator import _require_cuda_extension


@dataclass(frozen=True)
class InstanceBatch:
    """One N x H x W bool tensor per image, with N category indices.

    Predictions require floating point scores. Targets accept crowd flags and
    annotation areas (COCO's area filtering uses annotation area for targets).
    Masks may overlap. Empty images use tensors with N=0.
    """

    masks: Sequence[torch.Tensor]
    classes: Sequence[torch.Tensor]
    scores: Sequence[torch.Tensor] | None = None
    crowd: Sequence[torch.Tensor] | None = None
    areas: Sequence[torch.Tensor] | None = None


class _PreparedImage(NamedTuple):
    masks: torch.Tensor
    classes: torch.Tensor
    pixel_areas: torch.Tensor | None
    crowd: torch.Tensor
    annotation_areas: torch.Tensor | None
    scores: torch.Tensor | None
    order: torch.Tensor | None


class _InstanceRecord(NamedTuple):
    scores: torch.Tensor
    classes: torch.Tensor
    ranks: torch.Tensor
    status: torch.Tensor


class InstanceEvaluator:
    """COCO mask AP/AR with native CUDA intersections and greedy matching.

    Only scores, matches and counts are retained between updates, not masks.
    CPU tensors use a reference implementation without pycocotools.
    """

    def __init__(
        self,
        num_classes: int,
        *,
        device: str | torch.device = "cpu",
        validate: bool = True,
    ):
        if (
            not isinstance(num_classes, int)
            or isinstance(num_classes, bool)
            or num_classes <= 0
        ):
            raise ValueError("num_classes must be a positive integer")
        self.num_classes = num_classes
        self.device = torch.device(device)
        self.validate = validate
        self.reset()
        self.device = self._counts.device

    def reset(self):
        self._records = []
        self._counts = torch.zeros(
            (self.num_classes, 4), dtype=torch.int64, device=self.device
        )

    def _prepare(self, batch, *, prediction, defer_masks=False):
        count = len(batch.masks)
        for field in (batch.classes, batch.scores, batch.crowd, batch.areas):
            if field is not None and len(field) != count:
                raise ValueError("all fields must contain one tensor per image")
        if prediction and batch.scores is None:
            raise ValueError("predictions require scores")
        prepared = []
        for b, (masks, classes) in enumerate(zip(batch.masks, batch.classes)):
            if (
                masks.ndim != 3
                or masks.dtype != torch.bool
                or min(masks.shape[1:]) <= 0
            ):
                raise ValueError(
                    "masks must be bool N x H x W with positive image dimensions"
                )
            n = masks.shape[0]
            if classes.shape != (n,) or classes.dtype not in (torch.int32, torch.int64):
                raise ValueError("classes must be int32/int64 N tensors")
            for name in ("scores", "crowd", "areas"):
                field = getattr(batch, name)
                if field is None:
                    continue
                value = field[b]
                if value.shape != (n,) or value.device != self.device:
                    raise ValueError(
                        f"{name} must have shape N and share evaluator device"
                    )
                if name == "crowd":
                    if value.dtype != torch.bool:
                        raise ValueError("crowd must be bool")
                elif not value.is_floating_point():
                    raise ValueError(f"{name} must be floating point")
                elif self.validate and (
                    not bool(torch.isfinite(value).all())
                    or (name == "areas" and bool((value < 0).any()))
                ):
                    raise ValueError(
                        f"{name} must be finite; areas must be nonnegative"
                    )
            if masks.device != self.device or classes.device != self.device:
                raise ValueError("inputs must share evaluator device")
            area = None if defer_masks else masks.flatten(1).sum(1).double()
            if self.validate:
                if bool(((classes < 0) | (classes >= self.num_classes)).any()):
                    raise ValueError("classes must be valid category indices")
                if area is not None and bool((area == 0).any()):
                    raise ValueError("each instance mask must have pixels")
            crowd = (
                batch.crowd[b]
                if not prediction and batch.crowd is not None
                else torch.zeros(n, dtype=torch.bool, device=self.device)
            )
            annotation_area = (
                batch.areas[b].double()
                if not prediction and batch.areas is not None
                else area
            )
            scores = batch.scores[b].double() if prediction else None
            order = None
            if prediction:
                order = torch.argsort(scores, descending=True, stable=True)
                classes, scores = classes[order], scores[order]
                if not defer_masks:
                    masks, area = masks[order], area[order]
            prepared.append(
                _PreparedImage(
                    masks=masks.contiguous(),
                    classes=classes.to(torch.int32).contiguous(),
                    pixel_areas=area,
                    crowd=crowd.contiguous(),
                    annotation_areas=(
                        annotation_area.contiguous()
                        if annotation_area is not None
                        else None
                    ),
                    scores=scores,
                    order=order,
                )
            )
        return prepared

    def _commit(self, records, counts):
        self._records.extend(records)
        self._counts += counts

    @staticmethod
    def _snapshot(image, ranks, status):
        return _InstanceRecord(
            image.scores.clone(), image.classes.clone(), ranks, status
        )

    @torch.no_grad()
    def update(self, prediction: InstanceBatch, target: InstanceBatch):
        predictions = self._prepare(prediction, prediction=True)
        targets = self._prepare(target, prediction=False)
        if len(predictions) != len(targets) or any(
            p.masks.shape[1:] != g.masks.shape[1:] for p, g in zip(predictions, targets)
        ):
            raise ValueError(
                "prediction and target must have matching image counts and dimensions"
            )

        match = (
            _require_cuda_extension().instance_update
            if self.device.type == "cuda"
            else _match_cpu
        )
        records, counts = [], torch.zeros_like(self._counts)
        for prediction, target in zip(predictions, targets):
            status, ranks, positive = match(
                prediction.masks,
                target.masks,
                prediction.classes,
                target.classes,
                prediction.pixel_areas,
                target.pixel_areas,
                target.crowd,
                target.annotation_areas,
                self.num_classes,
            )
            records.append(self._snapshot(prediction, ranks, status))
            counts += positive
        self._commit(records, counts)
        return self

    @torch.no_grad()
    def compute(self):
        """Return COCO AP/AR summaries and full precision/recall arrays."""
        if self._records:
            scores = torch.cat([record.scores for record in self._records])
            classes = torch.cat([record.classes for record in self._records])
            ranks = torch.cat([record.ranks for record in self._records])
            status = torch.cat([record.status for record in self._records], dim=1)
            order = torch.argsort(scores, descending=True, stable=True)
            classes, ranks = classes[order], ranks[order]
            status = status[:, order].contiguous()
        else:
            classes = torch.empty(0, dtype=torch.int32, device=self.device)
            ranks = torch.empty_like(classes)
            status = torch.empty((40, 0), dtype=torch.int8, device=self.device)

        accumulate = (
            _require_cuda_extension().instance_compute
            if self.device.type == "cuda"
            else _accumulate_cpu
        )
        precision, recall = accumulate(classes, ranks, status, self._counts)
        return _summarize(precision, recall)


def _mean_valid(values):
    valid = values >= 0
    total = torch.where(valid, values, 0).sum()
    return torch.where(valid.any(), total / valid.sum().clamp_min(1), -1.0)


def _summarize(precision, recall):
    result = {
        "ap": _mean_valid(precision[:, :, :, 0, 2]),
        "ap50": _mean_valid(precision[0, :, :, 0, 2]),
        "ap75": _mean_valid(precision[5, :, :, 0, 2]),
    }
    for area, name in enumerate(("small", "medium", "large"), 1):
        result[f"ap_{name}"] = _mean_valid(precision[:, :, :, area, 2])
        result[f"ar_{name}"] = _mean_valid(recall[:, :, area, 2])
    for index, maximum in enumerate((1, 10, 100)):
        result[f"ar{maximum}"] = _mean_valid(recall[:, :, 0, index])
    result.update(precision=precision, recall=recall)
    return result


_RANGES = ((0.0, 1e10), (0.0, 1024.0), (1024.0, 9216.0), (9216.0, 1e10))
_IOU_THRESHOLDS = (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.8999999999999999, 0.95)


def _match_cpu(pm, gm, pc, gc, pa, ga, crowd, areas, classes):
    d, g = len(pm), len(gm)
    intersections = torch.zeros((d, g), dtype=torch.float64)
    for i in range(d):
        intersections[i] = (pm[i : i + 1] & gm).flatten(1).sum(1)
    union = torch.where(
        crowd[None], pa[:, None], pa[:, None] + ga[None] - intersections
    )
    ious = intersections / union.clamp_min(1)
    return _match_ious_cpu(ious, pc, gc, pa, crowd, areas, classes)


def _match_ious_cpu(ious, pc, gc, pa, crowd, areas, classes):
    d = len(pc)
    status = torch.full((40, d), -1, dtype=torch.int8)
    ranks = torch.zeros(d, dtype=torch.int32)
    counts = torch.zeros((classes, 4), dtype=torch.int64)
    for c in range(classes):
        ds = torch.where(pc == c)[0].tolist()[:100]
        for rank, i in enumerate(ds):
            ranks[i] = rank + 1
        gs = torch.where(gc == c)[0].tolist()
        for a, (low, high) in enumerate(_RANGES):
            ignored = crowd | (areas < low) | (areas > high)
            ordered = sorted(gs, key=lambda j: bool(ignored[j]))
            counts[c, a] = sum(not bool(ignored[j]) for j in gs)
            for t in range(10):
                used = set()
                for i in ds:
                    best, overlap = -1, _IOU_THRESHOLDS[t]
                    for j in ordered:
                        if j in used and not crowd[j]:
                            continue
                        if best >= 0 and not ignored[best] and ignored[j]:
                            break
                        if ious[i, j] >= overlap:
                            best, overlap = j, float(ious[i, j])
                    if best >= 0:
                        used.add(best)
                        status[a * 10 + t, i] = -1 if ignored[best] else 1
                    elif low <= pa[i] <= high:
                        status[a * 10 + t, i] = 0
    return status, ranks, counts


def _accumulate_cpu(classes, ranks, status, counts):
    C = counts.shape[0]
    precision = torch.full((10, 101, C, 4, 3), -1.0, dtype=torch.float64)
    recall = torch.full((10, C, 4, 3), -1.0, dtype=torch.float64)
    thresholds = torch.arange(101, dtype=torch.float64) * 0.01
    for c in range(C):
        for a in range(4):
            if not counts[c, a]:
                continue
            for m, maximum in enumerate((1, 10, 100)):
                selected = (classes == c) & (ranks > 0) & (ranks <= maximum)
                flags = status[a * 10 : a * 10 + 10, selected]
                tp, fp = (flags == 1).cumsum(1).double(), (flags == 0).cumsum(
                    1
                ).double()
                if flags.shape[1] == 0:
                    precision[:, :, c, a, m] = 0
                    recall[:, c, a, m] = 0
                    continue
                rc = tp / counts[c, a]
                pr = tp / (tp + fp + torch.finfo(torch.float64).eps)
                envelope = pr.flip(1).cummax(1).values.flip(1)
                positions = torch.searchsorted(
                    rc.contiguous(), thresholds.expand(10, -1).contiguous()
                )
                precision[:, :, c, a, m] = torch.where(
                    positions < flags.shape[1],
                    envelope.gather(1, positions.clamp_max(flags.shape[1] - 1)),
                    0,
                )
                recall[:, c, a, m] = rc[:, -1]
    return precision, recall
