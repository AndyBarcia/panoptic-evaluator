from __future__ import annotations

from dataclasses import dataclass, replace
from numbers import Integral
from typing import Sequence

import torch

try:
    from . import _cuda_evaluator
except ImportError:
    _cuda_evaluator = None


def _require_cuda_extension():
    if _cuda_evaluator is None:
        raise RuntimeError(
            "CUDA extension unavailable; run python setup.py build_ext --inplace"
        )
    return _cuda_evaluator


@dataclass(frozen=True)
class PanopticBatch:
    """Segment maps and metadata; optional lookup tables allow original IDs."""

    segment_map: torch.Tensor  # B x H x W; compact int32 or original int32/int64 IDs
    classes: torch.Tensor  # B x S, int32; slot 0 is void, class -1 is padding
    crowd: torch.Tensor | None = None  # B x S, bool; ground truth only
    lookup_ids: torch.Tensor | None = None  # B x S sorted int64 original IDs
    lookup_slots: torch.Tensor | None = None  # B x S int32 original metadata slots

    @classmethod
    def from_coco(
        cls,
        maps: torch.Tensor,
        segments_info: Sequence[Sequence[dict]],
        category_ids: Sequence[int],
        *,
        compact: bool = True,
    ) -> PanopticBatch:
        """Upload metadata once. Set compact=False to fuse ID lookup with counting.

        Metadata keeps input order for official crowd semantics. In the original-ID
        path, pixel validation is deferred to evaluator.update(validate=True).
        """
        if maps.ndim != 3 or maps.dtype not in (torch.int32, torch.int64):
            raise ValueError("maps must be integer B x H x W tensors")
        if len(segments_info) != maps.shape[0] or len(set(category_ids)) != len(
            category_ids
        ):
            raise ValueError("batch length or category IDs are invalid")
        lookup = {category: index for index, category in enumerate(category_ids)}
        capacity = max((len(info) for info in segments_info), default=0) + 1
        limit = torch.iinfo(torch.int64).max
        all_ids, all_slots, all_classes, all_crowd = [], [], [], []
        for info in segments_info:
            ids = [entry["id"] for entry in info]
            if any(
                not isinstance(i, Integral) or i <= 0 or i >= limit for i in ids
            ) or len(set(ids)) != len(ids):
                raise ValueError(
                    "segment IDs must be unique positive int64 values below INT64_MAX; 0 is void"
                )
            try:
                labels = [lookup[entry["category_id"]] for entry in info]
            except KeyError as error:
                raise ValueError("unknown category_id") from error
            padding = capacity - len(info) - 1
            ordered = sorted(
                (identifier, slot) for slot, identifier in enumerate(ids, 1)
            )
            all_ids.extend(
                [0] + [identifier for identifier, _ in ordered] + [limit] * padding
            )
            all_slots.extend([0] + [slot for _, slot in ordered] + [0] * padding)
            all_classes.extend([-1] + labels + [-1] * padding)
            all_crowd.extend(
                [False]
                + [bool(entry.get("iscrowd", 0)) for entry in info]
                + [False] * padding
            )
        # Typed views share one allocation and one host-to-device upload. Keep all
        # int64/int32 fields aligned by placing bool storage last.
        fields = (
            torch.tensor(all_ids, dtype=torch.int64, device="cpu"),
            torch.tensor(all_slots, dtype=torch.int32, device="cpu"),
            torch.tensor(all_classes, dtype=torch.int32, device="cpu"),
            torch.tensor(all_crowd, dtype=torch.bool, device="cpu"),
        )
        storage = torch.cat([field.view(torch.uint8) for field in fields]).to(
            maps.device
        )
        offset = 0
        views = []
        for field in fields:
            size = field.numel() * field.element_size()
            views.append(
                storage[offset : offset + size]
                .view(field.dtype)
                .reshape(maps.shape[0], capacity)
            )
            offset += size
        ids, slots, classes, crowd = views
        result = cls(maps, classes, crowd, ids, slots)
        return result.compact() if compact else result

    def with_maps(self, maps: torch.Tensor) -> PanopticBatch:
        """Reuse prepared metadata for another map batch with the same segment IDs."""
        if (
            maps.ndim != 3
            or maps.shape[0] != self.classes.shape[0]
            or maps.device != self.classes.device
        ):
            raise ValueError(
                "new maps must have the same batch size and metadata device"
            )
        return replace(self, segment_map=maps)

    def compact(self) -> PanopticBatch:
        """Materialize compact slots when needed; CUDA packing uses one pixel kernel."""
        if self.lookup_ids is None:
            return self
        maps = self.segment_map
        if maps.ndim != 3 or maps.dtype not in (torch.int32, torch.int64):
            raise ValueError("maps must be integer B x H x W tensors")
        if (
            self.lookup_slots is None
            or self.lookup_ids.shape != self.classes.shape
            or self.lookup_slots.shape != self.classes.shape
        ):
            raise ValueError("lookup IDs and slots must match classes shape")
        if (
            self.lookup_ids.device != maps.device
            or self.lookup_slots.device != maps.device
            or maps.shape[0] != self.classes.shape[0]
        ):
            raise ValueError("lookup tables and maps must share device and batch size")
        if maps.is_cuda:
            cuda = _require_cuda_extension()
            packed, error = cuda.pack(
                maps.contiguous(), self.lookup_ids, self.lookup_slots
            )
            if error.item():
                raise ValueError("map contains a segment absent from segments_info")
        else:
            packed = torch.empty_like(maps, dtype=torch.int32)
            for b in range(maps.shape[0]):
                ids = self.lookup_ids[b]
                positions = torch.searchsorted(ids, maps[b].long().contiguous()).clamp(
                    max=ids.numel() - 1
                )
                slots = self.lookup_slots[b][positions]
                valid = (ids[positions] == maps[b]) & ((maps[b] == 0) | (slots > 0))
                if not bool(valid.all()):
                    raise ValueError("map contains a segment absent from segments_info")
                packed[b] = slots
        return PanopticBatch(packed, self.classes, self.crowd)


def _prepare(batch: PanopticBatch, num_classes: int, validate: bool, *, ignore_crowd: bool = False, allow_empty_slots: bool = False):
    maps, classes = batch.segment_map, batch.classes
    if (
        maps.ndim != 3
        or classes.ndim != 2
        or classes.shape[0] != maps.shape[0]
        or classes.shape[1] < 1
    ):
        raise ValueError("expected maps B x H x W and classes B x S with S >= 1")
    original_ids = batch.lookup_ids is not None
    if (batch.lookup_slots is not None) != original_ids:
        raise ValueError("lookup IDs and slots must be supplied together")
    map_dtypes = (torch.int32, torch.int64) if original_ids else (torch.int32,)
    if maps.dtype not in map_dtypes or classes.dtype != torch.int32:
        raise ValueError(
            "classes and compact maps must be int32; original ID maps may also be int64"
        )
    if original_ids:
        for lookup, dtype in (
            (batch.lookup_ids, torch.int64),
            (batch.lookup_slots, torch.int32),
        ):
            if (
                lookup.shape != classes.shape
                or lookup.dtype != dtype
                or lookup.device != maps.device
            ):
                raise ValueError(
                    "lookup tables must match metadata shape, device and expected dtypes"
                )
    crowd = (
        None
        if ignore_crowd
        else (
            batch.crowd
            if batch.crowd is not None
            else torch.zeros_like(classes, dtype=torch.bool)
        )
    )
    if crowd is not None and (
        crowd.shape != classes.shape or crowd.dtype != torch.bool
    ):
        raise ValueError("crowd must be bool with the same shape as classes")
    if classes.device != maps.device or (
        crowd is not None and crowd.device != maps.device
    ):
        raise ValueError("batch tensors must share a device")
    if validate:
        if bool(((maps < 0) | (maps >= classes.shape[1])).any()):
            raise ValueError("segment map slot outside metadata capacity")
        if (
            bool(((classes < -1) | (classes >= num_classes)).any())
            or bool((classes[:, 0] != -1).any())
            or (crowd is not None and bool(crowd[:, 0].any()))
        ):
            raise ValueError(
                "classes must be -1 or valid category indices; slot 0 must be void"
            )
        areas = torch.zeros_like(classes, dtype=torch.int64)
        areas.scatter_add_(1, maps.flatten(1).long(), torch.ones_like(maps.flatten(1), dtype=torch.int64))
        invalid = ((areas[:, 1:] > 0) & (classes[:, 1:] < 0) if allow_empty_slots
                   else (areas[:, 1:] > 0) != (classes[:, 1:] >= 0))
        if bool(invalid.any()):
            raise ValueError("nonvoid map slots and active metadata must correspond exactly")
    return maps.contiguous(), classes.contiguous(), crowd.contiguous() if crowd is not None else None


def _histogram(gt, pred, gc, pc, crowd, C, *, pq_only=False):
    B, H, W = gt.shape
    G, P = gc.shape[1], pc.shape[1]
    offsets = torch.arange(B, device=gt.device)[:, None, None] * G * P
    pairs = torch.bincount((offsets + gt.long() * P + pred.long()).flatten(),
                           minlength=B * G * P).reshape(B, G, P)
    if pq_only:
        return pairs
    target = gc.gather(1, gt.flatten(1).long())
    prediction = pc.gather(1, pred.flatten(1).long())
    valid = (target >= 0) & ~crowd.gather(1, gt.flatten(1).long())
    prediction = torch.where(prediction >= 0, prediction, C)
    semantic = torch.bincount(
        (target * (C + 1) + prediction)[valid].long(), minlength=C * (C + 1)
    ).reshape(C, C + 1)
    return pairs, semantic


class PanopticEvaluator:
    """Accumulate dataset PQ/SQ/RQ and semantic IoU without copying pixels to CPU."""

    def __init__(
        self,
        num_classes: int,
        *,
        isthing: Sequence[bool] | None = None,
        device: str | torch.device = "cuda",
        validate: bool = True,
    ):
        if not isinstance(num_classes, int) or num_classes <= 0:
            raise ValueError("num_classes must be a positive integer")
        if isthing is not None and len(isthing) != num_classes:
            raise ValueError("isthing must contain one flag per category")
        self.num_classes = num_classes
        self.device = torch.device(device)
        self.validate = validate
        self.isthing = torch.tensor(
            isthing if isthing is not None else [True] * num_classes,
            dtype=torch.bool,
            device=self.device,
        )
        self._scratch = None
        self._default_crowd = None
        self._validation_error = torch.empty(1, dtype=torch.int32, device=self.device)
        self.reset()

    def reset(self):
        if hasattr(self, "tp"):
            for tensor in (self.tp, self.fp, self.fn, self.iou_sum, self.confusion):
                tensor.zero_()
            return
        C = self.num_classes
        self.iou_sum = torch.zeros(C, dtype=torch.float64, device=self.device)
        self.tp = torch.zeros(C, dtype=torch.int64, device=self.device)
        self.fp = torch.zeros_like(self.tp)
        self.fn = torch.zeros_like(self.tp)
        # Last prediction column records void predictions on valid GT pixels.
        self.confusion = torch.zeros((C, C + 1), dtype=torch.int64, device=self.device)

    @torch.no_grad()
    def update(self, prediction: PanopticBatch, target: PanopticBatch):
        return self._update(prediction, target)

    @torch.no_grad()
    def _update(
        self, prediction: PanopticBatch, target: PanopticBatch, *, return_areas=False
    ):
        if not target.segment_map.is_cuda:
            target = target.compact()
            prediction = prediction.compact()
        if target.segment_map.is_cuda and target.crowd is None:
            if (
                self._default_crowd is None
                or self._default_crowd.shape != target.classes.shape
                or self._default_crowd.device != target.classes.device
            ):
                self._default_crowd = torch.zeros_like(target.classes, dtype=torch.bool)
            target = replace(target, crowd=self._default_crowd)
        # CUDA value checks reuse native histogram/area work and one error read.
        host_validate = self.validate and not target.segment_map.is_cuda
        gt, gc, crowd = _prepare(target, self.num_classes, host_validate)
        pred, pc, _ = _prepare(
            prediction, self.num_classes, host_validate, ignore_crowd=True
        )
        if (
            gt.shape != pred.shape
            or gt.device != pred.device
            or gt.device != self.tp.device
        ):
            raise ValueError(
                "prediction and target must share shape and evaluator device"
            )
        if gt.is_cuda:
            cuda = _require_cuda_extension()
            shape = (gt.shape[0], gc.shape[1], pc.shape[1])
            if self._scratch is None or self._scratch[0].shape != shape:
                B, G, P = shape
                self._scratch = (
                    torch.empty(shape, dtype=torch.int64, device=gt.device),
                    torch.empty((B, G + P), dtype=torch.int64, device=gt.device),
                    torch.empty((B, G), dtype=torch.int32, device=gt.device),
                )
            error = cuda.update(
                gt,
                pred,
                gc,
                pc,
                crowd,
                *self._scratch,
                self.tp,
                self.fp,
                self.fn,
                self.iou_sum,
                self.confusion,
                (
                    target.lookup_ids.contiguous()
                    if target.lookup_ids is not None
                    else None
                ),
                (
                    target.lookup_slots.contiguous()
                    if target.lookup_slots is not None
                    else None
                ),
                (
                    prediction.lookup_ids.contiguous()
                    if prediction.lookup_ids is not None
                    else None
                ),
                (
                    prediction.lookup_slots.contiguous()
                    if prediction.lookup_slots is not None
                    else None
                ),
                self._validation_error,
                self.validate,
            )
            if error:
                if error & 1:
                    raise ValueError(
                        "map contains an unknown segment ID or a slot outside metadata capacity"
                    )
                if error & 2:
                    raise ValueError(
                        "classes must be -1 or valid category indices; slot 0 must be void"
                    )
                raise ValueError(
                    "nonvoid map slots and active metadata must correspond exactly"
                )
            return self._scratch[1][:, : gc.shape[1]] if return_areas else self
        pairs, semantic = _histogram(gt, pred, gc, pc, crowd, self.num_classes)
        ga, pa = pairs.sum(2), pairs.sum(1)
        union = ga[:, :, None] + pa[:, None, :] - pairs - pairs[:, :1, :]
        iou = pairs.double() / union.clamp_min(1)
        matches = (
            (gc[:, :, None] == pc[:, None, :])
            & (gc[:, :, None] >= 0)
            & ~crowd[:, :, None]
            & (iou > 0.5)
        )
        gm, pm = matches.any(2), matches.any(1)
        # Official API keeps the last crowd segment per category in metadata order.
        slots = torch.arange(gc.shape[1], device=gt.device)[None, :, None]
        same_crowd = (
            crowd[:, :, None]
            & (gc[:, :, None] == pc[:, None, :])
            & (gc[:, :, None] >= 0)
        )
        last = torch.where(same_crowd, slots, -1).amax(1)
        overlap = pairs.gather(1, last.clamp_min(0)[:, None, :]).squeeze(1)
        ignored_area = pairs[:, 0, :] + torch.where(last >= 0, overlap, 0)
        false_positive = (pc >= 0) & ~pm & (ignored_area.double() <= 0.5 * pa)
        false_negative = (gc >= 0) & ~crowd & ~gm
        for labels, mask, output in (
            (pc, pm, self.tp),
            (pc, false_positive, self.fp),
            (gc, false_negative, self.fn),
        ):
            output.scatter_add_(
                0, labels.clamp_min(0).flatten().long(), mask.flatten().long()
            )
        self.iou_sum.scatter_add_(
            0, pc.clamp_min(0).flatten().long(), (iou * matches).sum(1).flatten()
        )
        self.confusion += semantic
        return ga if return_areas else self

    @torch.no_grad()
    def compute(self) -> dict[str, torch.Tensor | dict]:
        """Return device tensors in [0, 1]. Absent classes have zero PQ and NaN IoU."""
        if self.tp.is_cuda:
            cuda = _require_cuda_extension()
            values, summaries, counts, miou, tp, fp, fn, sums, confusion = cuda.compute(
                self.tp, self.fp, self.fn, self.iou_sum, self.confusion, self.isthing
            )
            result = {
                name: {
                    "pq": summaries[i, 0],
                    "sq": summaries[i, 1],
                    "rq": summaries[i, 2],
                    "n": counts[i],
                }
                for i, name in enumerate(("All", "Things", "Stuff"))
            }
            result.update(
                per_class={
                    name: values[i] for i, name in enumerate(("pq", "sq", "rq", "iou"))
                },
                miou=miou,
                tp=tp,
                fp=fp,
                fn=fn,
                iou_sum=sums,
                confusion=confusion,
            )
            return result
        denom = self.tp.double() + 0.5 * (self.fp + self.fn).double()
        pq = self.iou_sum / denom.clamp_min(1)
        sq = self.iou_sum / self.tp.clamp_min(1)
        rq = self.tp / denom.clamp_min(1)
        active = denom > 0

        def average(mask):
            selected = mask & active
            n = selected.sum()
            return {"pq": (pq * selected).sum() / n.clamp_min(1),
                    "sq": (sq * selected).sum() / n.clamp_min(1),
                    "rq": (rq * selected).sum() / n.clamp_min(1), "n": n}
        diagonal = self.confusion[:, :self.num_classes].diagonal()
        union = self.confusion.sum(1) + self.confusion[:, :self.num_classes].sum(0) - diagonal
        semantic_iou = torch.where(union > 0, diagonal.double() / union.clamp_min(1), torch.nan)
        return {"All": average(torch.ones_like(active)), "Things": average(self.isthing),
                "Stuff": average(~self.isthing),
                "per_class": {"pq": pq, "sq": sq, "rq": rq, "iou": semantic_iou},
                "miou": torch.nan_to_num(semantic_iou).sum() / (union > 0).sum().clamp_min(1),
                "tp": self.tp.clone(), "fp": self.fp.clone(), "fn": self.fn.clone(),
                "iou_sum": self.iou_sum.clone(), "confusion": self.confusion.clone()}


@torch.no_grad()
def panoptic_quality(prediction: PanopticBatch, target: PanopticBatch,
                     num_classes: int, reduction: str = "none", *, validate: bool = True) -> torch.Tensor:
    """Stateless mean-class PQ per image (float64 on the input device).

    Prepared target and prediction metadata may be reused with ``with_maps``.
    Prediction slots without pixels are ignored, including fully occluded slots.
    Set validate=False for trusted inputs to avoid CUDA value-check synchronization.
    Only reduction="none" is supported; no dataset or semantic metrics are computed.
    """
    if not isinstance(num_classes, int) or num_classes <= 0:
        raise ValueError("num_classes must be a positive integer")
    if reduction != "none":
        raise ValueError('reduction must be "none"')
    if not target.segment_map.is_cuda:
        target, prediction = target.compact(), prediction.compact()
    host_validate = validate and not target.segment_map.is_cuda
    gt, gc, crowd = _prepare(target, num_classes, host_validate)
    pred, pc, _ = _prepare(prediction, num_classes, host_validate,
                           ignore_crowd=True, allow_empty_slots=True)
    if gt.shape != pred.shape or gt.device != pred.device:
        raise ValueError("prediction and target must share shape and device")
    B, G, P = gt.shape[0], gc.shape[1], pc.shape[1]
    tp = torch.zeros((B, num_classes), dtype=torch.int64, device=gt.device)
    fp, fn = torch.zeros_like(tp), torch.zeros_like(tp)
    sums = torch.zeros_like(tp, dtype=torch.float64)
    if gt.is_cuda:
        if _cuda_evaluator is None:
            raise RuntimeError("CUDA extension unavailable; run python setup.py build_ext --inplace")
        error = _cuda_evaluator.update(
            gt, pred, gc, pc, crowd,
            torch.empty((B, G, P), dtype=torch.int64, device=gt.device),
            torch.empty((B, G + P), dtype=torch.int64, device=gt.device),
            torch.empty((B, G), dtype=torch.int32, device=gt.device),
            tp, fp, fn, sums, torch.empty(0, dtype=torch.int64, device=gt.device),
            target.lookup_ids.contiguous() if target.lookup_ids is not None else None,
            target.lookup_slots.contiguous() if target.lookup_slots is not None else None,
            prediction.lookup_ids.contiguous() if prediction.lookup_ids is not None else None,
            prediction.lookup_slots.contiguous() if prediction.lookup_slots is not None else None,
            torch.empty(1, dtype=torch.int32, device=gt.device), validate, True)
        if error:
            if error & 1:
                raise ValueError("map contains an unknown segment ID or a slot outside metadata capacity")
            if error & 2:
                raise ValueError("classes must be -1 or valid category indices; slot 0 must be void")
            raise ValueError("nonvoid map slots and active metadata must correspond exactly")
    else:
        pairs = _histogram(gt, pred, gc, pc, crowd, num_classes, pq_only=True)
        ga, pa = pairs.sum(2), pairs.sum(1)
        union = ga[:, :, None] + pa[:, None, :] - pairs - pairs[:, :1, :]
        iou = pairs.double() / union.clamp_min(1)
        matches = (gc[:, :, None] == pc[:, None, :]) & (gc[:, :, None] >= 0) & ~crowd[:, :, None] & (iou > 0.5)
        gm, pm = matches.any(2), matches.any(1)
        # Official API keeps the last crowd segment per category in metadata order.
        slots = torch.arange(gc.shape[1], device=gt.device)[None, :, None]
        same_crowd = crowd[:, :, None] & (gc[:, :, None] == pc[:, None, :]) & (gc[:, :, None] >= 0)
        last = torch.where(same_crowd, slots, -1).amax(1)
        overlap = pairs.gather(1, last.clamp_min(0)[:, None, :]).squeeze(1)
        ignored_area = pairs[:, 0, :] + torch.where(last >= 0, overlap, 0)
        false_positive = (pc >= 0) & (pa > 0) & ~pm & (ignored_area.double() <= 0.5 * pa)
        false_negative = (gc >= 0) & ~crowd & ~gm
        for labels, mask, output in ((pc, pm, tp), (pc, false_positive, fp),
                                     (gc, false_negative, fn)):
            output.scatter_add_(1, labels.clamp_min(0).long(), mask.long())
        sums.scatter_add_(1, pc.clamp_min(0).long(), (iou * matches).sum(1))
    denominator = tp.double() + 0.5 * (fp + fn).double()
    active = denominator > 0
    pq = sums / denominator.clamp_min(1)
    return pq.sum(1) / active.sum(1).clamp_min(1)
