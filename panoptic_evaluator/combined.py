"""Joint panoptic, semantic and instance evaluation with a shared target map."""

from __future__ import annotations

from typing import Sequence

import torch

from .evaluator import (
    PanopticBatch,
    PanopticEvaluator,
    _prepare,
    _require_cuda_extension,
)
from .instance import InstanceBatch, InstanceEvaluator, _match_ious_cpu


class SegmentationEvaluator:
    """PQ/mIoU and mask AP/AR with separate predictions and one shared target.

    AP ground truth is exactly the thing segments of the panoptic target. Instance
    annotations that overlap or differ from those segments need InstanceEvaluator.
    """

    def __init__(
        self,
        num_classes: int,
        *,
        isthing: Sequence[bool] | None = None,
        device: str | torch.device = "cuda",
        validate: bool = True,
    ):
        self.panoptic = PanopticEvaluator(
            num_classes, isthing=isthing, device=device, validate=validate
        )
        self.device = self.panoptic.tp.device
        self.instance = InstanceEvaluator(
            num_classes, device=self.device, validate=validate
        )
        self.num_classes = num_classes
        self.validate = validate

    def reset(self):
        self.panoptic.reset()
        self.instance.reset()

    def _validate_areas(self, areas, classes):
        if areas is None:
            return
        if (
            areas.shape != classes.shape
            or areas.device != self.device
            or not areas.is_floating_point()
        ):
            raise ValueError(
                "instance_areas must be floating point B x S on evaluator device"
            )
        if self.validate and (
            not bool(torch.isfinite(areas).all()) or bool((areas < 0).any())
        ):
            raise ValueError("instance_areas must be finite and nonnegative")

    @torch.no_grad()
    def update(
        self,
        prediction: PanopticBatch,
        detections: InstanceBatch,
        target: PanopticBatch,
        *,
        instance_areas: torch.Tensor | None = None,
    ):
        """Accumulate all metrics; optional annotation areas have shape B x S."""
        target = target.compact()
        maps, classes, crowd = _prepare(
            target, self.num_classes, self.validate and self.device.type == "cpu"
        )
        if maps.device != self.device:
            raise ValueError("target must share evaluator device")
        prepared = self.instance._prepare(detections, prediction=True, defer_masks=True)
        if len(prepared) != len(maps) or any(
            image.masks.shape[1:] != maps.shape[1:] for image in prepared
        ):
            raise ValueError(
                "detections and target must have matching image counts and dimensions"
            )
        self._validate_areas(instance_areas, classes)
        cuda = _require_cuda_extension() if self.device.type == "cuda" else None
        isthing = self.panoptic.isthing

        histograms, detection_areas = [], []
        for image, target_map in zip(prepared, maps):
            if self.validate and bool((~isthing[image.classes.long()]).any()):
                raise ValueError("detection categories must be thing classes")
            if cuda is not None:
                pairs = cuda.instance_histogram(
                    image.masks, target_map, image.order, classes.shape[1]
                )
            else:
                pairs = torch.zeros(
                    (len(image.masks), classes.shape[1]), dtype=torch.int64
                )
                for rank, index in enumerate(image.order.tolist()):
                    pairs[rank] = torch.bincount(
                        target_map[image.masks[index]].long(),
                        minlength=classes.shape[1],
                    )
            # AP uses full detection areas, including overlap with void and stuff.
            areas = pairs.sum(1).double()
            if self.validate and bool((areas == 0).any()):
                raise ValueError("each instance mask must have pixels")
            histograms.append(pairs)
            detection_areas.append(areas)

        # Validate instance inputs before changing totals. Consume the shared
        # scratch areas before the next panoptic update overwrites them.
        target_areas = self.panoptic._update(prediction, target, return_areas=True)
        valid_classes = classes.clamp(0, self.num_classes - 1).long()
        gt_classes = torch.where((classes >= 0) & isthing[valid_classes], classes, -1)
        records, counts = [], torch.zeros_like(self.instance._counts)
        for index, (image, pairs, prediction_area) in enumerate(
            zip(prepared, histograms, detection_areas)
        ):
            target_class = gt_classes[index].contiguous()
            target_area = target_areas[index].double().contiguous()
            target_crowd = crowd[index].contiguous()
            annotation_area = (
                target_area
                if instance_areas is None
                else instance_areas[index].double().contiguous()
            )
            if cuda is not None:
                status, ranks, positive = cuda.instance_match_histogram(
                    pairs,
                    image.classes,
                    target_class,
                    prediction_area,
                    target_area,
                    target_crowd,
                    annotation_area,
                    self.num_classes,
                )
            else:
                intersections = pairs.double()
                union = torch.where(
                    target_crowd[None],
                    prediction_area[:, None],
                    prediction_area[:, None] + target_area[None] - intersections,
                )
                status, ranks, positive = _match_ious_cpu(
                    intersections / union.clamp_min(1),
                    image.classes,
                    target_class,
                    prediction_area,
                    target_crowd,
                    annotation_area,
                    self.num_classes,
                )
            records.append(self.instance._snapshot(image, ranks, status))
            counts += positive
        self.instance._commit(records, counts)
        return self

    def compute(self):
        """Return panoptic/semantic results plus an 'instance' AP/AR dictionary."""
        result = self.panoptic.compute()
        result["instance"] = self.instance.compute()
        return result
