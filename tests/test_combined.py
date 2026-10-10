import unittest

import torch

from panoptic_evaluator import (
    InstanceBatch,
    InstanceEvaluator,
    PanopticBatch,
    PanopticEvaluator,
    SegmentationEvaluator,
)


class CombinedTests(unittest.TestCase):
    def devices(self):
        return ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]

    def fixture(self, device):
        generator = torch.Generator().manual_seed(91)
        maps = torch.randint(0, 6, (2, 13, 35), generator=generator).to(
            device, torch.int32
        )
        classes = torch.tensor(
            [[-1, 0, 1, 0, 2, -1], [-1, 1, 0, 2, 0, 1]],
            device=device,
            dtype=torch.int32,
        )
        maps[0][maps[0] == 5] = 0
        crowd = torch.tensor(
            [
                [False, False, True, False, False, False],
                [False, False, False, False, True, False],
            ],
            device=device,
        )
        target = PanopticBatch(maps, classes, crowd)
        # Panoptic prediction deliberately differs from the overlapping detections.
        pred_maps = maps.clone()
        pred_maps[:, :3] = 0
        prediction = PanopticBatch(pred_maps, classes.clone())
        masks = [
            torch.rand((8, 13, 35), generator=generator).to(device) > 0.7
            for _ in range(2)
        ]
        for b in range(2):
            masks[b][:3] = (
                maps[b][None] == torch.tensor([1, 2, 3], device=device)[:, None, None]
            )
        masks[0][3] = True  # Includes both void and stuff in detection area.
        labels = [
            torch.tensor([0, 1, 0, 0, 1, 0, 1, 0], device=device, dtype=torch.int64),
            torch.tensor([1, 0, 0, 1, 0, 1, 0, 1], device=device, dtype=torch.int64),
        ]
        scores = [
            torch.tensor([0.9, 0.9, 0.8, 0.5, 0.5, 0.5, 0.1, 0.1], device=device)
            for _ in range(2)
        ]
        return prediction, InstanceBatch(masks, labels, scores), target

    def instance_target(self, target, areas=None):
        masks, classes, crowd, annotation_areas = [], [], [], []
        for b in range(len(target.segment_map)):
            slots = torch.where((target.classes[b] >= 0) & (target.classes[b] < 2))[0]
            masks.append(target.segment_map[b][None] == slots[:, None, None])
            classes.append(target.classes[b, slots])
            crowd.append(target.crowd[b, slots])
            if areas is not None:
                annotation_areas.append(areas[b, slots])
        return InstanceBatch(
            masks,
            classes,
            crowd=crowd,
            areas=annotation_areas if areas is not None else None,
        )

    def compare(self, actual, panoptic, instance):
        for key in ("tp", "fp", "fn", "iou_sum", "confusion", "miou"):
            torch.testing.assert_close(actual[key], panoptic[key], rtol=0, atol=1e-14)
        for key in ("All", "Things", "Stuff", "per_class"):
            for metric in panoptic[key]:
                torch.testing.assert_close(
                    actual[key][metric],
                    panoptic[key][metric],
                    rtol=0,
                    atol=1e-14,
                    equal_nan=True,
                )
        for key, value in instance.items():
            torch.testing.assert_close(
                actual["instance"][key], value, rtol=0, atol=1e-14
            )

    def test_separate_evaluator_parity(self):
        for device in self.devices():
            for validate in (True, False):
                with self.subTest(device=device, validate=validate):
                    p, d, g = self.fixture(device)
                    areas = torch.tensor(
                        [
                            [0.0, 1024.0, 1025.0, 9216.0, 9217.0, 0.0],
                            [0.0, 100.0, 1024.0, 200.0, 9217.0, 9216.0],
                        ],
                        device=device,
                    )
                    joint = SegmentationEvaluator(
                        3, isthing=[True, True, False], device=device, validate=validate
                    )
                    pq = PanopticEvaluator(
                        3, isthing=[True, True, False], device=device, validate=validate
                    )
                    ap = InstanceEvaluator(3, device=device, validate=validate)
                    for annotation_areas in (None, areas):
                        joint.update(p, d, g, instance_areas=annotation_areas)
                        pq.update(p, g)
                        ap.update(d, self.instance_target(g, annotation_areas))
                    self.compare(joint.compute(), pq.compute(), ap.compute())
                    snapshot = joint.compute()
                    joint.reset()
                    self.assertEqual(joint.compute()["instance"]["ap"].item(), -1.0)
                    self.compare(snapshot, pq.compute(), ap.compute())

    def test_original_ids_and_empty_detections(self):
        for device in self.devices():
            p, d, g = self.fixture(device)
            original = g.segment_map.long() * 10000000001
            info = [
                [
                    dict(
                        id=int(slot) * 10000000001,
                        category_id=int(g.classes[b, slot]),
                        iscrowd=int(g.crowd[b, slot]),
                    )
                    for slot in range(1, g.classes.shape[1])
                    if g.classes[b, slot] >= 0
                ]
                for b in range(2)
            ]
            target = PanopticBatch.from_coco(original, info, [0, 1, 2], compact=False)
            empty = InstanceBatch(
                [
                    torch.empty((0, 13, 35), dtype=torch.bool, device=device)
                    for _ in range(2)
                ],
                [torch.empty(0, dtype=torch.int32, device=device) for _ in range(2)],
                [torch.empty(0, device=device) for _ in range(2)],
            )
            joint = SegmentationEvaluator(3, isthing=[True, True, False], device=device)
            joint.update(p, empty, target)
            pq = PanopticEvaluator(3, isthing=[True, True, False], device=device)
            pq.update(p, g)
            ap = InstanceEvaluator(3, device=device)
            ap.update(empty, self.instance_target(g))
            self.compare(joint.compute(), pq.compute(), ap.compute())

    def test_failed_validation_preserves_all_metrics(self):
        for device in self.devices():
            p, d, g = self.fixture(device)
            joint = SegmentationEvaluator(3, isthing=[True, True, False], device=device)
            joint.update(p, d, g)
            before = joint.compute()
            invalid_labels = [x.clone() for x in d.classes]
            invalid_labels[1][-1] = 2
            with self.assertRaisesRegex(ValueError, "thing"):
                joint.update(p, InstanceBatch(d.masks, invalid_labels, d.scores), g)
            invalid_masks = [x.clone() for x in d.masks]
            invalid_masks[1][-1].zero_()
            with self.assertRaisesRegex(ValueError, "pixels"):
                joint.update(p, InstanceBatch(invalid_masks, d.classes, d.scores), g)
            with self.assertRaisesRegex(ValueError, "instance_areas"):
                joint.update(
                    p,
                    d,
                    g,
                    instance_areas=torch.full(
                        g.classes.shape, float("nan"), device=device
                    ),
                )
            self.compare(joint.compute(), before, before["instance"])

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_nondefault_stream_and_snapshot(self):
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            p, d, g = self.fixture("cuda")
            joint = SegmentationEvaluator(
                3, isthing=[True, True, False], device="cuda", validate=False
            )
            pq = PanopticEvaluator(
                3, isthing=[True, True, False], device="cuda", validate=False
            )
            ap = InstanceEvaluator(3, device="cuda", validate=False)
            target_masks = self.instance_target(g)
            pq.update(p, g)
            ap.update(d, target_masks)
            joint.update(p, d, g)
            actual, expected_pq, expected_ap = (
                joint.compute(),
                pq.compute(),
                ap.compute(),
            )
            g.segment_map.zero_()
            for x in d.masks + d.scores:
                x.zero_()
            joint.reset()
        stream.synchronize()
        self.compare(actual, expected_pq, expected_ap)
