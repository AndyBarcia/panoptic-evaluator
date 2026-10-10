import unittest

import torch

from panoptic_evaluator import InstanceBatch, InstanceEvaluator

try:
    import pycocotools
except ImportError:
    pycocotools = None


class InstanceTests(unittest.TestCase):
    def batch(self, *, scores=None, crowd=None, device="cpu"):
        masks = torch.zeros((2, 16, 16), dtype=torch.bool, device=device)
        masks[0, :10, :10] = True
        masks[1, 5:15, 5:15] = True
        return InstanceBatch(
            [masks],
            [torch.zeros(2, dtype=torch.int64, device=device)],
            scores=(
                [torch.tensor(scores, device=device)] if scores is not None else None
            ),
            crowd=[torch.tensor(crowd, device=device)] if crowd is not None else None,
        )

    def test_overlapping_perfect_masks(self):
        evaluator = InstanceEvaluator(2)
        evaluator.update(self.batch(scores=[0.9, 0.8]), self.batch())
        result = evaluator.compute()
        self.assertAlmostEqual(result["ap"].item(), 1.0)
        self.assertEqual(result["precision"].shape, (10, 101, 2, 4, 3))
        self.assertTrue((result["precision"][:, :, 1] == -1).all())
        evaluator.reset()
        self.assertEqual(evaluator.compute()["ap"].item(), -1.0)
        self.assertAlmostEqual(result["ap"].item(), 1.0)

    def test_missing_predictions(self):
        empty = InstanceBatch(
            [torch.empty((0, 16, 16), dtype=torch.bool)],
            [torch.empty(0, dtype=torch.int64)],
            [torch.empty(0)],
        )
        evaluator = InstanceEvaluator(1)
        evaluator.update(empty, self.batch())
        self.assertEqual(evaluator.compute()["ap"].item(), 0.0)

    def test_crowd_and_accumulation(self):
        evaluator = InstanceEvaluator(1)
        evaluator.update(self.batch(scores=[0.9, 0.8]), self.batch(crowd=[True, False]))
        evaluator.update(self.batch(scores=[0.9, 0.8]), self.batch())
        self.assertAlmostEqual(evaluator.compute()["ap"].item(), 1.0)

    def test_invalid_update_is_atomic(self):
        evaluator = InstanceEvaluator(1)
        with self.assertRaises(ValueError):
            evaluator.update(self.batch(), self.batch())
        self.assertEqual(evaluator.compute()["ap"].item(), -1.0)

    def test_scores_rank_false_positive_before_true_positive(self):
        target = self.batch()
        target = InstanceBatch([target.masks[0][:1]], [target.classes[0][:1]])
        prediction = self.batch(scores=[0.8, 0.9])
        evaluator = InstanceEvaluator(1)
        evaluator.update(prediction, target)
        self.assertAlmostEqual(evaluator.compute()["ap"].item(), 0.5)
        evaluator.reset()
        evaluator.update(self.batch(scores=[0.9, 0.8]), target)
        self.assertAlmostEqual(evaluator.compute()["ap"].item(), 1.0)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_cuda_inputs(self):
        evaluator = InstanceEvaluator(1, device="cuda")
        evaluator.update(
            self.batch(scores=[0.9, 0.8], device="cuda"), self.batch(device="cuda")
        )
        self.assertAlmostEqual(evaluator.compute()["ap"].item(), 1.0)
        self.assertTrue(evaluator.compute()["ap"].is_cuda)

    @unittest.skipIf(pycocotools is None, "requires pycocotools reference")
    def test_randomized_cocoeval_parity(self):
        import numpy as np
        from contextlib import redirect_stdout
        from io import StringIO
        from pycocotools import mask as mask_utils
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        rng = np.random.default_rng(47)
        targets, predictions, images, batches = [], [], [], []
        for image_id in range(1, 9):
            h, w = 24, 37
            images.append(dict(id=image_id, height=h, width=w))
            gm = rng.random((6, h, w)) > 0.65
            pm = np.concatenate([gm[:4].copy(), rng.random((5, h, w)) > 0.65])
            # Vary IoU, include duplicate masks and exact ties.
            pm[1] &= rng.random((h, w)) > 0.2
            pm[2] = gm[0]
            gc = rng.integers(0, 3, 6)
            pc = np.concatenate([gc[:4], rng.integers(0, 3, 5)])
            pc[2] = gc[0]
            scores = rng.choice([0.1, 0.5, 0.9], 9)
            crowd = rng.random(6) < 0.3
            areas = rng.choice([100.0, 1024.0, 1025.0, 9216.0, 9217.0], 6)
            for masks, labels, anns, is_target in (
                (gm, gc, targets, True),
                (pm, pc, predictions, False),
            ):
                for i, (pixels, label) in enumerate(zip(masks, labels)):
                    rle = mask_utils.encode(np.asfortranarray(pixels, dtype=np.uint8))
                    anns.append(
                        dict(
                            id=len(anns) + 1,
                            image_id=image_id,
                            category_id=int(label),
                            segmentation=rle,
                            area=(
                                float(areas[i])
                                if is_target
                                else float(mask_utils.area(rle))
                            ),
                            iscrowd=int(crowd[i]) if is_target else 0,
                            score=0.0 if is_target else float(scores[i]),
                        )
                    )
            batches.append(
                (
                    InstanceBatch(
                        [torch.tensor(pm)], [torch.tensor(pc)], [torch.tensor(scores)]
                    ),
                    InstanceBatch(
                        [torch.tensor(gm)],
                        [torch.tensor(gc)],
                        crowd=[torch.tensor(crowd)],
                        areas=[torch.tensor(areas)],
                    ),
                )
            )
        gt, dt = COCO(), COCO()
        for coco, annotations in ((gt, targets), (dt, predictions)):
            coco.dataset = dict(
                images=images,
                categories=[dict(id=c) for c in range(4)],
                annotations=annotations,
            )
        with redirect_stdout(StringIO()):
            gt.createIndex()
            dt.createIndex()
            reference = COCOeval(gt, dt, "segm")
            reference.evaluate()
            reference.accumulate()
            reference.summarize()
        for device in (["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]):
            evaluator = InstanceEvaluator(4, device=device)

            def move(batch):
                return InstanceBatch(
                    **{
                        name: (
                            [x.to(device) for x in value] if value is not None else None
                        )
                        for name, value in vars(batch).items()
                    }
                )

            for pred, target in batches:
                evaluator.update(move(pred), move(target))
            result = evaluator.compute()
            np.testing.assert_allclose(
                result["precision"].cpu(),
                reference.eval["precision"],
                atol=1e-14,
                rtol=0,
            )
            np.testing.assert_allclose(
                result["recall"].cpu(), reference.eval["recall"], atol=1e-14, rtol=0
            )

    def test_max_detections_per_category(self):
        # A low-ranked category-1 detection must survive the category-0 cap.
        masks = torch.ones((106, 1, 1), dtype=torch.bool)
        labels = torch.zeros(106, dtype=torch.int64)
        labels[-1] = 1
        for device in (["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]):
            evaluator = InstanceEvaluator(2, device=device)
            evaluator.update(
                InstanceBatch(
                    [masks.to(device)],
                    [labels.to(device)],
                    [torch.ones(106, device=device)],
                ),
                InstanceBatch([masks.to(device)], [labels.to(device)]),
            )
            result = evaluator.compute()
            self.assertAlmostEqual(result["ar1"].item(), (1 / 105 + 1) / 2)
            self.assertAlmostEqual(result["ar10"].item(), (10 / 105 + 1) / 2)
            self.assertAlmostEqual(result["ar100"].item(), (100 / 105 + 1) / 2)

    def test_exact_iou_threshold_and_empty_target(self):
        for device in (["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]):
            pm = torch.tensor([[[True, False]]], device=device)
            gm = torch.tensor([[[True, True]]], device=device)
            label = torch.tensor([0], device=device)
            pred = InstanceBatch([pm], [label], [torch.tensor([0.9], device=device)])
            evaluator = InstanceEvaluator(1, device=device)
            evaluator.update(pred, InstanceBatch([gm], [label]))
            result = evaluator.compute()
            self.assertAlmostEqual(result["ap50"].item(), 1.0)
            self.assertAlmostEqual(result["ap"].item(), 0.1)
            evaluator.reset()
            evaluator.update(pred, InstanceBatch([gm[:0]], [label[:0]]))
            self.assertEqual(evaluator.compute()["ap"].item(), -1.0)
            evaluator.reset()
            evaluator.update(
                InstanceBatch([pm[:0]], [label[:0]], [torch.empty(0, device=device)]),
                InstanceBatch([gm], [label]),
            )
            self.assertEqual(evaluator.compute()["ap"].item(), 0.0)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_nondefault_stream_snapshots_and_validation(self):
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            prediction = self.batch(scores=[0.9, 0.8], device="cuda")
            target = self.batch(device="cuda")
            evaluator = InstanceEvaluator(1, device="cuda", validate=False)
            evaluator.update(prediction, target)
            prediction.masks[0].zero_()
            prediction.scores[0].zero_()
            result = evaluator.compute()
            evaluator.reset()
        stream.synchronize()
        self.assertAlmostEqual(result["ap"].item(), 1.0)
        evaluator = InstanceEvaluator(1, device="cuda")
        with self.assertRaises(ValueError):
            evaluator.update(prediction, target)
        self.assertEqual(evaluator.compute()["ap"].item(), -1.0)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_native_matcher_adversarial_parity(self):
        from panoptic_evaluator.evaluator import _cuda_evaluator
        from panoptic_evaluator.instance import _match_ious_cpu

        generator = torch.Generator().manual_seed(802)
        # Includes category groups larger than a warp, the top-100 cap, exact
        # threshold ties, eligible-vs-ignored preference, and repeatable crowds.
        for D, G, C in (
            (0, 65, 3),
            (105, 0, 3),
            (113, 97, 3),
            (131, 137, 1),
            (17, 99, 1),
        ):
            with self.subTest(detections=D, targets=G, classes=C):
                pc = torch.randint(C, (D,), generator=generator, dtype=torch.int32)
                gc = torch.randint(C, (G,), generator=generator, dtype=torch.int32)
                crowd = torch.rand(G, generator=generator) < 0.2
                choices = torch.tensor(
                    [0.0, 1.0, 1024.0, 1025.0, 9216.0, 9217.0, 1e10, 1e10 + 1],
                    dtype=torch.float64,
                )
                ga = choices[torch.randint(len(choices), (G,), generator=generator)]
                # Exact .5 noncrowd ties span multiple lane strides; crowds have
                # IoU 1. The last case varies IoUs across those strides.
                pairs = torch.ones((D, G), dtype=torch.int64)
                pixel_ga = torch.full((G,), 2.0, dtype=torch.float64)
                pixel_pa = torch.ones(D, dtype=torch.float64)
                if D == 17:
                    pairs = torch.randint(0, 11, (D, G), generator=generator)
                    pixel_ga.fill_(10.0)
                    pixel_pa.fill_(10.0)
                union = torch.where(
                    crowd[None],
                    pixel_pa[:, None],
                    pixel_pa[:, None] + pixel_ga[None] - pairs,
                )
                ious = pairs.double() / union.clamp_min(1)
                # Annotation areas exercise ignore rules independently of the
                # geometric areas used for IoU.
                expected = _match_ious_cpu(ious, pc, gc, pixel_pa, crowd, ga, C)
                actual = _cuda_evaluator.instance_match_histogram(
                    pairs.cuda(),
                    pc.cuda(),
                    gc.cuda(),
                    pixel_pa.cuda(),
                    pixel_ga.cuda(),
                    crowd.cuda(),
                    ga.cuda(),
                    C,
                )
                torch.testing.assert_close(actual[0].cpu(), expected[0], rtol=0, atol=0)
                torch.testing.assert_close(actual[2].cpu(), expected[2], rtol=0, atol=0)
                # CPU marks truncated ranks as zero; both representations are
                # excluded by maxDets. Compare their effective retained ranks.
                effective = torch.where(actual[1].cpu() <= 100, actual[1].cpu(), 0)
                torch.testing.assert_close(effective, expected[1], rtol=0, atol=0)
