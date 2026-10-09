import unittest
from unittest.mock import patch

import torch

from panoptic_evaluator import PanopticBatch, PanopticEvaluator, panoptic_quality


class RewardTests(unittest.TestCase):
    device = 'cpu'

    def test_image_parity_and_metadata_reuse(self):
        # Exact half; above half; void subtraction; crowd ignore; all void;
        # empty prediction; occluded slot; different active classes per image.
        gt = torch.tensor([[[1, 1, 1, 1]], [[1, 1, 1, 0]],
                           [[1, 0, 0, 0]], [[2, 2, 0, 0]],
                           [[0, 0, 0, 0]], [[1, 1, 1, 1]],
                           [[1, 1, 1, 1]], [[1, 1, 2, 2]]],
                          dtype=torch.int32, device=self.device)
        pred = torch.tensor([[[1, 1, 0, 0]], [[1, 1, 0, 0]],
                             [[1, 1, 1, 1]], [[2, 2, 2, 2]],
                             [[0, 0, 0, 0]], [[0, 0, 0, 0]],
                             [[1, 1, 1, 1]], [[1, 1, 2, 2]]],
                            dtype=torch.int32, device=self.device)
        gc = torch.tensor([[-1, 0, -1], [-1, 0, -1], [-1, 0, -1],
                           [-1, -1, 1], [-1, -1, -1], [-1, 0, -1],
                           [-1, 0, -1], [-1, 0, 2]], dtype=torch.int32, device=self.device)
        pc = torch.tensor([[-1, 0, 2]] * 8, dtype=torch.int32, device=self.device)
        pc[3, 2] = 1
        crowd = torch.zeros_like(gc, dtype=torch.bool)
        crowd[3, 2] = True
        target, prediction = PanopticBatch(gt, gc, crowd), PanopticBatch(pred, pc)
        expected = []
        for b in range(8):
            present = torch.bincount(pred[b].flatten().long(), minlength=3) > 0
            labels = torch.where(present, pc[b], -1)[None]
            evaluator = PanopticEvaluator(4, device=self.device)
            evaluator.update(PanopticBatch(pred[b:b+1], labels),
                             PanopticBatch(gt[b:b+1], gc[b:b+1], crowd[b:b+1]))
            expected.append(evaluator.compute()['All']['pq'])
        expected = torch.stack(expected)
        for validate in (True, False):
            result = panoptic_quality(prediction, target, 4, validate=validate)
            torch.testing.assert_close(result, expected)
            self.assertEqual(result.shape, (8,))
            self.assertEqual(result.device, gt.device)
        self.assertEqual(expected[0].item(), 0)
        self.assertAlmostEqual(expected[1].item(), 2/3)
        self.assertEqual(expected[2].item(), 1)
        # Prepared arbitrary IDs, including IDs above int32 and metadata order.
        ids = [0, 2**40 + 9, 71]
        gi, pi = [], []
        for b in range(8):
            gi.append([dict(id=ids[s], category_id=int(gc[b, s]),
                            iscrowd=bool(crowd[b, s])) for s in (1, 2) if gc[b, s] >= 0])
            pi.append([dict(id=ids[s], category_id=int(pc[b, s])) for s in (1, 2)])
        original = torch.tensor(ids, device=self.device)
        prepared_gt = PanopticBatch.from_coco(original[gt.long()], gi, range(4), compact=False)
        prepared_pred = PanopticBatch.from_coco(original[pred.long()], pi, range(4), compact=False)
        torch.testing.assert_close(panoptic_quality(prepared_pred, prepared_gt, 4), expected)
        empty = prepared_pred.with_maps(torch.zeros_like(prepared_pred.segment_map))
        torch.testing.assert_close(panoptic_quality(empty, prepared_gt, 4), torch.zeros_like(expected))
        torch.testing.assert_close(panoptic_quality(prepared_pred, prepared_gt, 4), expected)

    def test_randomized_image_parity(self):
        generator = torch.Generator().manual_seed(19)
        for _ in range(8):
            gt = torch.randint(0, 7, (5, 9, 13), generator=generator).to(self.device).int()
            pred = gt.clone()
            pred[:, ::3] = torch.randint(0, 9, (5, 3, 13), generator=generator).to(self.device).int()
            gc = torch.randint(0, 4, (5, 7), generator=generator).to(self.device).int()
            pc = torch.cat((gc.clone(), torch.zeros((5, 2), device=self.device, dtype=torch.int32)), 1)
            gc[:, 0] = pc[:, 0] = -1
            crowd = (torch.rand((5, 7), generator=generator) < .25).to(self.device)
            crowd[:, 0] = False
            expected = []
            for b in range(5):
                evaluator = PanopticEvaluator(5, device=self.device)
                present = torch.bincount(pred[b].flatten().long(), minlength=9) > 0
                labels = torch.where(present, pc[b], -1)[None]
                evaluator.update(PanopticBatch(pred[b:b+1], labels),
                                 PanopticBatch(gt[b:b+1], gc[b:b+1], crowd[b:b+1]))
                expected.append(evaluator.compute()['All']['pq'])
            torch.testing.assert_close(
                panoptic_quality(PanopticBatch(pred, pc), PanopticBatch(gt, gc, crowd), 5),
                torch.stack(expected))

    def test_pq_only_and_invalid_inputs(self):
        batch = PanopticBatch(torch.ones((1, 1, 2), device=self.device, dtype=torch.int32),
                              torch.tensor([[-1, 0]], device=self.device, dtype=torch.int32))
        with patch.object(PanopticEvaluator, 'compute', side_effect=AssertionError):
            self.assertEqual(panoptic_quality(batch, batch, 1).item(), 1)
        with self.assertRaises(ValueError):
            panoptic_quality(batch, batch, 1, reduction='mean')
        with self.assertRaises(ValueError):
            panoptic_quality(batch.with_maps(torch.full_like(batch.segment_map, 9)), batch, 1)
        # The accumulating API continues rejecting unused active slots.
        with self.assertRaises(ValueError):
            PanopticEvaluator(1, device=self.device).update(
                batch.with_maps(torch.zeros_like(batch.segment_map)), batch)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
class CUDARewardTests(RewardTests):
    device = 'cuda'
