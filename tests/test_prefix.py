import unittest

import torch

from panoptic_evaluator import (InstanceBatch, PanopticBatch, PanopticEvaluator,
                                segmentation_prefixes)


class PrefixTests(unittest.TestCase):
    device = 'cpu'

    def target(self):
        return PanopticBatch(
            torch.tensor([[[1, 1, 2, 2, 3, 0]]], dtype=torch.int32, device=self.device),
            torch.tensor([[-1, 0, 0, 1]], dtype=torch.int32, device=self.device))

    def test_every_prefix_matches_canvas_evaluation(self):
        target = self.target()
        gen = torch.Generator().manual_seed(17)
        for _ in range(5):
            masks = (torch.rand((7, 1, 6), generator=gen) > .4).to(self.device)
            masks[:, :, 0] = True
            classes = torch.randint(0, 2, (7,), generator=gen).to(self.device).int()
            prediction = InstanceBatch([masks], [classes])
            actual = segmentation_prefixes(prediction, target, 3, isthing=[True, False, True],
                                           checkpoints=range(8))[0]
            canvas = torch.zeros_like(target.segment_map)
            for k, result in enumerate(actual):
                if k:
                    canvas[0][masks[k-1]] = k
                labels = torch.cat((classes.new_tensor([-1]), classes))
                present = torch.bincount(canvas.flatten().long(), minlength=8) > 0
                labels = torch.where(present, labels, -1)
                evaluator = PanopticEvaluator(3, isthing=[True, False, True], device=self.device)
                evaluator.update(PanopticBatch(canvas, labels[None]), target)
                expected = evaluator.compute()
                for key in ('tp', 'fp', 'fn', 'iou_sum', 'confusion', 'miou'):
                    torch.testing.assert_close(result[key], expected[key], equal_nan=True)
                for key in ('All', 'Things', 'Stuff', 'per_class'):
                    for metric in expected[key]:
                        torch.testing.assert_close(result[key][metric], expected[key][metric], equal_nan=True)

    def test_independent_masks_duplicates_and_crowd(self):
        target = self.target()
        # Full occlusion of the first detection does not remove its instance TP.
        masks = torch.tensor([[[1, 1, 0, 0, 0, 0]],
                              [[1, 1, 0, 0, 0, 0]],
                              [[0, 0, 1, 1, 0, 0]],
                              [[0, 0, 0, 0, 1, 0]]], dtype=torch.bool, device=self.device)
        classes = torch.tensor([0, 0, 0, 1], dtype=torch.int32, device=self.device)
        result = segmentation_prefixes(InstanceBatch([masks], [classes]), target, 2,
                                       isthing=[True, False])[0]
        self.assertEqual([r['instance']['tp'][0, 0].item() for r in result], [1, 1, 2, 2])
        self.assertEqual([r['instance']['fp'][0, 0].item() for r in result], [0, 1, 1, 1])
        self.assertAlmostEqual(result[-1]['instance']['precision'][0, 0].item(), 2/3)
        crowd = torch.tensor([[False, True, False, False]], device=self.device)
        target = PanopticBatch(target.segment_map, target.classes, crowd)
        result = segmentation_prefixes(InstanceBatch([masks], [classes]), target, 2)[0]
        self.assertEqual(result[1]['instance']['tp'].sum().item(), 0)
        self.assertEqual(result[1]['instance']['fp'].sum().item(), 0)
        self.assertEqual(result[2]['instance']['recall'][0, 0].item(), 1)

    def test_variable_lengths_original_ids_and_thresholds(self):
        target = self.target()
        ids = torch.tensor([0, 2**40, 71, 9], device=self.device)
        target = PanopticBatch.from_coco(
            ids[target.segment_map.long()].repeat(2, 1, 1),
            [[dict(id=2**40, category_id=0), dict(id=71, category_id=0),
              dict(id=9, category_id=1)]] * 2, [0, 1], compact=False)
        mask = torch.tensor([[[1, 0, 0, 0, 0, 0]]], dtype=torch.bool, device=self.device)
        empty = mask[:0]
        classes = torch.zeros(1, dtype=torch.int32, device=self.device)
        prediction = InstanceBatch([mask, empty], [classes, classes[:0]])
        result = segmentation_prefixes(prediction, target, 2)
        self.assertEqual([r[0]['prefix'] for r in result], [1, 0])
        self.assertEqual(result[0][0]['instance']['tp'][:, 0].tolist(), [1, 0])
        self.assertEqual(result[0][0]['instance']['fp'][:, 0].tolist(), [0, 1])
        selected = segmentation_prefixes(prediction, target, 2, checkpoints=[0])
        self.assertEqual([r[0]['prefix'] for r in selected], [0, 0])
        self.assertEqual(segmentation_prefixes(prediction, target, 2, checkpoints=[]), [[], []])

    def test_empty_checkpoints_and_validation(self):
        target = self.target()
        prediction = InstanceBatch([torch.empty((0, 1, 6), dtype=torch.bool, device=self.device)],
                                   [torch.empty(0, dtype=torch.int32, device=self.device)])
        result = segmentation_prefixes(prediction, target, 2)[0]
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['prefix'], 0)
        self.assertEqual(result[0]['instance']['fn'][0].sum().item(), 3)
        for kwargs in ({'checkpoints': [1]}, {'checkpoints': [0, 0]},
                       {'iou_thresholds': [0]}, {'iou_thresholds': [float('nan')]}):
            with self.assertRaises(ValueError):
                segmentation_prefixes(prediction, target, 2, **kwargs)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
class CUDAPrefixTests(PrefixTests):
    device = 'cuda'

    def test_native_cpu_parity_streams_and_snapshots(self):
        gen = torch.Generator().manual_seed(123)
        for g, n in ((3, 0), (9, 17), (65, 129), (257, 300)):
            maps = torch.randint(g, (1, 19, 23), generator=gen).int()
            maps.flatten()[:g] = torch.arange(g).int()
            labels = (torch.arange(g) % 3).int()[None]
            labels[:, 0] = -1
            crowd = torch.rand((1, g), generator=gen) < .3
            crowd[:, 0] = False
            slots = torch.randint(1, g, (n,), generator=gen)
            masks = maps[0][None] == slots[:, None, None]
            if n:
                masks |= torch.rand(masks.shape, generator=gen) < .01
            classes = labels[0, slots]
            pred = InstanceBatch([masks], [classes])
            target = PanopticBatch(maps, labels, crowd)
            kwargs = dict(isthing=[True, False, True],
                          iou_thresholds=(.5,) * 300 if n == 17 else (.5, .75, .95),
                          checkpoints=[0, n] if n else [0])
            expected = segmentation_prefixes(pred, target, 3, **kwargs)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                actual = segmentation_prefixes(
                    InstanceBatch([masks.cuda()], [classes.cuda()]),
                    PanopticBatch(maps.cuda(), labels.cuda(), crowd.cuda()), 3, **kwargs)
            stream.synchronize()
            def compare(a, e):
                if isinstance(e, dict):
                    self.assertEqual(a.keys(), e.keys())
                    for key in e:
                        compare(a[key], e[key])
                elif isinstance(e, list):
                    self.assertEqual(len(a), len(e))
                    for av, ev in zip(a, e):
                        compare(av, ev)
                elif isinstance(e, torch.Tensor):
                    torch.testing.assert_close(a.cpu(), e, rtol=1e-12, atol=1e-12, equal_nan=True)
                else:
                    self.assertEqual(a, e)
            compare(actual, expected)
            # A later native call must not mutate retained output storage.
            segmentation_prefixes(InstanceBatch([masks.cuda()], [classes.cuda()]),
                                  PanopticBatch(maps.cuda(), labels.cuda(), crowd.cuda()), 3, **kwargs)
            compare(actual, expected)

    def test_native_rejects_empty_masks(self):
        prediction = InstanceBatch([torch.zeros((1, 1, 6), dtype=torch.bool, device='cuda')],
                                   [torch.zeros(1, dtype=torch.int32, device='cuda')])
        with self.assertRaises(ValueError):
            segmentation_prefixes(prediction, self.target(), 2)
