"""python -m unittest discover -s tests; optional panopticapi enables direct parity."""
import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from panoptic_evaluator import PanopticBatch, PanopticEvaluator

try:
    from panopticapi.evaluation import pq_compute_single_core
except ImportError:
    pq_compute_single_core = None


def pack(values, info, device="cpu", categories=(7, 42, 91)):
    return PanopticBatch.from_coco(torch.tensor(values, dtype=torch.int64, device=device), info, categories)


def annotations(maps, metadata):
    result = []
    for b, info in enumerate(metadata):
        result.append({"image_id": b, "file_name": f"{b}.png", "segments_info": [
            dict(entry, area=int((maps[b] == entry["id"]).sum()), iscrowd=entry.get("iscrowd", 0))
            for entry in info]})
    return result


def official(gt, pred, gi, pi):
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        for name, maps in (("gt", gt), ("pred", pred)):
            (root / name).mkdir()
            for b, image in enumerate(maps):
                rgb = np.stack((image % 256, image // 256 % 256, image // 65536 % 256), -1).astype(np.uint8)
                Image.fromarray(rgb).save(root / name / f"{b}.png")
        with contextlib.redirect_stdout(io.StringIO()):
            return pq_compute_single_core(0, list(zip(annotations(gt, gi), annotations(pred, pi))),
                                          str(root / "gt"), str(root / "pred"),
                                          {i: {"isthing": i != 42} for i in (7, 42, 91)})


class EvaluatorTests(unittest.TestCase):
    device = "cpu"

    def evaluate(self, gt, pred, gi, pi):
        evaluator = PanopticEvaluator(3, device=self.device, isthing=[True, False, True])
        evaluator.update(pack(pred, pi, self.device), pack(gt, gi, self.device))
        return evaluator

    def test_perfect_and_reset(self):
        maps = [[[0, 1, 1, 2]]]
        info = [[{"id": 1, "category_id": 7}, {"id": 2, "category_id": 42}]]
        evaluator = self.evaluate(maps, maps, info, info)
        result = evaluator.compute()
        self.assertEqual(result["All"]["pq"].item(), 1)
        self.assertEqual(result["Things"]["n"].item(), 1)
        self.assertEqual(result["Stuff"]["pq"].item(), 1)
        self.assertTrue(torch.isnan(result["per_class"]["iou"][2]))
        self.assertEqual(result["miou"].item(), 1)
        evaluator.reset()
        self.assertEqual(evaluator.compute()["All"]["pq"].item(), 0)
        self.assertEqual(evaluator.compute()["miou"].item(), 0)

    def test_strict_half_and_void_union(self):
        gi = [[{"id": 1, "category_id": 7}]]
        pi = [[{"id": 9, "category_id": 7}]]
        result = self.evaluate([[[1, 1, 1, 1]]], [[[9, 9, 0, 0]]], gi, pi).compute()
        self.assertEqual(result["tp"].sum().item(), 0)
        self.assertEqual(result["fp"].sum().item(), 1)
        self.assertEqual(result["fn"].sum().item(), 1)
        self.assertEqual(result["miou"].item(), .5)
        result = self.evaluate([[[1, 1, 1]]], [[[9, 9, 0]]], gi, pi).compute()
        self.assertEqual(result["tp"].sum().item(), 1)
        self.assertAlmostEqual(result["iou_sum"].sum().item(), 2 / 3)
        result = self.evaluate([[[1, 0, 0, 0]]], [[[9, 9, 9, 9]]], gi, pi).compute()
        self.assertEqual(result["All"]["pq"].item(), 1)

    def test_crowd_and_half_ignore(self):
        gi = [[{"id": 1, "category_id": 7, "iscrowd": 1}]]
        pi = [[{"id": 9, "category_id": 7}]]
        result = self.evaluate([[[1, 1, 0]]], [[[9, 9, 9]]], gi, pi).compute()
        self.assertEqual(result["fp"].sum().item(), 0)
        self.assertEqual(result["fn"].sum().item(), 0)
        result = self.evaluate([[[1, 1, 1, 1]]], [[[9, 9, 9, 9]]], gi,
                               [[{"id": 9, "category_id": 42}]]).compute()
        self.assertEqual(result["fp"][1].item(), 1)
        result = self.evaluate([[[0, 0, 1, 1]]], [[[9, 9, 9, 9]]],
                               [[{"id": 1, "category_id": 42}]], pi).compute()
        self.assertEqual(result["fp"][0].item(), 1)

    def test_semantic_confusion_and_void_prediction(self):
        result = self.evaluate([[[1, 1, 2, 2]]], [[[9, 0, 9, 10]]],
            [[{"id": 1, "category_id": 7}, {"id": 2, "category_id": 42}]],
            [[{"id": 9, "category_id": 7}, {"id": 10, "category_id": 42}]]).compute()
        torch.testing.assert_close(result["confusion"].cpu(), torch.tensor([[1, 0, 0, 1], [1, 1, 0, 0], [0, 0, 0, 0]]))
        self.assertAlmostEqual(result["miou"].item(), (1 / 3 + 1 / 2) / 2)

    def test_arbitrary_ids_and_batch_accumulation(self):
        maps = [[[0, 16777215, 16777215]], [[16777215, 0, 0]]]
        info = [[{"id": 16777215, "category_id": 7}], [{"id": 16777215, "category_id": 42}]]
        batched = self.evaluate(maps, maps, info, info).compute()
        evaluator = PanopticEvaluator(3, device=self.device)
        for b in range(2):
            item = pack(maps[b:b + 1], info[b:b + 1], self.device)
            evaluator.update(item, item)
        for key in ("tp", "fp", "fn", "iou_sum", "confusion", "miou"):
            torch.testing.assert_close(batched[key], evaluator.compute()[key])

    def test_validation(self):
        with self.assertRaises(ValueError):
            pack([[[1]]], [[]], self.device)
        with self.assertRaises(ValueError):
            pack([[[1]]], [[{"id": 1, "category_id": 2}]], self.device)
        batch = PanopticBatch(torch.ones((1, 1, 1), dtype=torch.int32, device=self.device),
                              torch.tensor([[-1, -1]], dtype=torch.int32, device=self.device))
        with self.assertRaises(ValueError):
            PanopticEvaluator(3, device=self.device).update(batch, batch)

    def test_original_ids_and_metadata_reuse(self):
        large = 2**40 + 17
        gt = torch.tensor([[[0, large, large, 1234], [7, 7, large, 1234]],
                           [[99, 99, 0, 0], [0, 0, 0, 0]]], dtype=torch.int64, device=self.device)
        pred = torch.tensor([[[123, 123, 123, 123], [900, 900, 123, 123]],
                             [[321, 321, 0, 0], [0, 0, 0, 0]]], dtype=torch.int32, device=self.device)
        gi = [[{"id": 1234, "category_id": 7, "iscrowd": 1},
               {"id": large, "category_id": 7, "iscrowd": 1},
               {"id": 7, "category_id": 42}], [{"id": 99, "category_id": 91}]]
        pi = [[{"id": 900, "category_id": 42}, {"id": 123, "category_id": 7}],
              [{"id": 321, "category_id": 91}]]
        target = PanopticBatch.from_coco(gt, gi, (7, 42, 91), compact=False)
        prediction = PanopticBatch.from_coco(pred, pi, (7, 42, 91), compact=False)
        self.assertEqual(target.segment_map.data_ptr(), gt.data_ptr())
        reference = PanopticEvaluator(3, device="cpu")
        reference.update(PanopticBatch.from_coco(pred.cpu(), pi, (7, 42, 91)),
                         PanopticBatch.from_coco(gt.cpu(), gi, (7, 42, 91)))
        for valid in (True, False):
            for compact_target, compact_prediction in ((False, False), (True, False), (False, True)):
                evaluator = PanopticEvaluator(3, device=self.device, validate=valid)
                evaluator.update(prediction.compact() if compact_prediction else prediction,
                                 target.compact() if compact_target else target)
                for key in ("tp", "fp", "fn", "iou_sum", "confusion", "miou"):
                    torch.testing.assert_close(evaluator.compute()[key].cpu(), reference.compute()[key])
        new_maps = gt.flip(2)
        reused = target.with_maps(new_maps)
        for name in ("classes", "crowd", "lookup_ids", "lookup_slots"):
            self.assertEqual(getattr(reused, name).data_ptr(), getattr(target, name).data_ptr())
        evaluator.reset()
        evaluator.update(prediction, reused)
        reference.reset()
        reference.update(PanopticBatch.from_coco(pred.cpu(), pi, (7, 42, 91)),
                         PanopticBatch.from_coco(new_maps.cpu(), gi, (7, 42, 91)))
        torch.testing.assert_close(evaluator.compute()["confusion"].cpu(), reference.compute()["confusion"])
        # Exercise mixed map dtypes in the opposite direction, including the
        # rule that crowd flags are only meaningful for the target.
        evaluator.reset()
        evaluator.update(target, prediction)
        reference.reset()
        reference.update(PanopticBatch.from_coco(gt.cpu(), gi, (7, 42, 91)),
                         PanopticBatch.from_coco(pred.cpu(), pi, (7, 42, 91)))
        for key in ("tp", "fp", "fn", "iou_sum", "confusion", "miou"):
            torch.testing.assert_close(evaluator.compute()[key].cpu(), reference.compute()[key])

    def test_original_ids_across_image_boundaries(self):
        # A warp spans images sharing IDs but using different metadata slot
        # orders. The final warp is partial, and IDs vary within each warp.
        maps = torch.tensor([[[11, 22, 11, 0, 22, 11, 22, 11, 22,
                               11, 22, 11, 22, 11, 22, 11, 22]]] * 3,
                            dtype=torch.int64)
        info = [[{"id": 11, "category_id": 7}, {"id": 22, "category_id": 42}],
                [{"id": 22, "category_id": 7}, {"id": 11, "category_id": 42}],
                [{"id": 11, "category_id": 42}, {"id": 22, "category_id": 7}]]
        original = PanopticBatch.from_coco(maps.to(self.device), info, (7, 42), compact=False)
        reference_batch = PanopticBatch.from_coco(maps, info, (7, 42))
        torch.testing.assert_close(original.compact().segment_map.cpu(), reference_batch.segment_map)
        reference = PanopticEvaluator(2, device="cpu")
        reference.update(reference_batch, reference_batch)
        for validate in (False, True):
            evaluator = PanopticEvaluator(2, device=self.device, validate=validate)
            evaluator.update(original, original)
            for key in ("tp", "fp", "fn", "iou_sum", "confusion", "miou"):
                torch.testing.assert_close(evaluator.compute()[key].cpu(), reference.compute()[key])

    def test_validation_preserves_previous_totals(self):
        maps = torch.tensor([[[1, 1]]], dtype=torch.int32, device=self.device)
        classes = torch.tensor([[-1, 0]], dtype=torch.int32, device=self.device)
        good = PanopticBatch(maps, classes)
        evaluator = PanopticEvaluator(3, device=self.device)
        evaluator.update(good, good)
        before = evaluator.compute()
        unknown = PanopticBatch.from_coco(torch.tensor([[[7, 8]]], dtype=torch.int64, device=self.device),
                                         [[{"id": 7, "category_id": 7}]], (7, 42, 91), compact=False)
        bad_batches = [PanopticBatch(maps + 3, classes),
                       PanopticBatch(maps, torch.tensor([[-1, 4]], dtype=torch.int32, device=self.device)),
                       PanopticBatch(maps, torch.tensor([[-1, -1]], dtype=torch.int32, device=self.device)),
                       PanopticBatch(maps, torch.tensor([[-1, 0, 1]], dtype=torch.int32, device=self.device)),
                       PanopticBatch(maps, classes, torch.tensor([[True, False]], device=self.device)),
                       unknown]
        for index, bad in enumerate(bad_batches):
            with self.assertRaises(ValueError):
                evaluator.update(good, bad)
            if index != 4:  # prediction crowd flags are deliberately unused
                with self.assertRaises(ValueError):
                    evaluator.update(bad, good)
            for key in ("tp", "fp", "fn", "iou_sum", "confusion"):
                torch.testing.assert_close(evaluator.compute()[key], before[key])

    def test_empty_and_noncontiguous(self):
        for shape in ((0, 3, 4), (2, 0, 4)):
            batch = PanopticBatch(torch.empty(shape, dtype=torch.int32, device=self.device),
                                  torch.full((shape[0], 1), -1, dtype=torch.int32, device=self.device))
            evaluator = PanopticEvaluator(3, device=self.device)
            evaluator.update(batch, batch)
            self.assertEqual(evaluator.compute()["confusion"].sum().item(), 0)
            original = PanopticBatch.from_coco(batch.segment_map.long(), [[] for _ in range(shape[0])],
                                              (7, 42, 91), compact=False)
            evaluator.update(original, original)
            self.assertEqual(evaluator.compute()["confusion"].sum().item(), 0)
        empty = pack([[[0, 0]]], [[]], self.device)
        evaluator = PanopticEvaluator(3, device=self.device)
        evaluator.update(empty, empty)
        self.assertEqual(evaluator.compute()["All"]["n"].item(), 0)
        prediction = pack([[[1, 1]]], [[{"id": 1, "category_id": 7}]], self.device)
        evaluator.update(prediction, empty)  # predictions entirely on void are ignored
        self.assertEqual(evaluator.compute()["fp"].sum().item(), 0)
        evaluator.update(empty, prediction)
        self.assertEqual(evaluator.compute()["fn"][0].item(), 1)
        self.assertEqual(evaluator.compute()["miou"].item(), 0)
        maps = torch.ones((2, 4, 6), device=self.device, dtype=torch.int32)[:, ::2, ::2]
        # Use separate strided class storage whose slot one is an active category.
        labels = torch.tensor([[-1, -1, 0, -1]], device=self.device, dtype=torch.int32)[:, ::2].expand(2, -1)
        batch = PanopticBatch(maps, labels)
        evaluator.reset()
        evaluator.update(batch, batch)
        self.assertEqual(evaluator.compute()["All"]["pq"].item(), 1)

    @unittest.skipIf(pq_compute_single_core is None, "install panopticapi for direct parity")
    def test_official_randomized_parity(self):
        rng = np.random.default_rng(143)
        # Random small maps exercise mismatches, split/merge, padding and void.
        for iteration in range(25):
            B = 3
            gt = rng.integers(0, 5, (B, 8, 9))
            pred = gt.copy() if iteration % 3 == 0 else rng.integers(0, 6, gt.shape)
            pred[rng.random(gt.shape) < .15] = 0
            gi, pi = [], []
            for b in range(B):
                gi.append([{"id": int(i), "category_id": int(rng.choice([7, 42, 91])),
                            "iscrowd": int(rng.random() < .3)} for i in np.unique(gt[b]) if i])
                pi.append([{"id": int(i), "category_id": int(rng.choice([7, 42, 91]))}
                           for i in np.unique(pred[b]) if i])
                if iteration % 3 == 0:
                    for entry in pi[-1]:
                        entry["category_id"] = next(x["category_id"] for x in gi[-1] if x["id"] == entry["id"])
            reference = official(gt, pred, gi, pi)
            result = self.evaluate(gt, pred, gi, pi).compute()
            raw = PanopticEvaluator(3, device=self.device, isthing=[True, False, True])
            raw.update(PanopticBatch.from_coco(torch.tensor(pred, device=self.device), pi, (7, 42, 91), compact=False),
                       PanopticBatch.from_coco(torch.tensor(gt, device=self.device), gi, (7, 42, 91), compact=False))
            for key in ("tp", "fp", "fn", "iou_sum", "confusion", "miou"):
                torch.testing.assert_close(raw.compute()[key], result[key], rtol=1e-12, atol=1e-12)
            for c, category in enumerate((7, 42, 91)):
                stat = reference[category]
                for key in ("tp", "fp", "fn"):
                    self.assertEqual(result[key][c].item(), getattr(stat, key))
                self.assertAlmostEqual(result["iou_sum"][c].item(), stat.iou, places=10)
            for name, flag in (("All", None), ("Things", True), ("Stuff", False)):
                categories = {i: {"isthing": i != 42} for i in (7, 42, 91)}
                if result[name]["n"].item():
                    average, _ = reference.pq_average(categories, flag)
                    for key in ("pq", "sq", "rq", "n"):
                        self.assertAlmostEqual(result[name][key].item(), average[key], places=10)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class CUDAEvaluatorTests(EvaluatorTests):
    device = "cuda"

    def test_cooperative_area_reduction_with_validation(self):
        maps = torch.arange(1, 130, dtype=torch.int32).reshape(1, 3, 43)
        gc = torch.full((1, 137), -1, dtype=torch.int32)
        pc = torch.full((1, 193), -1, dtype=torch.int32)
        gc[:, 1:130] = pc[:, 1:130] = torch.arange(129, dtype=torch.int32) % 3
        reference = PanopticEvaluator(3, device="cpu")
        reference.update(PanopticBatch(maps, pc), PanopticBatch(maps, gc))
        for validate in (False, True):
            evaluator = PanopticEvaluator(3, validate=validate)
            evaluator.update(PanopticBatch(maps.cuda(), pc.cuda()),
                             PanopticBatch(maps.cuda(), gc.cuda()))
            for key in ("tp", "fp", "fn", "iou_sum", "confusion", "miou"):
                torch.testing.assert_close(evaluator.compute()[key].cpu(), reference.compute()[key])

    def test_fused_accumulation_varied_capacities_and_snapshots(self):
        generator = torch.Generator().manual_seed(857)
        cuda = PanopticEvaluator(5, validate=False)
        cpu = PanopticEvaluator(5, device="cpu", validate=False)
        for G, P in ((1, 1), (2, 3), (65, 97), (257, 513), (1025, 769), (2, 3)):
            with self.subTest(G=G, P=P):
                gt = torch.randint(G, (3, 31, 47), generator=generator, dtype=torch.int32)
                pred = torch.randint(P, gt.shape, generator=generator, dtype=torch.int32)
                gc = torch.randint(5, (3, G), generator=generator, dtype=torch.int32)
                pc = torch.randint(5, (3, P), generator=generator, dtype=torch.int32)
                gc[:, 0] = pc[:, 0] = -1
                crowd = torch.rand((3, G), generator=generator) < .25
                crowd[:, 0] = False
                # Include matches and missing padded slots as well as random errors.
                if G == 65:
                    pred = gt.clone()
                    pc[:, :G] = gc
                    pc[:, G:] = -1
                    pred[:, ::5, :] = 0
                cuda.update(PanopticBatch(pred.cuda(), pc.cuda()),
                            PanopticBatch(gt.cuda(), gc.cuda(), crowd.cuda()))
                cpu.update(PanopticBatch(pred, pc), PanopticBatch(gt, gc, crowd))
                actual, expected = cuda.compute(), cpu.compute()
                for key in ("tp", "fp", "fn", "iou_sum", "confusion", "miou"):
                    torch.testing.assert_close(actual[key].cpu(), expected[key], rtol=1e-12, atol=1e-12, equal_nan=True)
                for group in ("All", "Things", "Stuff", "per_class"):
                    for key in expected[group]:
                        torch.testing.assert_close(actual[group][key].cpu(), expected[group][key], rtol=1e-12, atol=1e-12, equal_nan=True)
        snapshot = cuda.compute()
        saved = {key: snapshot[key].clone() for key in ("tp", "fp", "fn", "iou_sum", "confusion")}
        pointers = tuple(t.data_ptr() for t in cuda._scratch)
        state_pointer = cuda.tp.data_ptr()
        cuda.reset()
        self.assertEqual(cuda.tp.data_ptr(), state_pointer)
        cuda.update(PanopticBatch(pred.cuda(), pc.cuda()),
                    PanopticBatch(gt.cuda(), gc.cuda(), crowd.cuda()))
        self.assertEqual(tuple(t.data_ptr() for t in cuda._scratch), pointers)
        for key in saved:
            torch.testing.assert_close(snapshot[key], saved[key])

    def test_graph_capture_reuses_scratch(self):
        maps = torch.tensor([[[1, 1, 2, 2]]], dtype=torch.int32, device="cuda")
        classes = torch.tensor([[-1, 0, 1]], dtype=torch.int32, device="cuda")
        compact = PanopticBatch(maps, classes)
        original = PanopticBatch.from_coco((maps.long() * 100003 + 2**40),
            [[{"id": 100003 + 2**40, "category_id": 7}, {"id": 200006 + 2**40, "category_id": 42}]],
            (7, 42, 91), compact=False)
        for item in (compact, original):
            evaluator = PanopticEvaluator(3, validate=False)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                evaluator.update(item, item)
            torch.cuda.current_stream().wait_stream(stream)
            evaluator.reset()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                evaluator.update(item, item)
            graph.replay()
            graph.replay()
            torch.testing.assert_close(evaluator.compute()["tp"].cpu(), torch.tensor([2, 2, 0]))

    def test_nondefault_stream_and_cpu_histogram(self):
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            gt = torch.randint(0, 17, (4, 128, 129), device="cuda", dtype=torch.int32)
            pred = torch.randint(0, 21, gt.shape, device="cuda", dtype=torch.int32)
            gc = torch.arange(17, device="cuda", dtype=torch.int32)[None].expand(4, -1).clone() % 3
            pc = torch.arange(21, device="cuda", dtype=torch.int32)[None].expand(4, -1).clone() % 3
            gc[:, 0] = pc[:, 0] = -1
            crowd = torch.zeros_like(gc, dtype=torch.bool)
            crowd[:, 3] = True
            evaluator = PanopticEvaluator(3)
            evaluator.update(PanopticBatch(pred, pc), PanopticBatch(gt, gc, crowd))
            result = evaluator.compute()
        stream.synchronize()
        cpu = PanopticEvaluator(3, device="cpu")
        cpu.update(PanopticBatch(pred.cpu(), pc.cpu()), PanopticBatch(gt.cpu(), gc.cpu(), crowd.cpu()))
        for key in ("confusion", "tp", "fp", "fn", "iou_sum"):
            torch.testing.assert_close(result[key].cpu(), cpu.compute()[key], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
