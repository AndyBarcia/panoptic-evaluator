"""Resident CUDA per-image reward scoring versus single-image evaluation."""
import argparse
import time
import torch
from panoptic_evaluator import PanopticBatch, PanopticEvaluator, panoptic_quality


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iterations', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--validate', action='store_true')
    args = parser.parse_args()
    if args.iterations <= 0 or args.batch_size <= 0:
        parser.error('iterations and batch size must be positive')
    B, H, W, S, C = args.batch_size, 512, 768, 64, 15
    x = torch.arange(W, device='cuda')[None]
    y = torch.arange(H, device='cuda')[:, None]
    maps = (1 + (x // 32 + y // 32 * 24) % S).int()[None].expand(B, -1, -1).contiguous()
    labels = (torch.arange(S + 1, device='cuda') % C).int()[None].expand(B, -1).clone()
    labels[:, 0] = -1
    target = PanopticBatch(maps, labels)
    pred = maps.clone()
    pred[:, ::11] = 0
    prediction = target.with_maps(pred)
    singles = [(PanopticBatch(pred[b:b+1], labels[b:b+1]),
                PanopticBatch(maps[b:b+1], labels[b:b+1]),
                PanopticEvaluator(C, validate=args.validate)) for b in range(B)]

    def batched():
        return panoptic_quality(prediction, target, C, validate=args.validate)

    def repeated():
        results = []
        for p, g, evaluator in singles:
            evaluator.reset()
            evaluator.update(p, g)
            results.append(evaluator.compute()['All']['pq'])
        return torch.stack(results)

    torch.testing.assert_close(batched(), repeated())
    timings = []
    for name, operation in [('batched reward', batched), ('repeated single-image evaluator', repeated)]:
        for _ in range(10):
            operation()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        wall = time.perf_counter()
        start.record()
        for _ in range(args.iterations):
            result = operation()
        end.record()
        end.synchronize()
        elapsed = (time.perf_counter() - wall) * 1000 / args.iterations
        timings.append(elapsed)
        print(f'{name}: CUDA {start.elapsed_time(end)/args.iterations:.3f} ms/batch; '
              f'wall {elapsed:.3f} ms/batch; {B*1000/elapsed:.0f} images/s; '
              f'peak {torch.cuda.max_memory_allocated()/2**20:.1f} MiB; PQ={result.mean().item():.6f}')
    print(f'Wall speedup: {timings[1]/timings[0]:.2f}x; validation={args.validate}; '
          f'{B} x {H} x {W}, {S} segments, {C} classes')


if __name__ == '__main__':
    main()
