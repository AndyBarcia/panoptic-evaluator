"""Run isolated, alternating baseline comparisons for experimental CUDA kernels."""
import argparse
import os
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=7)
    args = parser.parse_args()
    cases = [
        ("small/coherent", 16, [], [("shared", "single")]),
        ("medium/coherent", 64, [], [("shared", "single"), ("global", "multi"), ("shared", "multi")]),
        ("medium/dense", 64, ["--fragmented", "--independent-prediction"], [("shared", "single")]),
        ("medium/original/validated", 64, ["--original-ids", "--validate"], [("shared", "single")]),
        ("large/coherent", 512, [], [("global", "multi")]),
        ("large/dense", 512, ["--fragmented", "--independent-prediction"], [("global", "multi")]),
        ("large/single-image", 1024, ["--batch-size", "1"], [("global", "multi")]),
        ("large/validated", 512, ["--validate"], [("global", "multi")]),
    ]
    for name, segments, flags, modes in cases:
        for histogram, scoring in modes:
            print(f"\nCASE {name}: histogram={histogram}, scoring={scoring}", flush=True)
            env = dict(os.environ, PANOPTIC_HISTOGRAM=histogram, PANOPTIC_SCORING=scoring)
            subprocess.run([sys.executable, "-m", "examples.compare_versions",
                            "--baseline-dir", args.baseline_dir,
                            "--segments", str(segments), "--iterations", str(args.iterations),
                            "--rounds", str(args.rounds), *flags], env=env, check=True)


if __name__ == "__main__":
    main()
