"""
evaluation/evaluate.py

Loads saved predictions from outputs/ and computes metrics.
Completely decoupled from training — run this any time after exp/run.py.

Usage:
    python evaluation/evaluate.py --run_dir outputs/naive_garch_seed42
"""

import argparse
import numpy as np
from metrics import evaluate_all


def evaluate_run(run_dir: str):
    """
    Load samples.npy and targets.npy from a run directory and print metrics.
    """
    samples = np.load(f"{run_dir}/samples.npy")   # [num_samples, T, N]
    targets = np.load(f"{run_dir}/targets.npy")   # [T, N]

    print(f"\nEvaluating: {run_dir}")
    print(f"  samples shape: {samples.shape}")
    print(f"  targets shape: {targets.shape}")

    results = evaluate_all(samples, targets)

    print(f"\n{'Metric':<20} {'Value':>10}")
    print("-" * 32)
    for name, value in results.items():
        print(f"  {name:<18} {value:>10.4f}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True,
                        help="Path to output directory containing samples.npy and targets.npy")
    args = parser.parse_args()
    evaluate_run(args.run_dir)
    