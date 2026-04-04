"""
exp/run.py

Main script
    1. Load data (simulated or real)
    2. Get train/val/test DataLoaders
    3. Train model
    4. Run model on test set, collect samples and targets
    5. Save samples.npy and targets.npy to outputs/<run_id>/
    6. Run evaluation and print results

Usage:
    python exp/run.py --model naive --data garch --T 2000 --n_firms 30 --seed 42
    python exp/run.py --model naive --data djia
"""

import argparse
import os
import sys
import numpy as np
import torch

# make sure imports work when running from repo root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_provider.dataset import get_dataloaders, load_djia
from evaluation.metrics import evaluate_all


def load_data(args) -> np.ndarray:
    """
    Load returns array from either a simulator or the real DJIA CSV.
    Returns [T, N] numpy array.
    """
    if args.data == "djia":
        print("Loading DJIA data...")
        returns = load_djia(
            "data/historical_stock_data/dj30_returns_20160101_to_20260101_wide.csv"
        )
        print(f"  shape: {returns.shape}")
        return returns

    # simulated data
    print(f"Simulating {args.data} data (T={args.T}, n_firms={args.n_firms}, seed={args.seed})...")

    if args.data == "garch":
        from simulators.garch import GARCHSimulator
        sim = GARCHSimulator(T=args.T, n_firms=args.n_firms, seed=args.seed)

    elif args.data == "har":
        from simulators.har import HARSimulator
        sim = HARSimulator(T=args.T, n_firms=args.n_firms, seed=args.seed)

    elif args.data == "heavy_tail":
        from simulators.heavy_tail import HeavyTailSimulator
        sim = HeavyTailSimulator(T=args.T, n_firms=args.n_firms, seed=args.seed)

    elif args.data == "regime":
        from simulators.regime_switching import MarketRegimePanelSimulator
        sim = MarketRegimePanelSimulator(T=args.T, n_firms=args.n_firms, seed=args.seed)

    elif args.data == "hawkes":
        from simulators.hawkes import MarketHawkesPanelSimulator
        sim = MarketHawkesPanelSimulator(T=args.T, n_firms=args.n_firms, seed=args.seed)

    elif args.data == "zip":
        from simulators.zero_inflated import MarketZIPPanelSimulator
        sim = MarketZIPPanelSimulator(T=args.T, n_firms=args.n_firms, seed=args.seed)

    else:
        raise ValueError(f"Unknown data source: {args.data}. "
                         f"Choose from: djia, garch, har, heavy_tail, regime, hawkes, zip")

    result = sim.simulate()
    # regime switching stores returns under "y" not "returns"
    returns = result.get("returns", result.get("y"))
    print(f"  shape: {returns.shape}")
    return returns


def load_model(args, pred_len: int):
    """
    Instantiate the model specified by args.model.
    """
    if args.model == "naive":
        from models.naive import NaiveGaussian
        return NaiveGaussian(pred_len=pred_len)

    # future models go here:
    # elif args.model == "deepvar":
    #     from models.deep_var import DeepVAR
    #     return DeepVAR(pred_len=pred_len, ...)

    else:
        raise ValueError(f"Unknown model: {args.model}. "
                         f"Available: naive")


def run_test_loop(model, test_loader, num_samples: int, norm):
    """
    Loop through the entire test set, collect model samples and true targets.

    Returns:
        all_samples: [num_samples, T_test, N]  — model predictions
        all_targets: [T_test, N]               — true future returns
    """
    all_samples = []   # will be list of [num_samples, batch, pred_len, N]
    all_targets = []   # will be list of [batch, pred_len, N]

    print(f"Running test loop ({len(test_loader)} batches)...")

    for batch_idx, (x, y) in enumerate(test_loader):
        # x: [batch, context_len, N]
        # y: [batch, pred_len, N]

        samples = model.sample(x, num_samples=num_samples)
        # samples: [num_samples, batch, pred_len, N]

        all_samples.append(samples)
        all_targets.append(y.numpy())

        if (batch_idx + 1) % 10 == 0:
            print(f"  batch {batch_idx + 1}/{len(test_loader)}")

    # stack along batch dimension
    # each element of all_samples is [num_samples, batch, pred_len, N]
    # we want [num_samples, T_test_windows, pred_len, N]
    all_samples = np.concatenate(all_samples, axis=1)  # [M, T_windows, pred_len, N]
    all_targets = np.concatenate(all_targets, axis=0)  # [T_windows, pred_len, N]

    # flatten time and pred_len into one time axis for metrics
    # [M, T_windows, pred_len, N] → [M, T_windows * pred_len, N]
    M, T_windows, pred_len, N = all_samples.shape
    all_samples = all_samples.reshape(M, T_windows * pred_len, N)
    all_targets = all_targets.reshape(T_windows * pred_len, N)

    print(f"  done. samples: {all_samples.shape}, targets: {all_targets.shape}")
    return all_samples, all_targets


def save_outputs(run_id: str, samples: np.ndarray, targets: np.ndarray, args):
    """
    Save samples and targets to outputs/<run_id>/
    Also saves the run config so you know what produced these outputs.
    """
    out_dir = os.path.join("outputs", run_id)
    os.makedirs(out_dir, exist_ok=True)

    np.save(os.path.join(out_dir, "samples.npy"), samples)
    np.save(os.path.join(out_dir, "targets.npy"), targets)

    # save config as a simple text file
    config_path = os.path.join(out_dir, "config.txt")
    with open(config_path, "w") as f:
        for key, val in vars(args).items():
            f.write(f"{key}: {val}\n")

    print(f"\nSaved to {out_dir}/")
    print(f"  samples.npy: {samples.shape}")
    print(f"  targets.npy: {targets.shape}")
    print(f"  config.txt")


def main():
    parser = argparse.ArgumentParser(description="Run a probabilistic forecasting experiment")

    parser.add_argument("--model",       type=str,   default="naive",
                        help="Model to use: naive (more coming)")
    parser.add_argument("--data",        type=str,   default="garch",
                        help="Data source: djia, garch, har, heavy_tail, regime, hawkes, zip")
    parser.add_argument("--T",           type=int,   default=2000,
                        help="Length of simulated time series (ignored for djia)")
    parser.add_argument("--n_firms",     type=int,   default=30,
                        help="Number of firms to simulate (ignored for djia)")
    parser.add_argument("--seed",        type=int,   default=42,
                        help="Random seed for simulation and model")
    parser.add_argument("--context_len", type=int,   default=63,
                        help="Number of past timesteps model sees")
    parser.add_argument("--pred_len",    type=int,   default=21,
                        help="Number of future timesteps to predict")
    parser.add_argument("--batch_size",  type=int,   default=32)
    parser.add_argument("--num_samples", type=int,   default=100,
                        help="Number of sample paths to draw from the model")

    args = parser.parse_args()

    # build a human-readable run id so outputs are easy to find
    run_id = f"{args.model}_{args.data}_seed{args.seed}"
    print(f"\n{'='*50}")
    print(f"Run: {run_id}")
    print(f"{'='*50}")

    # 1. load data
    returns = load_data(args)

    # 2. get dataloaders
    train_loader, val_loader, test_loader, norm = get_dataloaders(
        returns,
        context_len=args.context_len,
        pred_len=args.pred_len,
        batch_size=args.batch_size,
    )

    # 3. load and train model
    model = load_model(args, pred_len=args.pred_len)
    print(f"\nTraining {args.model}...")
    model.train_model(train_loader, val_loader)

    # 4. run test loop
    print(f"\nSampling from {args.model} on test set...")
    samples, targets = run_test_loop(model, test_loader, args.num_samples, norm)

    # 5. save outputs
    save_outputs(run_id, samples, targets, args)

    # 6. evaluate and print results
    print(f"\nResults for {run_id}:")
    results = evaluate_all(samples, targets)
    print(f"\n{'Metric':<20} {'Value':>10}")
    print("-" * 32)
    for name, value in results.items():
        print(f"  {name:<18} {value:>10.4f}")


if __name__ == "__main__":
    main()
