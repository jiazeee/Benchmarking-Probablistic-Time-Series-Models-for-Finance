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


def load_model(args, pred_len: int, train_loader=None):
    if args.model == "naive":
        from models.naive import NaiveGaussian
        return NaiveGaussian(pred_len=pred_len)

    elif args.model == "deepvar":
        from models.deep_var import DeepVAR
        x_sample, _ = next(iter(train_loader))
        N = x_sample.shape[2]
        return DeepVAR(
            input_size=N,
            pred_len=pred_len,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            rank=args.rank,
            embed_dim=args.embed_dim,
            lr=args.lr,
            n_epochs=args.n_epochs,
            patience=args.patience,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

    elif args.model == "timegrad":
        from models.timegrad import TimeGrad
        x_sample, _ = next(iter(train_loader))
        N = x_sample.shape[2]
        return TimeGrad(
            input_size=N,
            pred_len=pred_len,
            hidden_size=args.hidden_size,
            conditioning_length=args.conditioning_length,
            diff_steps=args.diff_steps,
            beta_end=args.beta_end,
            beta_schedule=args.beta_schedule,
            residual_layers=args.residual_layers,
            residual_channels=args.residual_channels,
            dilation_cycle_length=args.dilation_cycle_length,
            lr=args.lr,
            n_epochs=args.n_epochs,
            patience=args.patience,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

    else:
        raise ValueError(f"Unknown model: {args.model}. "
                         f"Available: naive, deepvar, timegrad")


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
        samples = model.sample(x, num_samples=num_samples)
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
                        help="Model to use: naive, deepvar, timegrad")
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

    # Shared hyperparameters
    parser.add_argument("--hidden_size", type=int,   default=64)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--n_epochs",    type=int,   default=50)
    parser.add_argument("--patience",    type=int,   default=5)

    # DeepVAR-specific
    parser.add_argument("--num_layers",  type=int,   default=2)
    parser.add_argument("--rank",        type=int,   default=5)
    parser.add_argument("--embed_dim",   type=int,   default=4)

    # TimeGrad-specific
    parser.add_argument("--conditioning_length",  type=int,   default=64)
    parser.add_argument("--diff_steps",           type=int,   default=100)
    parser.add_argument("--beta_end",             type=float, default=0.1)
    parser.add_argument("--beta_schedule",        type=str,   default="linear")
    parser.add_argument("--residual_layers",      type=int,   default=8)
    parser.add_argument("--residual_channels",    type=int,   default=8)
    parser.add_argument("--dilation_cycle_length",type=int,   default=2)

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
    model = load_model(args, pred_len=args.pred_len, train_loader=train_loader)
    print(f"\nTraining {args.model}...")
    model.train_model(train_loader, val_loader)

    # 4. run test loop
    print(f"\nSampling from {args.model} on test set...")
    samples, targets = run_test_loop(model, test_loader, args.num_samples, norm)

    # 5. inverse-transform to original scale before saving and evaluating
    # samples: [M, T, N], targets: [T, N] — both normalised; undo here
    samples = norm.inverse_transform(samples)
    targets = norm.inverse_transform(targets)

    # 6. save outputs
    save_outputs(run_id, samples, targets, args)

    # 7. evaluate and print results
    print(f"\nResults for {run_id}:")
    results = evaluate_all(samples, targets)
    print(f"\n{'Metric':<20} {'Value':>10}")
    print("-" * 32)
    for name, value in results.items():
        print(f"  {name:<18} {value:>10.4f}")


if __name__ == "__main__":
    main()
