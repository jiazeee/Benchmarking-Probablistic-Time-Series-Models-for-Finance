"""
data_provider/dataset.py
 
Handles all data loading, normalization, and windowing for probabilistic time series forecasting use.
Ensures every model receives the exact same format of data.
Works identically for simulated data (from simulators/) and real DJIA data.

Usage:
    from data_provider.dataset import get_dataloaders
    
    # with simulated data
    from simulators.garch import GARCHSimulator
    returns = GARCHSimulator(T=2000, n_firms=30, seed=42).simulate()["returns"]
    train_loader, val_loader, test_loader, norm = get_dataloaders(returns)
 
    # with real data
    returns = load_djia("data/historical_stock_data/dj30_returns_20160101_to_20260101_wide.csv")
    train_loader, val_loader, test_loader, norm = get_dataloaders(returns)
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class Normalizer:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, train_returns: np.ndarray):
        """train_returns: [T_train, N]"""
        self.mean = train_returns.mean(axis=0)
        self.std = train_returns.std(axis=0)
        self.std = np.where(self.std < 1e-8, 1.0, self.std)

    def transform(self, returns: np.ndarray) -> np.ndarray:
        """returns: [T, N]"""
        if self.mean is None:
            raise RuntimeError("Normalizer has not been fit yet. Call fit() first.")
        return (returns - self.mean) / self.std
    
    def inverse_transform(self, returns: np.ndarray) -> np.ndarray:
        """Undo normalization — call this before computing metrics on predictions."""
        if self.mean is None:
            raise RuntimeError("Normalizer has not been fit yet. Call fit() first.")
        return returns * self.std + self.mean
        
class ProbTSDataset(Dataset):
    """
    Sliding window dataset for probabilistic time series forecasting.

    Inherits from Torch Dataset
 
    Given a returns matrix of shape [T, N], produces windows of:
        x (past):   [context_len, N]   — what the model sees
        y (future): [pred_len, N]      — what the model predicts
 
    The window slides one step at a time across the time axis:
        window 0: x=returns[0:63],   y=returns[63:84]
        window 1: x=returns[1:64],   y=returns[64:85]
        ...
 
    Args:
        returns:     np.ndarray of shape [T, N], already normalized
        context_len: number of past timesteps the model sees (default: 63 ~ 3 months)
        pred_len:    number of future timesteps to predict  (default: 21 ~ 1 month)
    """
    def __init__(self, returns: np.ndarray, context_len: int = 63, pred_len: int = 21):
        self.returns = torch.tensor(returns, dtype=torch.float32)
        self.context_len = context_len
        self.pred_len = pred_len

        # error if data is shorter than one window
        if len(self.returns) < context_len + pred_len:
            raise ValueError(
                f"Data length ({len(returns)} is too short for context_len={context_len} "
                f"+ pred_len={pred_len}. Need at least {context_len + pred_len} timesteps."
            )
        
    def __len__(self) -> int:
        return len(self.returns) - self.context_len - self.pred_len + 1

    def __getitem__(self, idx: int):
        x = self.returns[idx : idx + self.context_len]
        y = self.returns[idx + self.context_len : idx + self.context_len + self.pred_len]
        return x, y
    
def load_djia(csv_path: str) -> np.ndarray:
    """
    Load the DJIA returns CSV and return a clean [T, N] numpy array.
 
    Drops DOW which has NaNs before 2019 (only joined DJIA then).
    Returns log-returns as float32.
 
    Args:
        csv_path: path to dj30_returns_20160101_to_20260101_wide.csv
    Returns:
        returns: np.ndarray of shape [T, 29]  (30 stocks minus DOW)
    """
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)

    if "DOW" in df.columns:
        df = df.drop(columns=["DOW"])
    
    # ensure no other NaNs sneak in
    if df.isnull().any().any():
        null_cols = df.collumns[df.isnull().any()].tolist()
        raise ValueError(f"Unexpected NaN values in columns: {null_cols}. "
                         f"Inspect the data before proceeding.")
    
    return df.values.astype(np.float32)

def get_dataloaders(
        returns: np.ndarray,
        context_len: int = 63,
        pred_len: int = 21,
        batch_size: int = 32,
        train_frac: float = 0.6,
        val_frac: float = 0.2
    ):
    """
    Split data into train/val/test, normalize, and return DataLoaders.
 
    Splitting is always done by time (never randomly) to avoid leakage.
    Normalization is fit on training data only and applied to all splits.
 
    Args:
        returns:     [T, N] array of returns (simulated or real)
        context_len: past window length fed to the model
        pred_len:    future window length to predict
        batch_size:  DataLoader batch size
        train_frac:  fraction of data for training   (default 0.6)
        val_frac:    fraction of data for validation (default 0.2)
                     test gets the remaining 0.2
 
    Returns:
        train_loader: DataLoader
        val_loader:   DataLoader
        test_loader:  DataLoader
        norm:         fitted Normalizer (keep this — needed to inverse_transform predictions)
 
    Example:
        train_loader, val_loader, test_loader, norm = get_dataloaders(returns)
        for x, y in train_loader:
            # x: [batch, context_len, N]
            # y: [batch, pred_len, N]
            ...
    """
    if train_frac + val_frac >= 1.0:
        raise ValueError("train_frac + val_frac must be less than 1.0 (need some test data).")

    T = len(returns)
    train_end = int(train_frac * T)
    val_end = int((train_frac + val_frac) * T)

    train_raw = returns[:train_end]
    val_raw = returns[train_end:val_end]
    test_raw = returns[val_end:]

    # fit normalizer only on training data
    norm = Normalizer()
    norm.fit(train_raw)

    # normalize all splits with training statistics
    train_norm = norm.transform(train_raw)
    val_norm = norm.transform(val_raw)
    test_norm = norm.transform(test_raw)

    train_dataset = ProbTSDataset(train_norm, context_len, pred_len)
    val_dataset = ProbTSDataset(val_norm, context_len, pred_len)
    test_dataset = ProbTSDataset(test_norm, context_len, pred_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Data split:  train={len(train_raw)} | val={len(val_raw)} | test={len(test_raw)} timesteps")
    print(f"Windows:     train={len(train_dataset)} | val={len(val_dataset)} | test={len(test_dataset)}")
    print(f"Batch shape: x=[{batch_size}, {context_len}, {returns.shape[1]}]  "
          f"y=[{batch_size}, {pred_len}, {returns.shape[1]}]")

    return train_loader, val_loader, test_loader, norm
