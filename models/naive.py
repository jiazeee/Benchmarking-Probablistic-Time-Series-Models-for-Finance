"""
models/naive.py

Gaussian baseline model. For each series, fits a Gaussian to the context
window and samples from it. No learning, no parameters.

This exists purely to test that the pipeline works end to end. 
It should produce reasonable but not great CRPS.
"""

import numpy as np
import torch
from .base import ProbForecastModel


class NaiveGaussian(ProbForecastModel):
    """
    For each series, estimates mean and std from the context window,
    then samples pred_len steps from that Gaussian independently.

    No training needed — fit() does nothing.
    """

    def __init__(self, pred_len: int = 21):
        self.pred_len = pred_len

    def train_model(self, train_loader, val_loader):
        # nothing to train — this model has no parameters
        print("NaiveGaussian: no training needed.")

    def sample(self, past: torch.Tensor, num_samples: int = 100) -> np.ndarray:
        """
        Args:
            past: [batch, context_len, N]

        Returns:
            samples: [num_samples, batch, pred_len, N]
        """
        past_np = past.numpy()                        # [batch, context_len, N]
        batch, context_len, N = past_np.shape

        mu  = past_np.mean(axis=1)                    # [batch, N]
        std = past_np.std(axis=1)                     # [batch, N]
        std = np.where(std < 1e-8, 1e-8, std)        # floor to avoid zero std

        # sample: draw (num_samples, batch, pred_len, N) from N(mu, std)
        # mu and std are [batch, N] so we expand dims to broadcast correctly
        mu_exp  = mu[np.newaxis, :, np.newaxis, :]    # [1, batch, 1, N]
        std_exp = std[np.newaxis, :, np.newaxis, :]   # [1, batch, 1, N]

        noise   = np.random.randn(num_samples, batch, self.pred_len, N)
        samples = mu_exp + std_exp * noise             # [num_samples, batch, pred_len, N]

        return samples.astype(np.float32)
    