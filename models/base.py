"""
models/base.py

Abstract base class that every model in this library must inherit from.
Defines the interface: every model must implement train_model() and sample().
"""

from abc import ABC, abstractmethod
import numpy as np
import torch


class ProbForecastModel(ABC):
    """
    Every probabilistic forecasting model in this library inherits from this.

    Methods:
        - train_model(train_loader, val_loader)
        - sample(past, num_samples, pred_len)

    Everything else in the library (exp/run.py, evaluation/) assumes models
    follow this interface.
    """

    @abstractmethod
    def train_model(self, train_loader, val_loader):
        """
        Train the model.

        Args:
            train_loader: DataLoader yielding (x, y) batches from training set
            val_loader:   DataLoader yielding (x, y) batches from validation set
        """
        raise NotImplementedError

    @abstractmethod
    def sample(self, past: torch.Tensor, num_samples: int = 100) -> np.ndarray:
        """
        Generate probabilistic predictions as samples.

        Args:
            past:        [batch, context_len, N] — the past window(s)
            num_samples: how many future paths to sample

        Returns:
            samples: np.ndarray of shape [num_samples, batch, pred_len, N]
                     — num_samples plausible futures for each item in the batch
        """
        raise NotImplementedError
    