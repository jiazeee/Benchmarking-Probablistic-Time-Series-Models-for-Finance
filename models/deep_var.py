"""
DeepVAR: High-Dimensional Multivariate Forecasting with Low-Rank Gaussian
Copula Processes.
Salinas et al., NeurIPS 2019.

Adapted from pytorch-ts (zalandoresearch/pytorch-ts) DeepVARTrainingNetwork /
DeepVARPredictionNetwork (pts/model/deepvar/deepvar_network.py), which is
itself based on GluonTS (awslabs/gluonts).

Simplifications vs. the original pytorch-ts implementation:
  - No time covariates (omitted; financial returns lack strong calendar seasonality)
  - Lag-1 inputs only (original uses multiple lags requiring extended history)
  - Data pre-normalised upstream by Normalizer; MeanScaler omitted
  - No missing-value masking (clean financial data assumed)
  - Parallel sampling via repeat_interleave follows pytorch-ts sampling_decoder

Distribution output: PyTorch built-in LowRankMultivariateNormal (equivalent to
the LowRankMultivariateNormalOutput used in the original via GluonTS).
Dimension embeddings follow pytorch-ts FeatureEmbedder pattern.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import LowRankMultivariateNormal

from .base import ProbForecastModel


class DeepVAR(ProbForecastModel):
    """
    Adapted from pytorch-ts DeepVARTrainingNetwork / DeepVARPredictionNetwork
    (zalandoresearch/pytorch-ts).

    Architecture:
      - LSTM encoder-decoder with learned per-dimension embeddings
        (following pytorch-ts FeatureEmbedder pattern)
      - LowRankMultivariateNormal output: Σ = V Vᵀ + diag(D)
      - NLL training loss with teacher forcing
      - Parallel sampling via repeat_interleave (pytorch-ts sampling_decoder)
    """

    def __init__(
        self,
        input_size,
        pred_len=21,
        hidden_size=64,
        num_layers=2,
        rank=5,
        embed_dim=4,
        lr=1e-3,
        n_epochs=50,
        patience=5,
        device="cpu",
    ):
        self.input_size  = input_size
        self.pred_len    = pred_len
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.rank        = rank
        self.embed_dim   = embed_dim
        self.lr          = lr
        self.n_epochs    = n_epochs
        self.patience    = patience
        self.device      = device
        self.optimizer   = None
        self.train_losses, self.val_losses = [], []

        # Per-dimension embedding — following pytorch-ts FeatureEmbedder pattern.
        # Tells the model which series is which; concatenated to every RNN input.
        self.dim_embed = nn.Embedding(input_size, embed_dim).to(device)

        # LSTM: input is [scaled_value (N) + dim_embeddings (N * embed_dim)]
        lstm_input_size = input_size + input_size * embed_dim
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        ).to(device)

        # Projection head: hidden_size -> mu (N) + cov_factor (N*rank) + cov_diag (N)
        proj_out = input_size * (2 + rank)
        self.head = nn.Linear(hidden_size, proj_out).to(device)

    def _nets(self):
        return [self.dim_embed, self.lstm, self.head]

    def _params(self):
        params = []
        for net in self._nets():
            params += list(net.parameters())
        return params

    def _set_train(self, mode):
        for net in self._nets():
            net.train(mode)

    def _save_state(self):
        return {
            "dim_embed": {k: v.cpu().clone() for k, v in self.dim_embed.state_dict().items()},
            "lstm":      {k: v.cpu().clone() for k, v in self.lstm.state_dict().items()},
            "head":      {k: v.cpu().clone() for k, v in self.head.state_dict().items()},
        }

    def _load_state(self, state):
        self.dim_embed.load_state_dict(state["dim_embed"])
        self.lstm.load_state_dict(state["lstm"])
        self.head.load_state_dict(state["head"])
        for net in self._nets():
            net.to(self.device)

    def _dim_embeddings(self, batch_size, seq_len=1):
        """
        Returns per-dimension static embeddings repeated across time.
        Following pytorch-ts: dim indicator [0..N-1] is embedded and
        concatenated to every RNN input step.

        Returns: [batch_size * seq_len, N * embed_dim]
        """
        indicator = torch.arange(self.input_size, device=self.device)   # [N]
        emb = self.dim_embed(indicator)                                   # [N, embed_dim]
        emb = emb.reshape(1, 1, self.input_size * self.embed_dim)        # [1, 1, N*embed_dim]
        return emb.expand(batch_size, seq_len, -1)                       # [B, seq_len, N*embed_dim]

    def _make_input(self, values):
        """
        Build RNN input: concatenate values with static dim embeddings.
        values: [B, seq_len, N] or [B, N] (single step)
        Returns: [B, seq_len, N + N*embed_dim]
        """
        if values.dim() == 2:
            values = values.unsqueeze(1)    # [B, 1, N]
        B, T, _ = values.shape
        emb = self._dim_embeddings(B, T)    # [B, T, N*embed_dim]
        return torch.cat([values, emb], dim=-1)

    def _encode(self, x):
        # x: [B, context_len, N] -> h, c: [num_layers, B, H]
        inp = self._make_input(x)
        _, (h, c) = self.lstm(inp)
        return h, c

    def _decode_step(self, prev, h, c):
        # prev: [B, N], h/c: [num_layers, B, H]
        inp = self._make_input(prev)               # [B, 1, N + N*embed_dim]
        out, (h, c) = self.lstm(inp, (h, c))
        proj = self.head(out.squeeze(1))           # [B, N*(2+rank)]

        N, R = self.input_size, self.rank
        mu         = proj[:, :N]
        cov_factor = proj[:, N:N + N*R].reshape(-1, N, R)
        cov_diag   = F.softplus(proj[:, N + N*R:]) + 1e-6
        return mu, cov_factor, cov_diag, h, c


    # Training (NLL with teacher forcing, following pytorch-ts forward)


    def _nll_pass(self, loader, train=False):
        self._set_train(train)
        total = 0.0
        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                _, pred_len, _ = y.shape

                if train:
                    self.optimizer.zero_grad()

                h, c = self._encode(x)
                loss = 0.0
                prev = x[:, -1, :]                   # last context value
                for t in range(pred_len):
                    mu, cf, cd, h, c = self._decode_step(prev, h, c)
                    loss -= LowRankMultivariateNormal(mu, cf, cd).log_prob(y[:, t, :]).mean()
                    prev = y[:, t, :]                # teacher forcing

                loss = loss / pred_len
                if train:
                    loss.backward()
                    nn.utils.clip_grad_norm_(self._params(), 1.0)
                    self.optimizer.step()
                total += loss.item()

        return total / len(loader)

    def train_model(self, train_loader, val_loader):
        self.optimizer = torch.optim.Adam(self._params(), lr=self.lr)
        best_val, best_state, no_improve = float("inf"), None, 0

        for epoch in range(self.n_epochs):
            tr  = self._nll_pass(train_loader, train=True)
            val = self._nll_pass(val_loader,   train=False)
            self.train_losses.append(tr)
            self.val_losses.append(val)
            print(f"Epoch {epoch+1:3d}/{self.n_epochs}  train={tr:.4f}  val={val:.4f}")

            if val < best_val:
                best_val   = val
                best_state = self._save_state()
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        self._load_state(best_state)
        print(f"Best val NLL: {best_val:.4f}")


    # Sampling (parallel via repeat_interleave, following pytorch-ts
    # sampling_decoder)
    

    def sample(self, past: torch.Tensor, num_samples: int = 100) -> np.ndarray:
        # past: [batch, context_len, N]
        # returns: [num_samples, batch, pred_len, N]
        self._set_train(False)
        past = past.to(self.device)
        B    = past.shape[0]

        with torch.no_grad():
            h, c = self._encode(past)

            # Expand batch for parallel sample paths — following pytorch-ts repeat_interleave
            h    = h.repeat_interleave(num_samples, dim=1)               # [layers, B*S, H]
            c    = c.repeat_interleave(num_samples, dim=1)
            prev = past[:, -1, :].repeat_interleave(num_samples, dim=0) # [B*S, N]

            future_samples = []
            for _ in range(self.pred_len):
                mu, cf, cd, h, c = self._decode_step(prev, h, c)
                prev = LowRankMultivariateNormal(mu, cf, cd).sample()   # [B*S, N]
                future_samples.append(prev.unsqueeze(1))

            # [B*S, pred_len, N] -> [B, S, pred_len, N] -> [S, B, pred_len, N]
            samples = torch.cat(future_samples, dim=1)                   # [B*S, pred_len, N]
            samples = samples.reshape(B, num_samples, self.pred_len, self.input_size)
            samples = samples.permute(1, 0, 2, 3)                        # [S, B, pred_len, N]

        return samples.cpu().numpy().astype(np.float32)
