"""
TimeGrad: Autoregressive Denoising Diffusion Models for Multivariate
Probabilistic Time Series Forecasting.
Rasul et al., ICML 2021.

Adapted from pytorch-ts (zalandoresearch/pytorch-ts), which is itself based on
GluonTS (awslabs/gluonts). EpsilonTheta and GaussianDiffusion are copied
verbatim from:
  pts/model/time_grad/epsilon_theta.py
  pts/modules/gaussian_diffusion.py

Simplifications vs. the original pytorch-ts implementation:
  - No time covariates (omitted; financial returns lack strong calendar seasonality)
  - Lag-1 inputs only (original uses multiple lags requiring extended history)
  - Data pre-normalised upstream by Normalizer; MeanScaler omitted
  - No missing-value masking (clean financial data assumed)
  - Parallel sampling via repeat_interleave follows pytorch-ts sampling_decoder
"""

import math
from functools import partial
from inspect import isfunction

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import ProbForecastModel



# Copied verbatim from pts/model/time_grad/epsilon_theta.py (pytorch-ts)


class DiffusionEmbedding(nn.Module):
    def __init__(self, dim, proj_dim, max_steps=500):
        super().__init__()
        self.register_buffer(
            "embedding", self._build_embedding(dim, max_steps), persistent=False
        )
        self.projection1 = nn.Linear(dim * 2, proj_dim)
        self.projection2 = nn.Linear(proj_dim, proj_dim)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    def _build_embedding(self, dim, max_steps):
        steps = torch.arange(max_steps).unsqueeze(1)   # [T, 1]
        dims  = torch.arange(dim).unsqueeze(0)          # [1, dim]
        table = steps * 10.0 ** (dims * 4.0 / dim)     # [T, dim]
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
        return table


class ResidualBlock(nn.Module):
    def __init__(self, hidden_size, residual_channels, dilation):
        super().__init__()
        self.dilated_conv = nn.Conv1d(
            residual_channels,
            2 * residual_channels,
            3,
            padding=dilation,
            dilation=dilation,
            padding_mode="circular",
        )
        self.diffusion_projection  = nn.Linear(hidden_size, residual_channels)
        self.conditioner_projection = nn.Conv1d(
            1, 2 * residual_channels, 1, padding=2, padding_mode="circular"
        )
        self.output_projection = nn.Conv1d(residual_channels, 2 * residual_channels, 1)

        nn.init.kaiming_normal_(self.conditioner_projection.weight)
        nn.init.kaiming_normal_(self.output_projection.weight)

    def forward(self, x, conditioner, diffusion_step):
        diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1)
        conditioner    = self.conditioner_projection(conditioner)

        y = x + diffusion_step
        y = self.dilated_conv(y) + conditioner

        gate, filter_ = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter_)

        y = self.output_projection(y)
        y = F.leaky_relu(y, 0.4)
        residual, skip = torch.chunk(y, 2, dim=1)
        return (x + residual) / math.sqrt(2.0), skip


class CondUpsampler(nn.Module):
    def __init__(self, cond_length, target_dim):
        super().__init__()
        self.linear1 = nn.Linear(cond_length, target_dim // 2)
        self.linear2 = nn.Linear(target_dim // 2, target_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = F.leaky_relu(x, 0.4)
        x = self.linear2(x)
        x = F.leaky_relu(x, 0.4)
        return x


class EpsilonTheta(nn.Module):
    def __init__(
        self,
        target_dim,
        cond_length,
        time_emb_dim=16,
        residual_layers=8,
        residual_channels=8,
        dilation_cycle_length=2,
        residual_hidden=64,
    ):
        super().__init__()
        self.input_projection = nn.Conv1d(
            1, residual_channels, 1, padding=2, padding_mode="circular"
        )
        self.diffusion_embedding = DiffusionEmbedding(
            time_emb_dim, proj_dim=residual_hidden
        )
        self.cond_upsampler = CondUpsampler(
            target_dim=target_dim, cond_length=cond_length
        )
        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    residual_channels=residual_channels,
                    dilation=2 ** (i % dilation_cycle_length),
                    hidden_size=residual_hidden,
                )
                for i in range(residual_layers)
            ]
        )
        self.skip_projection   = nn.Conv1d(residual_channels, residual_channels, 3)
        self.output_projection = nn.Conv1d(residual_channels, 1, 3)

        nn.init.kaiming_normal_(self.input_projection.weight)
        nn.init.kaiming_normal_(self.skip_projection.weight)
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, inputs, time, cond):
        x = self.input_projection(inputs)
        x = F.leaky_relu(x, 0.4)

        diffusion_step = self.diffusion_embedding(time)
        cond_up = self.cond_upsampler(cond)
        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, cond_up, diffusion_step)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        x = self.skip_projection(x)
        x = F.leaky_relu(x, 0.4)
        x = self.output_projection(x)
        return x



# Copied verbatim from pts/modules/gaussian_diffusion.py (pytorch-ts)


def _default(val, d):
    if val is not None:
        return val
    return d() if isfunction(d) else d


def _extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def _noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(
        shape[0], *((1,) * (len(shape) - 1))
    )
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()


def _cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = np.linspace(0, timesteps, steps)
    alphas_cumprod = np.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, 0, 0.999)


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn,
        input_size,
        beta_end=0.1,
        diff_steps=100,
        loss_type="l2",
        betas=None,
        beta_schedule="linear",
    ):
        super().__init__()
        self.denoise_fn = denoise_fn
        self.input_size = input_size
        self.__scale   = None

        if betas is not None:
            betas = (
                betas.detach().cpu().numpy()
                if isinstance(betas, torch.Tensor)
                else betas
            )
        else:
            if beta_schedule == "linear":
                betas = np.linspace(1e-4, beta_end, diff_steps)
            elif beta_schedule == "quad":
                betas = np.linspace(1e-4 ** 0.5, beta_end ** 0.5, diff_steps) ** 2
            elif beta_schedule == "const":
                betas = beta_end * np.ones(diff_steps)
            elif beta_schedule == "jsd":
                betas = 1.0 / np.linspace(diff_steps, 1, diff_steps)
            elif beta_schedule == "sigmoid":
                betas = np.linspace(-6, 6, diff_steps)
                betas = (beta_end - 1e-4) / (np.exp(-betas) + 1) + 1e-4
            elif beta_schedule == "cosine":
                betas = _cosine_beta_schedule(diff_steps)
            else:
                raise NotImplementedError(beta_schedule)

        alphas             = 1.0 - betas
        alphas_cumprod     = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        (timesteps,)          = betas.shape
        self.num_timesteps    = int(timesteps)
        self.loss_type        = loss_type

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer("betas",             to_torch(betas))
        self.register_buffer("alphas_cumprod",    to_torch(alphas_cumprod))
        self.register_buffer("alphas_cumprod_prev", to_torch(alphas_cumprod_prev))

        self.register_buffer("sqrt_alphas_cumprod",
                             to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer("sqrt_one_minus_alphas_cumprod",
                             to_torch(np.sqrt(1.0 - alphas_cumprod)))
        self.register_buffer("log_one_minus_alphas_cumprod",
                             to_torch(np.log(1.0 - alphas_cumprod)))
        self.register_buffer("sqrt_recip_alphas_cumprod",
                             to_torch(np.sqrt(1.0 / alphas_cumprod)))
        self.register_buffer("sqrt_recipm1_alphas_cumprod",
                             to_torch(np.sqrt(1.0 / alphas_cumprod - 1)))

        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.register_buffer("posterior_variance", to_torch(posterior_variance))
        self.register_buffer(
            "posterior_log_variance_clipped",
            to_torch(np.log(np.maximum(posterior_variance, 1e-20))),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            to_torch(
                (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod)
            ),
        )

    @property
    def scale(self):
        return self.__scale

    @scale.setter
    def scale(self, scale):
        self.__scale = scale

    def q_mean_variance(self, x_start, t):
        mean         = _extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance     = _extract(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            _extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            _extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance             = _extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, cond, t, clip_denoised: bool):
        x_recon = self.predict_start_from_noise(
            x, t=t, noise=self.denoise_fn(x, t, cond=cond)
        )
        if clip_denoised:
            x_recon.clamp_(-1.0, 1.0)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, cond, t, clip_denoised=False, repeat_noise=False):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(
            x=x, cond=cond, t=t, clip_denoised=clip_denoised
        )
        noise         = _noise_like(x.shape, device, repeat_noise)
        nonzero_mask  = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, cond):
        device = self.betas.device
        b      = shape[0]
        img    = torch.randn(shape, device=device)
        for i in reversed(range(0, self.num_timesteps)):
            img = self.p_sample(
                img, cond, torch.full((b,), i, device=device, dtype=torch.long)
            )
        return img

    @torch.no_grad()
    def sample(self, sample_shape=torch.Size(), cond=None):
        if cond is not None:
            shape = cond.shape[:-1] + (self.input_size,)
        else:
            shape = sample_shape
        x_hat = self.p_sample_loop(shape, cond)
        if self.scale is not None:
            x_hat *= self.scale
        return x_hat

    def q_sample(self, x_start, t, noise=None):
        noise = _default(noise, lambda: torch.randn_like(x_start))
        return (
            _extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, cond, t, noise=None):
        noise   = _default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_recon = self.denoise_fn(x_noisy, t, cond=cond)

        if self.loss_type == "l1":
            loss = F.l1_loss(x_recon, noise)
        elif self.loss_type == "l2":
            loss = F.mse_loss(x_recon, noise)
        elif self.loss_type == "huber":
            loss = F.smooth_l1_loss(x_recon, noise)
        else:
            raise NotImplementedError()
        return loss

    def log_prob(self, x, cond, *args, **kwargs):
        if self.scale is not None:
            x = x / self.scale
        B, T, _ = x.shape
        time = torch.randint(0, self.num_timesteps, (B * T,), device=x.device).long()
        loss = self.p_losses(
            x.reshape(B * T, 1, -1), cond.reshape(B * T, 1, -1), time, *args, **kwargs
        )
        return loss



# TimeGrad wrapper — interface adapted to (x, y) DataLoader format


class TimeGrad(ProbForecastModel):
    """
    Adapted from pytorch-ts TimeGradTrainingNetwork / TimeGradPredictionNetwork
    (zalandoresearch/pytorch-ts). Core modules EpsilonTheta and GaussianDiffusion
    are copied verbatim from pytorch-ts.

    Training follows pytorch-ts: the GRU unrolls over the full
    context + prediction window (teacher-forced) and the diffusion loss is
    computed at every timestep.

    Sampling follows pytorch-ts sampling_decoder: all sample paths are decoded
    in a single batched forward pass via repeat_interleave, then reshaped.
    """

    def __init__(
        self,
        input_size,
        pred_len=21,
        hidden_size=64,
        conditioning_length=64,
        diff_steps=100,
        loss_type="l2",
        beta_end=0.1,
        beta_schedule="linear",
        residual_layers=8,
        residual_channels=8,
        dilation_cycle_length=2,
        lr=1e-3,
        n_epochs=50,
        patience=5,
        device="cpu",
    ):
        self.input_size          = input_size
        self.pred_len            = pred_len
        self.hidden_size         = hidden_size
        self.conditioning_length = conditioning_length
        self.lr                  = lr
        self.n_epochs            = n_epochs
        self.patience            = patience
        self.device              = device
        self.optimizer           = None
        self.train_losses, self.val_losses = [], []

        # GRU encoder — processes lag-1 inputs (context + teacher-forced future)
        self.rnn      = nn.GRU(input_size, hidden_size, batch_first=True).to(device)
        # Project RNN hidden states to conditioning vectors for EpsilonTheta
        self.proj_cond = nn.Linear(hidden_size, conditioning_length).to(device)

        # EpsilonTheta (dilated residual conv noise predictor) — copied from pytorch-ts
        denoise_fn = EpsilonTheta(
            target_dim=input_size,
            cond_length=conditioning_length,
            residual_layers=residual_layers,
            residual_channels=residual_channels,
            dilation_cycle_length=dilation_cycle_length,
        )
        # GaussianDiffusion — copied from pytorch-ts; owns denoise_fn as submodule
        self.diffusion = GaussianDiffusion(
            denoise_fn,
            input_size=input_size,
            diff_steps=diff_steps,
            loss_type=loss_type,
            beta_end=beta_end,
            beta_schedule=beta_schedule,
        ).to(device)

    def _nets(self):
        return [self.rnn, self.proj_cond, self.diffusion]

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
            "rnn":       {k: v.cpu().clone() for k, v in self.rnn.state_dict().items()},
            "proj_cond": {k: v.cpu().clone() for k, v in self.proj_cond.state_dict().items()},
            "diffusion": {k: v.cpu().clone() for k, v in self.diffusion.state_dict().items()},
        }

    def _load_state(self, state):
        self.rnn.load_state_dict(state["rnn"])
        self.proj_cond.load_state_dict(state["proj_cond"])
        self.diffusion.load_state_dict(state["diffusion"])
        for net in self._nets():
            net.to(self.device)


    # Training (follows pytorch-ts: loss over full context + pred window)


    def _loss_pass(self, loader, train=False):
        self._set_train(train)
        total = 0.0
        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)

                # Concatenate context and future for teacher-forced full-sequence loss
                # Following pytorch-ts unroll_encoder with future_target_cdf provided
                full_seq = torch.cat([x, y], dim=1)  # [B, context_len + pred_len, N]

                rnn_out, _ = self.rnn(full_seq)       # [B, context_len + pred_len, H]
                cond = self.proj_cond(rnn_out)         # [B, context_len + pred_len, cond_len]

                # GaussianDiffusion.log_prob: reshapes internally to [B*T, 1, N]
                loss = self.diffusion.log_prob(full_seq, cond)

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self._params(), 1.0)
                    self.optimizer.step()

                total += loss.item()

        return total / len(loader)

    def train_model(self, train_loader, val_loader):
        self.optimizer = torch.optim.Adam(self._params(), lr=self.lr)
        best_val, best_state, no_improve = float("inf"), None, 0

        for epoch in range(self.n_epochs):
            tr  = self._loss_pass(train_loader, train=True)
            val = self._loss_pass(val_loader,   train=False)
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
        print(f"Best val loss: {best_val:.4f}")


    # Sampling (follows pytorch-ts sampling_decoder: parallel via repeat_interleave)


    def sample(self, past: torch.Tensor, num_samples: int = 100) -> np.ndarray:
        # past: [batch, context_len, N]
        # returns: [num_samples, batch, pred_len, N]
        self._set_train(False)
        past = past.to(self.device)
        B    = past.shape[0]

        with torch.no_grad():
            # Encode context
            _, h = self.rnn(past)   # h: [1, B, H]

            # Expand batch for parallel sample paths — following pytorch-ts repeat_interleave
            h    = h.repeat_interleave(num_samples, dim=1)       # [1, B*S, H]
            prev = past[:, -1, :].repeat_interleave(num_samples, dim=0)  # [B*S, N]

            future_samples = []
            for _ in range(self.pred_len):
                rnn_out, h = self.rnn(prev.unsqueeze(1), h)    # rnn_out: [B*S, 1, H]
                cond = self.proj_cond(rnn_out)                  # [B*S, 1, cond_len]

                # Reverse diffusion to sample next step — following pytorch-ts diffusion.sample
                new_sample = self.diffusion.sample(cond=cond)  # [B*S, 1, N]
                future_samples.append(new_sample)
                prev = new_sample.squeeze(1)                    # [B*S, N]

            # [B*S, pred_len, N] -> reshape -> [B, S, pred_len, N] -> [S, B, pred_len, N]
            samples = torch.cat(future_samples, dim=1)                        # [B*S, pred_len, N]
            samples = samples.reshape(B, num_samples, self.pred_len, self.input_size)
            samples = samples.permute(1, 0, 2, 3)                             # [S, B, pred_len, N]

        return samples.cpu().numpy().astype(np.float32)
