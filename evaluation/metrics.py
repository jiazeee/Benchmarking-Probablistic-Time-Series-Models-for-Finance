"""
evaluation/metrics.py

Metrics for evaluating probabilistic forecasts.
All functions take:
    samples: np.ndarray [num_samples, T, N] - predicted samples
    targets: np.ndarray [T, N] - actual observed values

And return a scalar (lower is better for all metrics).
"""

import numpy as np


def crps_score(samples: np.ndarray, targets: np.ndarray) -> float:
    """
    Compute mean CRPS across all timesteps and series.

    CRPS for a single target y given samples x_1...x_M is:
        CRPS = E|X - y| - 0.5 * E|X - X'|
    where X, X' are independent draws from your forecast distribution.

    This is computed efficiently without needing the properscoring library.

    Args:
        samples: [num_samples, T, N]
        targets: [T, N]

    Returns:
        scalar — mean CRPS (lower is better)
    """
    M = samples.shape[0]

    # term 1: E|X - y|  →  mean over samples of |sample - target|
    # expand targets to [1, T, N] to broadcast against [M, T, N]
    term1 = np.abs(samples - targets[np.newaxis]).mean(axis=0)   # [T, N]

    # term 2: E|X - X'|  →  mean over all pairs of samples
    # efficient computation: for each pair (i,j), |x_i - x_j|
    # equivalent to: 2/(M*(M-1)) * sum_{i<j} |x_i - x_j|
    # but easier to compute as: mean over i,j of |x_i - x_j| * M/(M-1)
    # we use a simpler O(M^2) approach here — fine for M=100
    term2 = np.zeros(targets.shape)   # [T, N]
    for i in range(M):
        for j in range(i + 1, M):
            term2 += np.abs(samples[i] - samples[j])
    term2 = term2 / (M * (M - 1) / 2)   # normalize by number of pairs

    crps_per_step = term1 - 0.5 * term2   # [T, N]
    return float(crps_per_step.mean())


def crps_score_fast(samples: np.ndarray, targets: np.ndarray) -> float:
    """
    Faster CRPS computation using sorted samples.
    Use this once M gets large (>100 samples) — same result as crps_score.

    Based on the identity:
        E|X - X'| = 2/M^2 * sum_{i<j}|x_i - x_j|
                  = (2/M) * sum_i x_(i) * (i/M - (M-i)/M)   [sorted x]
    """
    M = samples.shape[0]

    term1 = np.abs(samples - targets[np.newaxis]).mean(axis=0)   # [T, N]

    # term 2 via sorted samples
    sorted_samples = np.sort(samples, axis=0)   # [M, T, N]
    # weights: (2i - M - 1) / M^2  for i=1..M
    weights = (2 * np.arange(1, M + 1) - M - 1) / (M ** 2)
    weights = weights.reshape(-1, 1, 1)   # [M, 1, 1] for broadcasting
    term2 = (weights * sorted_samples).sum(axis=0)   # [T, N]

    crps_per_step = term1 - term2
    return float(crps_per_step.mean())


def energy_score(samples: np.ndarray, targets: np.ndarray) -> float:
    """
    Multivariate Energy Score — captures joint distribution quality.
    Unlike CRPS which scores each series independently, this penalises
    models that get individual series right but miss cross-series correlations.

    ES = E||X - y||  -  0.5 * E||X - X'||
    where ||.|| is Euclidean norm across the N series dimension.

    Args:
        samples: [num_samples, T, N]
        targets: [T, N]

    Returns:
        scalar - mean Energy Score over timesteps (lower is better)
    """
    M = samples.shape[0]

    # term 1: E||X - y||  →  mean Euclidean distance from samples to target
    diff1 = samples - targets[np.newaxis]           # [M, T, N]
    term1 = np.sqrt((diff1 ** 2).sum(axis=-1))      # [M, T]  — norm over N
    term1 = term1.mean(axis=0)                       # [T]

    # term 2: E||X - X'||  →  mean Euclidean distance between sample pairs
    term2 = np.zeros(targets.shape[0])   # [T]
    n_pairs = 0
    for i in range(M):
        for j in range(i + 1, M):
            diff2 = samples[i] - samples[j]                    # [T, N]
            term2 += np.sqrt((diff2 ** 2).sum(axis=-1))        # [T]
            n_pairs += 1
    term2 /= n_pairs

    es_per_step = term1 - 0.5 * term2   # [T]
    return float(es_per_step.mean())


def quantile_loss(samples: np.ndarray, targets: np.ndarray,
                  quantiles: list = [0.1, 0.5, 0.9]) -> dict:
    """
    Compute pinball loss at specified quantiles.
    Useful for checking calibration — e.g. does the 90th percentile of your
    samples actually contain the true value 90% of the time?

    Args:
        samples:   [num_samples, T, N]
        targets:   [T, N]
        quantiles: list of quantile levels to evaluate

    Returns:
        dict mapping quantile level -> mean pinball loss
    """
    results = {}
    for q in quantiles:
        # estimate quantile from samples
        q_hat = np.quantile(samples, q, axis=0)   # [T, N]

        # pinball loss
        error = targets - q_hat
        loss  = np.where(error >= 0, q * error, (q - 1) * error)
        results[f"QL_{q:.2f}"] = float(loss.mean())

    return results


def coverage(samples: np.ndarray, targets: np.ndarray,
             levels: list = [0.5, 0.9]) -> dict:
    """
    Empirical coverage at specified prediction interval levels.
    For a well-calibrated model, coverage at level L should be ~L.

    E.g. if you ask for 90% coverage and get 0.65, your intervals are
    too narrow — the model is overconfident.

    Args:
        samples: [num_samples, T, N]
        targets: [T, N]
        levels:  list of interval levels

    Returns:
        dict mapping level -> empirical coverage fraction
    """
    results = {}
    for level in levels:
        alpha = (1 - level) / 2
        lower = np.quantile(samples, alpha,         axis=0)   # [T, N]
        upper = np.quantile(samples, 1 - alpha,     axis=0)   # [T, N]
        inside = (targets >= lower) & (targets <= upper)
        results[f"Coverage_{level:.0%}"] = float(inside.mean())
    return results


def evaluate_all(samples: np.ndarray, targets: np.ndarray) -> dict:
    """
    Run all metrics and return as a single dict.

    Args:
        samples: [num_samples, T, N]
        targets: [T, N]

    Returns:
        dict of metric_name -> value
    """
    results = {}
    results["CRPS"]         = crps_score_fast(samples, targets)
    results["EnergyScore"]  = energy_score(samples, targets)
    results.update(quantile_loss(samples, targets))
    results.update(coverage(samples, targets))
    return results
