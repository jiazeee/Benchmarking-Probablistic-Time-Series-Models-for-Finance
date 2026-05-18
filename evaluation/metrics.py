"""
evaluation/metrics.py

Metrics for evaluating probabilistic forecasts.

All functions expect:
    samples: np.ndarray [num_samples, T, N]  — predicted sample paths
    targets: np.ndarray [T, N]               — observed values

All inputs should be in the ORIGINAL (inverse-transformed) scale.

Metrics implemented via scoringrules (frazane/scoringrules), which provides
vectorised, numerically stable proper scoring rule implementations:

    CRPS          — marginal calibration per series
    CRPS_Sum      — sum of marginal CRPSes across series (mean over time)
    Energy Score  — joint distribution quality (captures cross-series correlations)
    Variogram Score — tests whether modelled covariance structure matches observed
    Quantile Loss — pinball loss at specified quantile levels
    Coverage      — empirical prediction interval coverage
"""

import numpy as np
import scoringrules


def crps(samples: np.ndarray, targets: np.ndarray) -> float:
    """
    Mean CRPS averaged over all timesteps and series.

    Uses scoringrules.crps_ensemble with the energy-score estimator (nrg),
    consistent with the standard definition in Gneiting & Raftery (2007).

    Args:
        samples: [M, T, N]
        targets: [T, N]

    Returns:
        scalar — mean CRPS (lower is better)
    """
    # scoringrules expects fct: [..., M] with members on the last axis
    fct = samples.transpose(1, 2, 0)                    # [T, N, M]
    scores = scoringrules.crps_ensemble(targets, fct, estimator="nrg")  # [T, N]
    return float(scores.mean())


def crps_sum(samples: np.ndarray, targets: np.ndarray) -> float:
    """
    CRPS-Sum: sum of marginal CRPSes across series, averaged over time.

    Used in TimeGrad (Rasul et al. 2021) and related work as a scale-sensitive
    multivariate metric. Measures total calibration across all series jointly.

    Args:
        samples: [M, T, N]
        targets: [T, N]

    Returns:
        scalar — CRPS-Sum (lower is better)
    """
    fct = samples.transpose(1, 2, 0)                    # [T, N, M]
    scores = scoringrules.crps_ensemble(targets, fct, estimator="nrg")  # [T, N]
    return float(scores.sum(axis=-1).mean())             # sum over N, mean over T


def energy_score(samples: np.ndarray, targets: np.ndarray) -> float:
    """
    Energy Score — multivariate proper scoring rule.

    Captures joint distribution quality including cross-series correlations.
    Reduces to CRPS for univariate forecasts.

    ES(F, y) = E||X - y|| - 0.5 * E||X - X'||

    Uses scoringrules.es_ensemble with the energy estimator (nrg).

    Args:
        samples: [M, T, N]
        targets: [T, N]

    Returns:
        scalar — mean Energy Score over timesteps (lower is better)
    """
    # scoringrules expects fct: [..., M, N] (members second-to-last, variables last)
    fct = samples.transpose(1, 0, 2)                    # [T, M, N]
    scores = scoringrules.es_ensemble(targets, fct, estimator="nrg")  # [T]
    return float(scores.mean())


def variogram_score(samples: np.ndarray, targets: np.ndarray, p: float = 0.5) -> float:
    """
    Variogram Score — tests whether the forecast captures cross-series
    dependence structure (covariances), not just marginal distributions.

    A model can have good CRPS but poor Variogram Score if it gets each
    series right independently but misses their correlations.

    Uses scoringrules.vs_ensemble (p=0.5 by default, following Scheuerer &
    Hamill 2015).

    Args:
        samples: [M, T, N]
        targets: [T, N]
        p:       order parameter (0.5 is standard)

    Returns:
        scalar — mean Variogram Score over timesteps (lower is better)
    """
    # scoringrules expects fct: [..., M, N]
    fct = samples.transpose(1, 0, 2)                    # [T, M, N]
    scores = scoringrules.vs_ensemble(targets, fct, p=p, estimator="nrg")  # [T]
    return float(scores.mean())


def quantile_loss(
    samples: np.ndarray,
    targets: np.ndarray,
    quantiles: list = [0.1, 0.5, 0.9],
) -> dict:
    """
    Pinball (quantile) loss at specified quantile levels.

    Args:
        samples:   [M, T, N]
        targets:   [T, N]
        quantiles: list of quantile levels

    Returns:
        dict mapping "QL_q" -> mean pinball loss
    """
    results = {}
    for q in quantiles:
        q_hat = np.quantile(samples, q, axis=0)         # [T, N]
        error = targets - q_hat
        loss  = np.where(error >= 0, q * error, (q - 1) * error)
        results[f"QL_{q:.2f}"] = float(loss.mean())
    return results


def coverage(
    samples: np.ndarray,
    targets: np.ndarray,
    levels: list = [0.5, 0.9],
) -> dict:
    """
    Empirical prediction interval coverage.

    For a well-calibrated model, coverage at level L should equal L.
    Values below L indicate overconfident (too-narrow) intervals.

    Args:
        samples: [M, T, N]
        targets: [T, N]
        levels:  interval coverage levels

    Returns:
        dict mapping "Coverage_X%" -> empirical fraction
    """
    results = {}
    for level in levels:
        alpha = (1 - level) / 2
        lower = np.quantile(samples, alpha,     axis=0)  # [T, N]
        upper = np.quantile(samples, 1 - alpha, axis=0)  # [T, N]
        inside = (targets >= lower) & (targets <= upper)
        results[f"Coverage_{level:.0%}"] = float(inside.mean())
    return results


def evaluate_all(samples: np.ndarray, targets: np.ndarray) -> dict:
    """
    Run all metrics and return as a single flat dict.

    Inputs should be in original (inverse-transformed) scale.

    Args:
        samples: [M, T, N]
        targets: [T, N]

    Returns:
        dict of metric_name -> scalar value (all lower is better except Coverage,
        which should be close to the nominal level)
    """
    results = {}
    results["CRPS"]            = crps(samples, targets)
    results["CRPS_Sum"]        = crps_sum(samples, targets)
    results["EnergyScore"]     = energy_score(samples, targets)
    results["VariogramScore"]  = variogram_score(samples, targets)
    results.update(quantile_loss(samples, targets))
    results.update(coverage(samples, targets))
    return results
