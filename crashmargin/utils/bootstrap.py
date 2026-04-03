"""
Bootstrap Confidence Intervals (Section 4.1).

Computes 95% bootstrap confidence intervals across 10 random seeds.
Used for reporting all metrics in the paper with statistical rigor.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def bootstrap_ci(
    values: np.ndarray | list[float],
    confidence: float = 0.95,
    n_bootstrap: int = 10_000,
    random_state: int | np.random.RandomState | None = None,
) -> dict:
    """Compute bootstrap confidence interval for a set of values.

    Uses the percentile method to construct confidence intervals from
    bootstrap resamples of the input values (typically metrics from
    10 random seeds as described in Section 4.1).

    Args:
        values: Array of metric values (e.g., from 10 seeds).
            Shape (n_values,).
        confidence: Confidence level for the interval. Default: 0.95.
        n_bootstrap: Number of bootstrap resamples. Default: 10000.
        random_state: Random seed or RandomState for reproducibility.

    Returns:
        Dictionary containing:
            - mean: Point estimate (sample mean).
            - std: Sample standard deviation.
            - ci_lower: Lower bound of the confidence interval.
            - ci_upper: Upper bound of the confidence interval.
            - confidence: The confidence level used.
            - n_values: Number of input values.
    """
    values = np.asarray(values, dtype=np.float64).ravel()
    n = len(values)

    if n == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "ci_lower": float("nan"),
            "ci_upper": float("nan"),
            "confidence": confidence,
            "n_values": 0,
        }

    if n == 1:
        val = float(values[0])
        return {
            "mean": val,
            "std": 0.0,
            "ci_lower": val,
            "ci_upper": val,
            "confidence": confidence,
            "n_values": 1,
        }

    rng = np.random.RandomState(random_state) if isinstance(
        random_state, (int, type(None))
    ) else random_state

    # Bootstrap resampling
    bootstrap_means = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        resample = rng.choice(values, size=n, replace=True)
        bootstrap_means[i] = np.mean(resample)

    # Percentile confidence interval
    alpha = 1 - confidence
    ci_lower = float(np.percentile(bootstrap_means, 100 * alpha / 2))
    ci_upper = float(np.percentile(bootstrap_means, 100 * (1 - alpha / 2)))

    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "confidence": confidence,
        "n_values": n,
    }


def aggregate_seed_results(
    seed_results: list[dict],
    metric_keys: list[str] | None = None,
    confidence: float = 0.95,
    n_bootstrap: int = 10_000,
    random_state: int | None = 42,
) -> dict[str, dict]:
    """Aggregate metrics across random seeds with bootstrap CIs.

    Takes a list of metric dictionaries (one per seed) and computes
    the mean and 95% bootstrap confidence interval for each metric.
    Used for the 10-seed evaluation protocol described in Section 4.1.

    Args:
        seed_results: List of dictionaries, each containing metric
            key-value pairs from one seed run.
        metric_keys: Which metrics to aggregate. If None, aggregates
            all numeric keys found in the first result dictionary.
        confidence: Confidence level for bootstrap CIs. Default: 0.95.
        n_bootstrap: Number of bootstrap resamples. Default: 10000.
        random_state: Random seed for reproducibility. Default: 42.

    Returns:
        Dictionary mapping each metric name to its bootstrap CI dict
        (with keys: mean, std, ci_lower, ci_upper, confidence, n_values).

    Example:
        >>> results = [{"auroc": 0.91}, {"auroc": 0.93}, ...]  # 10 seeds
        >>> agg = aggregate_seed_results(results)
        >>> print(f"AUROC: {agg['auroc']['mean']:.3f} "
        ...       f"[{agg['auroc']['ci_lower']:.3f}, "
        ...       f"{agg['auroc']['ci_upper']:.3f}]")
        AUROC: 0.920 [0.912, 0.928]
    """
    if len(seed_results) == 0:
        return {}

    # Determine which keys to aggregate
    if metric_keys is None:
        metric_keys = [
            k for k, v in seed_results[0].items()
            if isinstance(v, (int, float, np.integer, np.floating))
        ]

    aggregated = {}
    for key in metric_keys:
        values = []
        for result in seed_results:
            if key in result:
                val = result[key]
                if isinstance(val, (int, float, np.integer, np.floating)):
                    values.append(float(val))

        if values:
            aggregated[key] = bootstrap_ci(
                values,
                confidence=confidence,
                n_bootstrap=n_bootstrap,
                random_state=random_state,
            )

    return aggregated


def format_ci(ci_result: dict, decimals: int = 3) -> str:
    """Format a bootstrap CI result as a string for paper tables.

    Args:
        ci_result: Output from bootstrap_ci().
        decimals: Number of decimal places. Default: 3.

    Returns:
        Formatted string like "0.920 [0.912, 0.928]".
    """
    fmt = f"{{:.{decimals}f}}"
    mean_str = fmt.format(ci_result["mean"])
    lower_str = fmt.format(ci_result["ci_lower"])
    upper_str = fmt.format(ci_result["ci_upper"])
    return f"{mean_str} [{lower_str}, {upper_str}]"
