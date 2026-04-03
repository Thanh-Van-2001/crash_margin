"""CrashMargin utility functions.

Bootstrap confidence interval computation and other shared utilities.
"""

from crashmargin.utils.bootstrap import bootstrap_ci, aggregate_seed_results

__all__ = [
    "bootstrap_ci",
    "aggregate_seed_results",
]
