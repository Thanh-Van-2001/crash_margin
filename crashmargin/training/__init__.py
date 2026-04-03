"""CrashMargin training pipeline (Section 4).

Walk-forward training with focal loss, evaluation metrics, and
bootstrap confidence intervals for the ICAIF 2026 paper.
"""

from crashmargin.training.losses import FocalLoss
from crashmargin.training.trainer import WalkForwardTrainer
from crashmargin.training.metrics import (
    compute_classification_metrics,
    delong_test,
    compute_economic_metrics,
)

__all__ = [
    "FocalLoss",
    "WalkForwardTrainer",
    "compute_classification_metrics",
    "delong_test",
    "compute_economic_metrics",
]
