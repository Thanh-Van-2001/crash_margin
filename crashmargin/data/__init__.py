"""CrashMargin data loading and dataset utilities."""

from crashmargin.data.dataset import CrashMarginDataset, create_temporal_splits
from crashmargin.data.synthetic import SyntheticVNData

__all__ = [
    "CrashMarginDataset",
    "create_temporal_splits",
    "SyntheticVNData",
]
