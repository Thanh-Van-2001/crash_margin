"""CrashMargin feature engineering pipeline (Section 3.2 of the paper).

Feature groups:
  - Market microstructure (47 features)
  - Vietnamese news sentiment (4 features)
  - Industry contagion & margin exposure graphs
  - Margin lending indicators (7 features)
  - Crash labels (Section 3.1)
"""

from crashmargin.features.market import MarketFeatureExtractor, compute_market_features
from crashmargin.features.sentiment import (
    SentimentFeatureExtractor,
    compute_sentiment_features,
)
from crashmargin.features.graph import (
    IndustryGraphBuilder,
    MarginExposureGraphBuilder,
    build_dual_graph,
)
from crashmargin.features.margin import MarginFeatureExtractor, compute_margin_features
from crashmargin.features.labels import CrashLabeler

__all__ = [
    "MarketFeatureExtractor",
    "compute_market_features",
    "SentimentFeatureExtractor",
    "compute_sentiment_features",
    "IndustryGraphBuilder",
    "MarginExposureGraphBuilder",
    "build_dual_graph",
    "MarginFeatureExtractor",
    "compute_margin_features",
    "CrashLabeler",
]
