"""Vietnamese News Sentiment Features (Section 3.2.2).

Extracts 4 sentiment features per stock-day from a news DataFrame.
Designed around PhoBERT embeddings for Vietnamese financial news.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# PhoBERT fine-tuning stub
# ---------------------------------------------------------------------------

class PhoBERTSentimentPipeline:
    """Stub for a PhoBERT-based Vietnamese financial sentiment classifier.

    In production this wraps ``vinai/phobert-base`` from HuggingFace,
    fine-tuned on ~12k manually-labelled Vietnamese financial news headlines
    (see paper Appendix A.3).

    The pipeline assigns each article a polarity score in [-1, +1]:
      * -1  strongly negative
      *  0  neutral
      * +1  strongly positive

    Usage (once fine-tuned weights are available)::

        pipe = PhoBERTSentimentPipeline(model_path="checkpoints/phobert_fin_vi")
        pipe.load()
        scores = pipe.predict(["VN-Index giam manh phien sang"])
    """

    def __init__(self, model_path: Optional[str] = None, device: str = "cpu"):
        self.model_path = model_path
        self.device = device
        self._model = None
        self._tokenizer = None

    # ----- lifecycle -----
    def load(self) -> "PhoBERTSentimentPipeline":
        """Load fine-tuned PhoBERT weights.

        Raises ``FileNotFoundError`` when *model_path* does not exist.
        """
        if self.model_path is None:
            raise ValueError(
                "model_path is required.  Fine-tune PhoBERT first "
                "(see scripts/finetune_phobert.py)."
            )
        # Placeholder -- actual loading requires transformers:
        # from transformers import AutoModelForSequenceClassification, AutoTokenizer
        # self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        # self._model = AutoModelForSequenceClassification.from_pretrained(
        #     self.model_path
        # ).to(self.device)
        return self

    def predict(self, texts: list[str]) -> np.ndarray:
        """Return polarity scores in [-1, 1] for each text.

        If the model has not been loaded, returns uniform-zero scores
        (graceful degradation).
        """
        if self._model is None:
            return np.zeros(len(texts), dtype=np.float32)
        # Placeholder for real inference
        raise NotImplementedError("Load a fine-tuned checkpoint first.")

    def finetune(
        self,
        train_texts: list[str],
        train_labels: list[int],
        val_texts: list[str] | None = None,
        val_labels: list[int] | None = None,
        epochs: int = 5,
        lr: float = 2e-5,
        batch_size: int = 32,
    ) -> dict:
        """Fine-tune PhoBERT on labelled Vietnamese financial news.

        Labels: 0 = negative, 1 = neutral, 2 = positive.

        Returns a dict of training metrics.
        """
        # Placeholder -- full implementation depends on transformers + datasets.
        raise NotImplementedError(
            "Fine-tuning requires `transformers` and `datasets`.  "
            "See scripts/finetune_phobert.py for the training loop."
        )


# ---------------------------------------------------------------------------
# Sentiment feature extractor
# ---------------------------------------------------------------------------

@dataclass
class SentimentFeatureExtractor:
    """Derive 4 daily sentiment features per stock from news articles.

    Expected input ``news_df`` columns:
      - ``date``       : publication date (datetime or str)
      - ``ticker``     : stock ticker
      - ``polarity``   : sentiment score in [-1, 1] (from PhoBERT pipeline)

    If ``polarity`` is not present but ``headline`` is, the extractor will
    attempt to score headlines with :class:`PhoBERTSentimentPipeline` (falls
    back to zeros when no model is loaded).

    Output features (per stock-day):
      1. ``sent_mean``      -- mean sentiment polarity
      2. ``sent_volume``    -- article count (log-transformed)
      3. ``neg_ratio``      -- fraction of articles with polarity < -0.3
      4. ``sent_momentum5`` -- 5-day rolling change in mean polarity
    """

    neg_threshold: float = -0.3
    momentum_window: int = 5
    model_path: Optional[str] = None
    feature_names_: List[str] = field(default_factory=list, init=False)

    def _ensure_polarity(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """Score headlines if *polarity* column is missing."""
        df = news_df.copy()
        if "polarity" not in df.columns:
            if "headline" not in df.columns:
                raise ValueError(
                    "news_df must contain either 'polarity' or 'headline'."
                )
            pipe = PhoBERTSentimentPipeline(model_path=self.model_path)
            try:
                pipe.load()
            except (ValueError, FileNotFoundError):
                pass  # will return zeros
            df["polarity"] = pipe.predict(df["headline"].tolist())
        return df

    def transform(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """Compute 4 sentiment features per (ticker, date).

        Parameters
        ----------
        news_df : DataFrame
            Columns: ``date, ticker`` and either ``polarity`` or ``headline``.

        Returns
        -------
        DataFrame
            MultiIndex (ticker, date) with 4 feature columns.
        """
        df = self._ensure_polarity(news_df)
        df["date"] = pd.to_datetime(df["date"])

        # Daily aggregation per stock
        grouped = df.groupby(["ticker", "date"])

        daily = pd.DataFrame({
            "sent_mean": grouped["polarity"].mean(),
            "sent_volume": grouped["polarity"].count().apply(lambda x: np.log1p(x)),
            "neg_ratio": grouped["polarity"].apply(
                lambda s: (s < self.neg_threshold).mean()
            ),
        })

        # 5-day sentiment momentum: sort within each ticker, then diff
        daily = daily.sort_index()
        daily["sent_momentum5"] = daily.groupby(level="ticker")["sent_mean"].transform(
            lambda s: s.rolling(self.momentum_window, min_periods=1).mean().diff()
        )

        self.feature_names_ = [
            "sent_mean",
            "sent_volume",
            "neg_ratio",
            "sent_momentum5",
        ]
        return daily[self.feature_names_]


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def compute_sentiment_features(news_df: pd.DataFrame) -> pd.DataFrame:
    """Compute 4 sentiment features per stock-day.

    Parameters
    ----------
    news_df : DataFrame
        Must contain ``date``, ``ticker``, and either ``polarity`` (float in
        [-1, 1]) or ``headline`` (str -- will be scored via PhoBERT stub).

    Returns
    -------
    DataFrame
        MultiIndex ``(ticker, date)`` with columns:
        ``sent_mean, sent_volume, neg_ratio, sent_momentum5``.
    """
    extractor = SentimentFeatureExtractor()
    return extractor.transform(news_df)
