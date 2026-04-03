"""PyTorch Dataset for CrashMargin (Section 3).

Each sample returns:
  - market_features : (20, 47) -- 20-day lookback of market microstructure
  - sentiment       : (20, 4)  -- 20-day lookback of sentiment features
  - graph_feat      : (n_stocks, node_feat_dim) -- node features at time t
  - margin_feat     : (7,) -- margin lending features at time t
  - label           : scalar binary crash label

Temporal splits follow the paper:
  - Train : 2018--2020
  - Val   : 2021
  - Test  : 2022--2024  (walk-forward retraining every 6 months)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

@dataclass
class _SampleIndex:
    """Lightweight pointer into the feature matrices for a single sample."""
    ticker_idx: int
    time_idx: int  # index of the *label* date in the weekly label array
    date: pd.Timestamp


class CrashMarginDataset(Dataset):
    """PyTorch dataset for the CrashMargin model.

    Parameters
    ----------
    market_features : dict[str, DataFrame]
        ``{ticker: DataFrame(dates x 47)}``.
    sentiment_features : dict[str, DataFrame]
        ``{ticker: DataFrame(dates x 4)}``.
    graph_node_features : ndarray
        Shape ``(n_stocks, node_feat_dim)`` -- static or time-indexed.
    adj_industry : ndarray or dict
        ``(n_stocks, n_stocks)`` or ``{date: ndarray}``.
    adj_margin : ndarray or dict
        Same as *adj_industry*.
    margin_features : dict[str, DataFrame]
        ``{ticker: DataFrame(dates x 7)}``.
    labels : DataFrame
        Weekly binary crash labels (columns = tickers, index = dates).
    tickers : list[str]
        Ordered list of tickers (defines node ordering in graphs).
    lookback : int
        Number of trading days of history per sample (default 20).
    """

    def __init__(
        self,
        market_features: Dict[str, pd.DataFrame],
        sentiment_features: Dict[str, pd.DataFrame],
        graph_node_features: np.ndarray,
        adj_industry: np.ndarray | Dict[pd.Timestamp, np.ndarray],
        adj_margin: np.ndarray | Dict[pd.Timestamp, np.ndarray],
        margin_features: Dict[str, pd.DataFrame],
        labels: pd.DataFrame,
        tickers: List[str],
        lookback: int = 20,
    ):
        self.market_features = market_features
        self.sentiment_features = sentiment_features
        self.graph_node_features = graph_node_features
        self.adj_industry = adj_industry
        self.adj_margin = adj_margin
        self.margin_features = margin_features
        self.labels = labels
        self.tickers = tickers
        self.lookback = lookback

        self._ticker_to_idx = {t: i for i, t in enumerate(tickers)}
        self._samples = self._build_index()

    # ------------------------------------------------------------------ #
    def _build_index(self) -> List[_SampleIndex]:
        """Enumerate all valid (ticker, week) samples."""
        samples: list[_SampleIndex] = []
        label_dates = self.labels.index

        for t_idx, ticker in enumerate(self.tickers):
            if ticker not in self.labels.columns:
                continue
            mkt = self.market_features.get(ticker)
            if mkt is None:
                continue
            mkt_dates = mkt.index

            for w_idx, w_date in enumerate(label_dates):
                lbl = self.labels.loc[w_date, ticker]
                if np.isnan(lbl):
                    continue

                # Need `lookback` trading days ending on or before `w_date`
                available = mkt_dates[mkt_dates <= w_date]
                if len(available) < self.lookback:
                    continue

                samples.append(_SampleIndex(ticker_idx=t_idx, time_idx=w_idx, date=w_date))

        return samples

    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        sample = self._samples[idx]
        ticker = self.tickers[sample.ticker_idx]
        w_date = sample.date

        # --- market features (20, 47) ---
        mkt_df = self.market_features[ticker]
        avail = mkt_df.index[mkt_df.index <= w_date]
        window = avail[-self.lookback :]
        mkt_arr = mkt_df.loc[window].values.astype(np.float32)
        # Pad if needed
        if mkt_arr.shape[0] < self.lookback:
            pad = np.zeros((self.lookback - mkt_arr.shape[0], mkt_arr.shape[1]), dtype=np.float32)
            mkt_arr = np.concatenate([pad, mkt_arr], axis=0)
        mkt_arr = np.nan_to_num(mkt_arr, nan=0.0)

        # --- sentiment features (20, 4) ---
        sent_df = self.sentiment_features.get(ticker)
        if sent_df is not None and len(sent_df) > 0:
            sent_avail = sent_df.index[sent_df.index <= w_date]
            if len(sent_avail) >= self.lookback:
                sent_window = sent_avail[-self.lookback :]
            else:
                sent_window = sent_avail
            sent_arr = sent_df.loc[sent_window].values.astype(np.float32)
            if sent_arr.shape[0] < self.lookback:
                pad = np.zeros((self.lookback - sent_arr.shape[0], sent_arr.shape[1]), dtype=np.float32)
                sent_arr = np.concatenate([pad, sent_arr], axis=0)
        else:
            sent_arr = np.zeros((self.lookback, 4), dtype=np.float32)
        sent_arr = np.nan_to_num(sent_arr, nan=0.0)

        # --- graph node features ---
        graph_feat = self.graph_node_features.astype(np.float32)

        # --- margin features (7,) ---
        margin_df = self.margin_features.get(ticker)
        if margin_df is not None and len(margin_df) > 0:
            margin_avail = margin_df.index[margin_df.index <= w_date]
            if len(margin_avail) > 0:
                margin_arr = margin_df.loc[margin_avail[-1]].values.astype(np.float32)
            else:
                margin_arr = np.zeros(7, dtype=np.float32)
        else:
            margin_arr = np.zeros(7, dtype=np.float32)
        margin_arr = np.nan_to_num(margin_arr, nan=0.0)

        # --- label ---
        label = float(self.labels.loc[w_date, ticker])

        return (
            torch.from_numpy(mkt_arr),       # (20, 47)
            torch.from_numpy(sent_arr),       # (20, 4)
            torch.from_numpy(graph_feat),     # (n_stocks, node_feat_dim)
            torch.from_numpy(margin_arr),     # (7,)
            torch.tensor(label, dtype=torch.float32),
        )


# ---------------------------------------------------------------------------
# Temporal splits
# ---------------------------------------------------------------------------

@dataclass
class TemporalSplit:
    """Container for a single temporal split (train / val / test window)."""
    name: str
    start: pd.Timestamp
    end: pd.Timestamp


def create_temporal_splits(
    dataset: CrashMarginDataset,
    walk_forward_months: int = 6,
) -> Dict[str, List[int]]:
    """Partition dataset indices into temporal splits.

    Split boundaries (per paper):
      - Train : 2018-01-01 to 2020-12-31
      - Val   : 2021-01-01 to 2021-12-31
      - Test  : 2022-01-01 to 2024-12-31

    For the test period, indices are further grouped into 6-month windows
    to support walk-forward retraining.

    Parameters
    ----------
    dataset : CrashMarginDataset
    walk_forward_months : int
        Retraining cadence during the test period (default 6).

    Returns
    -------
    dict
        Keys: ``"train"``, ``"val"``, ``"test"``, plus
        ``"test_wf_0"``, ``"test_wf_1"``, ... for walk-forward windows.
        Values: lists of integer indices into *dataset*.
    """
    train_end = pd.Timestamp("2020-12-31")
    val_end = pd.Timestamp("2021-12-31")
    test_end = pd.Timestamp("2024-12-31")

    splits: Dict[str, List[int]] = {"train": [], "val": [], "test": []}

    for i, sample in enumerate(dataset._samples):
        d = sample.date
        if d <= train_end:
            splits["train"].append(i)
        elif d <= val_end:
            splits["val"].append(i)
        elif d <= test_end:
            splits["test"].append(i)

    # Walk-forward sub-windows within test
    if splits["test"]:
        test_dates = [dataset._samples[i].date for i in splits["test"]]
        min_date = min(test_dates)
        window_start = min_date
        wf_idx = 0
        while window_start <= test_end:
            window_end = window_start + pd.DateOffset(months=walk_forward_months)
            key = f"test_wf_{wf_idx}"
            splits[key] = [
                i for i in splits["test"]
                if window_start <= dataset._samples[i].date < window_end
            ]
            window_start = window_end
            wf_idx += 1

    return splits
