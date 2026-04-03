"""Industry Contagion & Margin Exposure Graphs (Section 3.2.3).

Builds two dynamic adjacency matrices per time step:
  1. Industry correlation graph -- edges from 20-day rolling return
     correlations, pruned below the 70th percentile, updated weekly.
  2. Margin exposure graph -- edges weighted by shared high-LTV margin
     portfolio co-occurrence.

Node features are one-hot sector encoding concatenated with market-cap
quintile indicators.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Node feature utilities
# ---------------------------------------------------------------------------

def _one_hot_sector(tickers: List[str], sector_map: Dict[str, str]) -> np.ndarray:
    """Create one-hot sector vectors for each ticker.

    Returns shape ``(n_stocks, n_sectors)``.
    """
    sectors = [sector_map.get(t, "Unknown") for t in tickers]
    unique = sorted(set(sectors))
    idx_map = {s: i for i, s in enumerate(unique)}
    out = np.zeros((len(tickers), len(unique)), dtype=np.float32)
    for i, s in enumerate(sectors):
        out[i, idx_map[s]] = 1.0
    return out


def _market_cap_quintile(market_caps: pd.Series) -> np.ndarray:
    """One-hot encode market-cap quintile (5 bins).

    Parameters
    ----------
    market_caps : Series
        Index = ticker, value = market cap (most recent).

    Returns
    -------
    ndarray of shape ``(n_stocks, 5)``
    """
    labels = pd.qcut(market_caps, q=5, labels=False, duplicates="drop")
    n = len(market_caps)
    n_bins = int(labels.max()) + 1
    out = np.zeros((n, 5), dtype=np.float32)
    for i, q in enumerate(labels.values):
        if not np.isnan(q):
            out[i, int(q)] = 1.0
    return out


def build_node_features(
    tickers: List[str],
    sector_map: Dict[str, str],
    market_caps: pd.Series,
) -> np.ndarray:
    """Concatenate one-hot sector + market-cap quintile.

    Returns shape ``(n_stocks, n_sectors + 5)``.
    """
    sec = _one_hot_sector(tickers, sector_map)
    cap = _market_cap_quintile(market_caps.reindex(tickers))
    return np.concatenate([sec, cap], axis=1)


# ---------------------------------------------------------------------------
# Industry correlation graph
# ---------------------------------------------------------------------------

@dataclass
class IndustryGraphBuilder:
    """Dynamic graph G_t = (V, E_t) based on rolling return correlations.

    Parameters
    ----------
    corr_window : int
        Number of trading days for the rolling correlation window (default 20).
    prune_percentile : float
        Edges with absolute correlation below this percentile of all pairwise
        values are zeroed out (default 0.70 = 70th percentile).
    update_freq : str
        Frequency at which the graph is recomputed (default ``'W'`` = weekly).
    """

    corr_window: int = 20
    prune_percentile: float = 0.70
    update_freq: str = "W"

    def build(
        self,
        returns: pd.DataFrame,
        as_of: Optional[pd.Timestamp] = None,
    ) -> np.ndarray:
        """Compute a single adjacency matrix.

        Parameters
        ----------
        returns : DataFrame
            Daily returns with DatetimeIndex, one column per stock.
        as_of : Timestamp, optional
            Build the graph using the *corr_window* days ending on this date.
            Defaults to the last available date.

        Returns
        -------
        ndarray of shape ``(n_stocks, n_stocks)``
        """
        if as_of is None:
            as_of = returns.index[-1]

        window = returns.loc[:as_of].iloc[-self.corr_window :]
        corr = window.corr().values.astype(np.float32)
        np.fill_diagonal(corr, 0.0)

        # Prune below percentile
        abs_corr = np.abs(corr)
        threshold = np.nanpercentile(abs_corr[abs_corr > 0], self.prune_percentile * 100)
        corr[abs_corr < threshold] = 0.0

        return corr

    def build_temporal(
        self, returns: pd.DataFrame
    ) -> Dict[pd.Timestamp, np.ndarray]:
        """Build adjacency matrices for every rebalance date.

        Returns
        -------
        dict
            ``{rebalance_date: adj_matrix}`` for each week-end in *returns*.
        """
        rebalance_dates = (
            returns.resample(self.update_freq).last().dropna(how="all").index
        )
        out: Dict[pd.Timestamp, np.ndarray] = {}
        for dt in rebalance_dates:
            if (returns.index <= dt).sum() >= self.corr_window:
                out[dt] = self.build(returns, as_of=dt)
        return out


# ---------------------------------------------------------------------------
# Margin exposure graph
# ---------------------------------------------------------------------------

@dataclass
class MarginExposureGraphBuilder:
    """Graph where edge weights reflect shared margin portfolio exposure.

    Two stocks that frequently co-appear in high-LTV margin portfolios receive
    a stronger edge -- capturing forced-selling contagion risk.

    Parameters
    ----------
    ltv_threshold : float
        Loan-to-value ratio above which a portfolio is considered "high-LTV"
        (default 0.60 = 60%).
    min_co_occurrence : int
        Minimum number of shared high-LTV portfolios for an edge to exist.
    """

    ltv_threshold: float = 0.60
    min_co_occurrence: int = 2

    def build(self, margin_data: pd.DataFrame) -> np.ndarray:
        """Build adjacency matrix from margin portfolio data.

        Parameters
        ----------
        margin_data : DataFrame
            Expected columns:
              - ``portfolio_id`` : int/str identifier for a margin account
              - ``ticker``       : stock held in the portfolio
              - ``ltv``          : loan-to-value ratio of the portfolio
              - ``weight``       : portfolio weight of this stock (optional)

        Returns
        -------
        ndarray of shape ``(n_stocks, n_stocks)``
            Symmetric, non-negative edge weights.
        """
        df = margin_data.copy()

        # Keep only high-LTV portfolios
        high_ltv = df[df["ltv"] >= self.ltv_threshold]

        tickers = sorted(df["ticker"].unique())
        n = len(tickers)
        tick_idx = {t: i for i, t in enumerate(tickers)}

        adj = np.zeros((n, n), dtype=np.float32)

        for _, group in high_ltv.groupby("portfolio_id"):
            stocks_in_port = group["ticker"].unique()
            if "weight" in group.columns:
                weights = group.set_index("ticker")["weight"]
            else:
                weights = pd.Series(
                    1.0 / len(stocks_in_port), index=stocks_in_port
                )
            for i_idx, s1 in enumerate(stocks_in_port):
                for s2 in stocks_in_port[i_idx + 1 :]:
                    if s1 in tick_idx and s2 in tick_idx:
                        w = float(weights.get(s1, 0) * weights.get(s2, 0))
                        adj[tick_idx[s1], tick_idx[s2]] += w
                        adj[tick_idx[s2], tick_idx[s1]] += w

        # Prune weak edges
        adj[adj < self.min_co_occurrence * (1.0 / (n * n + 1))] = 0.0

        # Row-normalise for GNN consumption
        row_sum = adj.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        adj = adj / row_sum

        return adj


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_dual_graph(
    returns: pd.DataFrame,
    margin_data: pd.DataFrame,
    sector_info: Dict[str, str],
    market_caps: Optional[pd.Series] = None,
    as_of: Optional[pd.Timestamp] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build both adjacency matrices for a given time step.

    Parameters
    ----------
    returns : DataFrame
        Daily returns, columns = tickers, DatetimeIndex.
    margin_data : DataFrame
        Margin portfolio-level data (see :class:`MarginExposureGraphBuilder`).
    sector_info : dict
        ``{ticker: sector_name}`` mapping.
    market_caps : Series, optional
        Market capitalisation per ticker (for node features -- not used in
        adjacency construction but stored for downstream access).
    as_of : Timestamp, optional
        Date for the industry correlation snapshot.

    Returns
    -------
    (adj_industry, adj_margin)
        Two ``(n_stocks, n_stocks)`` float32 numpy arrays.
    """
    adj_industry = IndustryGraphBuilder().build(returns, as_of=as_of)
    adj_margin = MarginExposureGraphBuilder().build(margin_data)
    return adj_industry, adj_margin
