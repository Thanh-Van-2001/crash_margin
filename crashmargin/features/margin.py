"""Margin Lending Features (Section 3.2.4).

Extracts 7 margin-specific risk indicators per stock-day.  All features
are computed causally (no future information).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    return a / b.replace(0, np.nan)


# ---------------------------------------------------------------------------
# Individual feature computations
# ---------------------------------------------------------------------------

def _margin_debt_to_mcap(
    margin_debt: pd.Series, market_cap: pd.Series
) -> pd.Series:
    """Feature 1: margin_debt / market_cap."""
    return _safe_div(margin_debt, market_cap)


def _margin_debt_growth(margin_debt: pd.Series, window: int) -> pd.Series:
    """Percentage growth of margin debt over *window* days."""
    lagged = margin_debt.shift(window)
    return _safe_div(margin_debt - lagged, lagged)


def _ltv_concentration_index(
    ltv_weights: pd.DataFrame, top_k: int = 10
) -> pd.Series:
    """Feature 4: share of total margin debt held by top-*k* stocks.

    Parameters
    ----------
    ltv_weights : DataFrame
        Columns = tickers, rows = dates, values = margin-debt weight of each
        stock in the aggregate margin portfolio.

    Returns
    -------
    Series
        One value per date -- the sum of the *top_k* largest weights.
    """
    def _top_share(row: pd.Series) -> float:
        sorted_vals = row.dropna().sort_values(ascending=False)
        return float(sorted_vals.iloc[:top_k].sum()) if len(sorted_vals) > 0 else np.nan

    return ltv_weights.apply(_top_share, axis=1)


def _margin_call_frequency(
    margin_calls: pd.Series, window: int = 20
) -> pd.Series:
    """Feature 5: rolling count of margin call events in *window* days.

    *margin_calls* should be 1 on days a margin call occurred, 0 otherwise.
    """
    return margin_calls.rolling(window, min_periods=1).sum()


def _sector_margin_exposure(
    margin_debt_by_sector: pd.DataFrame, sector: str
) -> pd.Series:
    """Feature 6: sector's share of total margin debt.

    Parameters
    ----------
    margin_debt_by_sector : DataFrame
        Columns = sector names, rows = dates, values = sector margin debt.
    sector : str
        Sector for which to compute the ratio.
    """
    total = margin_debt_by_sector.sum(axis=1)
    if sector in margin_debt_by_sector.columns:
        return _safe_div(margin_debt_by_sector[sector], total)
    return pd.Series(np.nan, index=margin_debt_by_sector.index)


def _distance_to_maintenance(
    ltv: pd.Series, maintenance_ratio: float = 0.80
) -> pd.Series:
    """Feature 7: distance to maintenance margin = maintenance_ratio - LTV.

    Smaller (or negative) values indicate higher forced-liquidation risk.
    """
    return maintenance_ratio - ltv


# ---------------------------------------------------------------------------
# Extractor class
# ---------------------------------------------------------------------------

@dataclass
class MarginFeatureExtractor:
    """Compute 7 margin lending features for a single stock.

    The input ``margin_df`` should be a stock-level time series with at least::

        date, margin_debt, ltv, margin_call_flag

    ``market_cap_df`` provides the stock's daily market capitalisation.

    Optional context DataFrames (for cross-stock features) can be supplied
    via :meth:`set_context`.

    Output features:
      1. ``margin_debt_mcap``       -- margin debt / market cap
      2. ``margin_debt_growth_5``   -- 5-day margin debt growth
      3. ``margin_debt_growth_20``  -- 20-day margin debt growth
      4. ``ltv_concentration``      -- top-10 stock LTV concentration
      5. ``margin_call_freq_20``    -- rolling 20-day margin call count
      6. ``sector_margin_exposure`` -- sector's share of total margin debt
      7. ``dist_to_maintenance``    -- distance to maintenance margin
    """

    maintenance_ratio: float = 0.80
    top_k: int = 10
    feature_names_: List[str] = field(default_factory=list, init=False)

    # Optional cross-stock context (set before calling transform)
    _ltv_weights: pd.DataFrame | None = field(default=None, init=False, repr=False)
    _sector_margin: pd.DataFrame | None = field(default=None, init=False, repr=False)
    _sector_name: str | None = field(default=None, init=False, repr=False)

    def set_context(
        self,
        ltv_weights: pd.DataFrame | None = None,
        sector_margin_debt: pd.DataFrame | None = None,
        sector_name: str | None = None,
    ) -> "MarginFeatureExtractor":
        """Supply cross-stock DataFrames needed for features 4 and 6.

        Parameters
        ----------
        ltv_weights : DataFrame
            Columns = tickers, rows = dates, values = per-stock margin debt
            weight in the aggregate portfolio.
        sector_margin_debt : DataFrame
            Columns = sectors, rows = dates, values = sector-level margin debt.
        sector_name : str
            The sector of the stock being processed (for feature 6).
        """
        self._ltv_weights = ltv_weights
        self._sector_margin = sector_margin_debt
        self._sector_name = sector_name
        return self

    def transform(
        self,
        margin_df: pd.DataFrame,
        market_cap_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute 7 margin features aligned to *margin_df*'s index.

        Parameters
        ----------
        margin_df : DataFrame
            Stock-level margin data with columns:
            ``date, margin_debt, ltv, margin_call_flag``.
        market_cap_df : DataFrame
            Must contain a ``market_cap`` column aligned to the same dates.

        Returns
        -------
        DataFrame with 7 columns.
        """
        idx = margin_df.index
        debt = margin_df["margin_debt"]
        ltv = margin_df["ltv"]
        mcall = margin_df.get("margin_call_flag", pd.Series(0, index=idx))
        mcap = market_cap_df["market_cap"] if "market_cap" in market_cap_df.columns else market_cap_df.iloc[:, 0]

        feats: dict[str, pd.Series] = {}

        # 1. Margin debt / market cap
        feats["margin_debt_mcap"] = _margin_debt_to_mcap(debt, mcap)

        # 2-3. Margin debt growth
        feats["margin_debt_growth_5"] = _margin_debt_growth(debt, 5)
        feats["margin_debt_growth_20"] = _margin_debt_growth(debt, 20)

        # 4. LTV concentration index (cross-stock)
        if self._ltv_weights is not None:
            conc = _ltv_concentration_index(self._ltv_weights, self.top_k)
            feats["ltv_concentration"] = conc.reindex(idx)
        else:
            feats["ltv_concentration"] = pd.Series(np.nan, index=idx)

        # 5. Rolling margin call frequency
        feats["margin_call_freq_20"] = _margin_call_frequency(mcall, 20)

        # 6. Sector margin exposure
        if self._sector_margin is not None and self._sector_name is not None:
            sec_exp = _sector_margin_exposure(self._sector_margin, self._sector_name)
            feats["sector_margin_exposure"] = sec_exp.reindex(idx)
        else:
            feats["sector_margin_exposure"] = pd.Series(np.nan, index=idx)

        # 7. Distance to maintenance margin
        feats["dist_to_maintenance"] = _distance_to_maintenance(ltv, self.maintenance_ratio)

        result = pd.DataFrame(feats, index=idx)
        self.feature_names_ = list(result.columns)
        return result


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def compute_margin_features(
    margin_df: pd.DataFrame,
    market_cap_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute 7 margin lending features.

    Parameters
    ----------
    margin_df : DataFrame
        Stock-level time series with columns:
        ``date, margin_debt, ltv, margin_call_flag``.
    market_cap_df : DataFrame
        Aligned market capitalisation (column ``market_cap``).

    Returns
    -------
    DataFrame with 7 feature columns.
    """
    extractor = MarginFeatureExtractor()
    return extractor.transform(margin_df, market_cap_df)
