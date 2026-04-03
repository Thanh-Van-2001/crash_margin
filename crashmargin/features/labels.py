"""Crash Label Definition (Section 3.1).

Implements the firm-specific crash event indicator:

  1. Regress each stock's daily returns on market + industry factor returns
     to obtain firm-specific weekly residuals W_{i,t}.
  2. A crash event C_{i,t} = 1 when W_{i,t} < mu_i - 3.09 * sigma_i
     (i.e., below the 0.1th percentile of the normal distribution).

The paper reports a 4.4% crash rate (7,014 / 158,175 stock-weeks).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd


@dataclass
class CrashLabeler:
    """Compute binary crash labels from daily stock returns.

    Parameters
    ----------
    threshold_z : float
        Number of standard deviations below the mean for a crash.
        Default 3.09 (0.1th percentile of the normal distribution).
    min_history_weeks : int
        Minimum number of weekly observations before labels are produced
        (to estimate mu and sigma reliably).  Default 26 (~6 months).
    """

    threshold_z: float = 3.09
    min_history_weeks: int = 26

    # ------------------------------------------------------------------ #
    # Internal: factor regression
    # ------------------------------------------------------------------ #

    @staticmethod
    def _resample_weekly_returns(daily: pd.Series) -> pd.Series:
        """Compound daily returns into weekly (Friday-ending) returns."""
        log_ret = np.log1p(daily)
        weekly_log = log_ret.resample("W-FRI").sum()
        return np.expm1(weekly_log)

    @staticmethod
    def _regress_out_factors(
        stock_ret: pd.Series,
        market_ret: pd.Series,
        industry_ret: pd.Series,
    ) -> pd.Series:
        """OLS: r_i = alpha + beta_m * r_m + beta_ind * r_ind + epsilon.

        Returns the residual series (firm-specific return W_{i,t}).
        Uses expanding-window regression -- each week uses all data up to that
        point so there is no lookahead.
        """
        # Align
        df = pd.DataFrame({
            "stock": stock_ret,
            "market": market_ret,
            "industry": industry_ret,
        }).dropna()

        if len(df) < 10:
            return pd.Series(np.nan, index=stock_ret.index, name="residual")

        residuals = pd.Series(np.nan, index=df.index, dtype=np.float64)

        # Expanding-window OLS (causal)
        y = df["stock"].values
        X = np.column_stack([np.ones(len(df)), df["market"].values, df["industry"].values])

        for t in range(10, len(df)):
            X_t = X[: t + 1]
            y_t = y[: t + 1]
            try:
                beta = np.linalg.lstsq(X_t, y_t, rcond=None)[0]
                residuals.iloc[t] = y[t] - X[t] @ beta
            except np.linalg.LinAlgError:
                residuals.iloc[t] = np.nan

        return residuals

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def compute_crash_labels(
        self,
        returns_df: pd.DataFrame,
        market_returns: Optional[pd.Series] = None,
        industry_returns: Optional[Dict[str, pd.Series]] = None,
        sector_map: Optional[Dict[str, str]] = None,
    ) -> pd.DataFrame:
        """Produce binary crash labels for every (stock, week).

        Parameters
        ----------
        returns_df : DataFrame
            Daily simple returns.  Columns = tickers, DatetimeIndex.
        market_returns : Series, optional
            Market-wide daily returns (e.g., VN-Index).
            If not provided, the equal-weighted mean of *returns_df* is used.
        industry_returns : dict, optional
            ``{sector_name: daily_return_series}``.  If not provided, the
            equal-weighted mean of same-sector stocks is used as a proxy
            (requires *sector_map*).
        sector_map : dict, optional
            ``{ticker: sector_name}``.  Required when *industry_returns* is
            ``None``.

        Returns
        -------
        DataFrame
            Columns = tickers, index = weekly dates, values in {0, 1, NaN}.
            ``1`` indicates a crash event.
        """
        # Defaults
        if market_returns is None:
            market_returns = returns_df.mean(axis=1)

        if industry_returns is None and sector_map is not None:
            industry_returns = self._build_industry_returns(returns_df, sector_map)
        elif industry_returns is None:
            # Fall back: use market return as industry proxy
            industry_returns = {
                "__market__": market_returns,
            }
            sector_map = {t: "__market__" for t in returns_df.columns}

        # Weekly market / industry returns
        mkt_w = self._resample_weekly_returns(market_returns)

        ind_w: Dict[str, pd.Series] = {}
        for sec, ser in industry_returns.items():
            ind_w[sec] = self._resample_weekly_returns(ser)

        labels = pd.DataFrame(index=mkt_w.index, columns=returns_df.columns, dtype=np.float64)

        for ticker in returns_df.columns:
            daily_ret = returns_df[ticker].dropna()
            weekly_ret = self._resample_weekly_returns(daily_ret)

            sec = sector_map.get(ticker, "__market__") if sector_map else "__market__"
            ind_series = ind_w.get(sec, mkt_w)

            residuals = self._regress_out_factors(weekly_ret, mkt_w, ind_series)

            # Expanding mean and std (causal)
            mu = residuals.expanding(min_periods=self.min_history_weeks).mean()
            sigma = residuals.expanding(min_periods=self.min_history_weeks).std()

            crash_threshold = mu - self.threshold_z * sigma
            crash = (residuals < crash_threshold).astype(float)
            crash[sigma.isna()] = np.nan

            labels[ticker] = crash.reindex(labels.index)

        return labels

    # ------------------------------------------------------------------ #
    def _build_industry_returns(
        self,
        returns_df: pd.DataFrame,
        sector_map: Dict[str, str],
    ) -> Dict[str, pd.Series]:
        """Equal-weighted industry return series from stock returns."""
        sectors: Dict[str, list] = {}
        for ticker, sector in sector_map.items():
            if ticker in returns_df.columns:
                sectors.setdefault(sector, []).append(ticker)

        out: Dict[str, pd.Series] = {}
        for sector, tickers in sectors.items():
            out[sector] = returns_df[tickers].mean(axis=1)
        return out

    # ------------------------------------------------------------------ #
    def label_stats(self, labels: pd.DataFrame) -> Dict[str, float]:
        """Summary statistics for crash labels.

        Returns dict with keys: ``n_total, n_crash, crash_rate``.
        """
        flat = labels.values.flatten()
        valid = flat[~np.isnan(flat)]
        n_total = len(valid)
        n_crash = int(valid.sum())
        return {
            "n_total": n_total,
            "n_crash": n_crash,
            "crash_rate": n_crash / n_total if n_total > 0 else 0.0,
        }
