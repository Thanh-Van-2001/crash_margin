"""Synthetic Vietnamese Market Data Generator.

Produces realistic proxy data for ~95 HOSE/HNX-listed stocks over 2018--2024,
including:
  - Daily OHLCV with sector-correlated returns
  - Crash events at ~4.4% weekly rate, clustered in 2020 (COVID) and 2022
  - Margin lending statistics (debt, LTV, margin calls)
  - News sentiment (headline counts + polarity)
  - Sector composition matching paper Figure 2c

All data is synthetic and intended for development / unit-testing only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Vietnamese sector composition (Figure 2c of the paper)
# ---------------------------------------------------------------------------

_SECTOR_ALLOCATION: Dict[str, float] = {
    "Banking": 0.19,
    "Real Estate": 0.11,
    "Materials": 0.17,
    "Industrials": 0.10,
    "Consumer Staples": 0.08,
    "Consumer Discretionary": 0.07,
    "Technology": 0.06,
    "Energy": 0.05,
    "Utilities": 0.05,
    "Healthcare": 0.04,
    "Insurance": 0.04,
    "Securities": 0.04,
}

_EXCHANGE_SPLIT = {"HOSE": 0.65, "HNX": 0.35}

# Crisis periods with elevated crash probability
_CRISIS_WINDOWS: List[Tuple[str, str, float]] = [
    # (start, end, crash_multiplier vs baseline)
    ("2020-02-01", "2020-06-30", 4.0),   # COVID-19
    ("2022-09-01", "2023-03-31", 3.5),   # VN real-estate / bond correction
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _generate_tickers(n: int, rng: np.random.Generator) -> List[Dict]:
    """Create synthetic ticker metadata."""
    tickers = []
    sectors = list(_SECTOR_ALLOCATION.keys())
    weights = np.array(list(_SECTOR_ALLOCATION.values()))
    # Normalise in case weights don't sum to 1
    weights = weights / weights.sum()

    sector_counts = np.round(weights * n).astype(int)
    # Adjust rounding so total == n
    diff = n - sector_counts.sum()
    for i in range(abs(diff)):
        idx = i % len(sector_counts)
        sector_counts[idx] += np.sign(diff)

    exchanges = rng.choice(
        list(_EXCHANGE_SPLIT.keys()),
        size=n,
        p=list(_EXCHANGE_SPLIT.values()),
    )

    idx = 0
    for sec_i, sector in enumerate(sectors):
        for j in range(sector_counts[sec_i]):
            code = f"VN{idx:03d}"
            tickers.append({
                "ticker": code,
                "sector": sector,
                "exchange": exchanges[idx],
                "market_cap_bn": float(rng.lognormal(mean=8, sigma=1.2)),  # VND billions
            })
            idx += 1

    return tickers[:n]


def _trading_calendar(start: str, end: str) -> pd.DatetimeIndex:
    """Vietnamese market calendar approximation (weekdays, no holidays)."""
    return pd.bdate_range(start, end, freq="B")


def _correlated_returns(
    n_stocks: int,
    n_days: int,
    sector_ids: np.ndarray,
    rng: np.random.Generator,
    base_vol: float = 0.02,
) -> np.ndarray:
    """Generate sector-correlated daily returns.

    Returns shape ``(n_days, n_stocks)``.
    """
    n_sectors = int(sector_ids.max()) + 1

    # Sector factor returns
    sector_factor = rng.normal(0, base_vol * 0.6, size=(n_days, n_sectors))

    # Market factor
    market_factor = rng.normal(0, base_vol * 0.4, size=(n_days, 1))

    # Idiosyncratic
    idio = rng.normal(0, base_vol * 0.7, size=(n_days, n_stocks))

    # Combine
    returns = np.zeros((n_days, n_stocks))
    for i in range(n_stocks):
        sec = sector_ids[i]
        returns[:, i] = (
            0.3 * market_factor[:, 0]
            + 0.3 * sector_factor[:, sec]
            + 0.4 * idio[:, i]
        )

    return returns


def _inject_crashes(
    returns: np.ndarray,
    dates: pd.DatetimeIndex,
    target_crash_rate: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Inject crash-like drawdowns to achieve the target weekly crash rate.

    Returns
    -------
    returns : modified return array
    crash_weeks : bool array (n_weeks, n_stocks)
    """
    n_days, n_stocks = returns.shape

    # Compute weekly boundaries
    week_ends = pd.Series(dates).dt.to_period("W").drop_duplicates().index
    n_weeks = len(dates.to_series().resample("W-FRI")) + 1

    # Base crash probability per stock-week
    base_p = target_crash_rate * 0.5  # will be boosted in crisis windows

    for start_s, end_s, mult in _CRISIS_WINDOWS:
        start_dt = pd.Timestamp(start_s)
        end_dt = pd.Timestamp(end_s)
        mask = (dates >= start_dt) & (dates <= end_dt)
        day_indices = np.where(mask)[0]

        for d in day_indices:
            # Each stock has a chance of a large negative shock
            crash_mask = rng.random(n_stocks) < (base_p * mult / 5)
            shock = rng.uniform(-0.06, -0.03, size=n_stocks)
            returns[d] += crash_mask * shock

    # Also sprinkle idiosyncratic crashes outside crisis windows
    for d in range(n_days):
        dt = dates[d]
        in_crisis = any(
            pd.Timestamp(s) <= dt <= pd.Timestamp(e)
            for s, e, _ in _CRISIS_WINDOWS
        )
        p = base_p * 0.3 if not in_crisis else 0.0
        crash_mask = rng.random(n_stocks) < (p / 5)
        shock = rng.uniform(-0.05, -0.02, size=n_stocks)
        returns[d] += crash_mask * shock

    return returns


def _returns_to_ohlcv(
    returns: np.ndarray,
    dates: pd.DatetimeIndex,
    tickers: List[str],
    rng: np.random.Generator,
    initial_price: float = 25_000.0,
) -> Dict[str, pd.DataFrame]:
    """Convert return matrix to per-stock OHLCV DataFrames."""
    n_days, n_stocks = returns.shape
    result: Dict[str, pd.DataFrame] = {}

    for i, ticker in enumerate(tickers):
        close = np.zeros(n_days)
        close[0] = initial_price * rng.uniform(0.4, 2.5)
        for d in range(1, n_days):
            close[d] = close[d - 1] * (1 + returns[d, i])
            close[d] = max(close[d], 100)  # floor price

        high = close * (1 + np.abs(rng.normal(0, 0.008, n_days)))
        low = close * (1 - np.abs(rng.normal(0, 0.008, n_days)))
        open_ = low + rng.random(n_days) * (high - low)

        # Volume: log-normal with some autocorrelation
        base_vol = rng.lognormal(mean=13, sigma=0.8)
        vol_noise = rng.lognormal(mean=0, sigma=0.5, size=n_days)
        volume = base_vol * vol_noise

        # Foreign flow
        foreign_pct = rng.uniform(0.05, 0.25)
        fb = volume * foreign_pct * rng.uniform(0.3, 0.7, n_days)
        fs = volume * foreign_pct * rng.uniform(0.3, 0.7, n_days)

        result[ticker] = pd.DataFrame({
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "foreign_buy_volume": fb,
            "foreign_sell_volume": fs,
        }, index=dates)

    return result


# ---------------------------------------------------------------------------
# Margin data generator
# ---------------------------------------------------------------------------

def _generate_margin_data(
    tickers: List[str],
    dates: pd.DatetimeIndex,
    sectors: List[str],
    rng: np.random.Generator,
) -> Dict[str, pd.DataFrame]:
    """Generate per-stock margin lending time series.

    Returns a dict with keys:
      - ``"stock_margin"`` : DataFrame with (date, ticker, margin_debt, ltv,
        margin_call_flag)
      - ``"sector_margin"`` : DataFrame (dates x sectors) of sector margin debt
      - ``"ltv_weights"`` : DataFrame (dates x tickers) of aggregate weights
      - ``"portfolio_margin"`` : DataFrame for graph construction
    """
    n_days = len(dates)
    n_stocks = len(tickers)

    # Per-stock margin debt (random walk with drift)
    stock_rows = []
    all_debts = np.zeros((n_days, n_stocks))

    for i, ticker in enumerate(tickers):
        base_debt = rng.lognormal(mean=10, sigma=1.0)
        debt = np.zeros(n_days)
        debt[0] = base_debt
        for d in range(1, n_days):
            debt[d] = debt[d - 1] * (1 + rng.normal(0.0002, 0.015))
            debt[d] = max(debt[d], 0)
        all_debts[:, i] = debt

        # LTV: hovering around 0.45-0.65 with occasional spikes
        ltv = 0.50 + rng.normal(0, 0.08, n_days).cumsum() * 0.001
        ltv = np.clip(ltv, 0.20, 0.85)

        # Margin calls when LTV > 0.70
        margin_calls = (ltv > 0.70).astype(float)

        for d in range(n_days):
            stock_rows.append({
                "date": dates[d],
                "ticker": ticker,
                "margin_debt": debt[d],
                "ltv": ltv[d],
                "margin_call_flag": margin_calls[d],
            })

    stock_margin = pd.DataFrame(stock_rows)

    # Sector margin debt
    unique_sectors = sorted(set(sectors))
    sector_debt = pd.DataFrame(index=dates, columns=unique_sectors, dtype=np.float64)
    for sec in unique_sectors:
        sec_tickers = [t for t, s in zip(tickers, sectors) if s == sec]
        sec_idxs = [tickers.index(t) for t in sec_tickers]
        if sec_idxs:
            sector_debt[sec] = all_debts[:, sec_idxs].sum(axis=1)
        else:
            sector_debt[sec] = 0.0

    # LTV weights (each stock's share of total margin debt)
    total_debt = all_debts.sum(axis=1, keepdims=True)
    total_debt[total_debt == 0] = 1.0
    ltv_weights = pd.DataFrame(
        all_debts / total_debt,
        index=dates,
        columns=tickers,
    )

    # Portfolio-level margin data (for graph builder)
    n_portfolios = max(50, n_stocks * 2)
    port_rows = []
    for pid in range(n_portfolios):
        n_holdings = rng.integers(2, min(10, n_stocks))
        held = rng.choice(tickers, size=n_holdings, replace=False)
        port_ltv = rng.uniform(0.30, 0.85)
        weights = rng.dirichlet(np.ones(n_holdings))
        for h, w in zip(held, weights):
            port_rows.append({
                "portfolio_id": pid,
                "ticker": h,
                "ltv": port_ltv,
                "weight": w,
            })
    portfolio_margin = pd.DataFrame(port_rows)

    return {
        "stock_margin": stock_margin,
        "sector_margin": sector_debt,
        "ltv_weights": ltv_weights,
        "portfolio_margin": portfolio_margin,
    }


# ---------------------------------------------------------------------------
# News / sentiment data generator
# ---------------------------------------------------------------------------

def _generate_news_data(
    tickers: List[str],
    dates: pd.DatetimeIndex,
    rng: np.random.Generator,
    avg_articles_per_stock_day: float = 0.8,
) -> pd.DataFrame:
    """Generate synthetic news sentiment data.

    Returns DataFrame with columns: ``date, ticker, headline, polarity``.
    """
    rows = []
    for d in dates:
        for ticker in tickers:
            n_articles = rng.poisson(avg_articles_per_stock_day)
            for _ in range(n_articles):
                # Polarity centred around 0 with occasional extremes
                polarity = float(np.clip(rng.normal(0, 0.35), -1, 1))
                rows.append({
                    "date": d,
                    "ticker": ticker,
                    "headline": f"Synthetic headline for {ticker}",
                    "polarity": polarity,
                })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main generator class
# ---------------------------------------------------------------------------

@dataclass
class SyntheticVNData:
    """Generate a complete synthetic Vietnamese stock market dataset.

    Matches the statistical properties described in the CrashMargin paper:
      - 95 stocks across HOSE and HNX
      - 2018--2024 daily data
      - ~4.4% weekly crash rate
      - Sector composition per Figure 2c

    Usage::

        gen = SyntheticVNData()
        data = gen.generate(n_stocks=95, seed=42)

        ohlcv = data["ohlcv"]          # dict[ticker -> DataFrame]
        margin = data["margin"]         # dict of DataFrames
        news = data["news"]             # DataFrame
        meta = data["metadata"]         # list of dicts
        sector_map = data["sector_map"] # {ticker: sector}
    """

    start_date: str = "2018-01-02"
    end_date: str = "2024-12-31"
    target_crash_rate: float = 0.044  # 4.4%

    def generate(
        self,
        n_stocks: int = 95,
        seed: int = 42,
    ) -> Dict[str, object]:
        """Generate the full dataset.

        Parameters
        ----------
        n_stocks : int
            Number of synthetic stocks (default 95).
        seed : int
            Random seed for reproducibility.

        Returns
        -------
        dict with keys:
          - ``"ohlcv"``       : dict[str, DataFrame] -- per-stock OHLCV
          - ``"returns"``     : DataFrame (dates x tickers) daily simple returns
          - ``"margin"``      : dict of margin DataFrames
          - ``"news"``        : DataFrame of news sentiment
          - ``"metadata"``    : list of ticker metadata dicts
          - ``"sector_map"``  : {ticker: sector}
          - ``"market_caps"`` : Series (ticker -> market cap in VND bn)
          - ``"dates"``       : DatetimeIndex
        """
        rng = np.random.default_rng(seed)

        # --- Metadata ---
        meta = _generate_tickers(n_stocks, rng)
        tickers = [m["ticker"] for m in meta]
        sectors = [m["sector"] for m in meta]
        sector_map = {m["ticker"]: m["sector"] for m in meta}
        market_caps = pd.Series(
            {m["ticker"]: m["market_cap_bn"] for m in meta}
        )

        # Sector integer ids for correlated returns
        unique_sectors = sorted(set(sectors))
        sec_to_id = {s: i for i, s in enumerate(unique_sectors)}
        sector_ids = np.array([sec_to_id[s] for s in sectors])

        # --- Calendar ---
        dates = _trading_calendar(self.start_date, self.end_date)
        n_days = len(dates)

        # --- Returns ---
        returns = _correlated_returns(n_stocks, n_days, sector_ids, rng)
        returns = _inject_crashes(returns, dates, self.target_crash_rate, rng)

        returns_df = pd.DataFrame(returns, index=dates, columns=tickers)

        # --- OHLCV ---
        ohlcv = _returns_to_ohlcv(returns, dates, tickers, rng)

        # --- Margin ---
        margin = _generate_margin_data(tickers, dates, sectors, rng)

        # --- News ---
        # Sample a subset of dates for news to keep data size manageable
        news_dates = dates[::3]  # every 3rd trading day
        news = _generate_news_data(tickers, news_dates, rng, avg_articles_per_stock_day=0.6)

        return {
            "ohlcv": ohlcv,
            "returns": returns_df,
            "margin": margin,
            "news": news,
            "metadata": meta,
            "sector_map": sector_map,
            "market_caps": market_caps,
            "dates": dates,
        }
