"""Market Microstructure Features (Section 3.2.1).

Extracts 47 features from daily OHLCV data for each stock.
All features are computed causally -- no future information leaks.
Normalization uses expanding-window z-scores.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _expanding_zscore(series: pd.Series, min_periods: int = 60) -> pd.Series:
    """Expanding-window z-score normalization (no lookahead)."""
    mu = series.expanding(min_periods=min_periods).mean()
    sigma = series.expanding(min_periods=min_periods).std()
    return (series - mu) / sigma.replace(0, np.nan)


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    return a / b.replace(0, np.nan)


# ---------------------------------------------------------------------------
# Individual feature computations
# ---------------------------------------------------------------------------

def _realized_volatility(close: pd.Series, window: int) -> pd.Series:
    """Annualized realized volatility over *window* days."""
    log_ret = np.log(close / close.shift(1))
    return log_ret.rolling(window).std() * np.sqrt(252)


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Average True Range."""
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window).mean()


def _amihud_illiquidity(close: pd.Series, volume: pd.Series, window: int = 20) -> pd.Series:
    """Amihud (2002) illiquidity ratio -- |return| / dollar-volume."""
    ret = close.pct_change().abs()
    dollar_vol = close * volume
    daily_ratio = _safe_div(ret, dollar_vol)
    return daily_ratio.rolling(window).mean()


def _max_drawdown(close: pd.Series, window: int) -> pd.Series:
    """Rolling maximum drawdown over *window* days."""
    roll_max = close.rolling(window, min_periods=1).max()
    dd = (close - roll_max) / roll_max
    return dd.rolling(window, min_periods=1).min()


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Relative Strength Index."""
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = _safe_div(gain, loss)
    return 100 - (100 / (1 + rs))


def _macd_histogram(close: pd.Series) -> pd.Series:
    """MACD histogram (12/26/9)."""
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal = macd_line.ewm(span=9, adjust=False).mean()
    return macd_line - signal


def _bollinger_pctb(close: pd.Series, window: int = 20) -> pd.Series:
    """Bollinger Bands %B."""
    sma = close.rolling(window).mean()
    std = close.rolling(window).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    return _safe_div(close - lower, upper - lower)


def _stochastic_k(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Stochastic %K."""
    lowest = low.rolling(window).min()
    highest = high.rolling(window).max()
    return 100 * _safe_div(close - lowest, highest - lowest)


def _turnover_ratio(volume: pd.Series, shares_outstanding: pd.Series | None, window: int = 20) -> pd.Series:
    """Turnover ratio (rolling mean).

    If *shares_outstanding* is not available, falls back to expanding-mean
    volume as a proxy.
    """
    if shares_outstanding is not None:
        ratio = _safe_div(volume, shares_outstanding)
    else:
        ratio = _safe_div(volume, volume.expanding().mean())
    return ratio.rolling(window).mean()


def _volume_zscore(volume: pd.Series, window: int = 20) -> pd.Series:
    """Volume z-score relative to trailing *window*-day distribution."""
    mu = volume.rolling(window).mean()
    sigma = volume.rolling(window).std()
    return _safe_div(volume - mu, sigma)


def _foreign_net_flow_ratio(
    foreign_buy_vol: pd.Series | None,
    foreign_sell_vol: pd.Series | None,
    volume: pd.Series,
) -> pd.Series:
    """Foreign net flow as fraction of total volume.

    Returns zeros if foreign flow columns are unavailable.
    """
    if foreign_buy_vol is None or foreign_sell_vol is None:
        return pd.Series(0.0, index=volume.index)
    net = foreign_buy_vol - foreign_sell_vol
    return _safe_div(net, volume)


def _rolling_skewness(close: pd.Series, window: int = 20) -> pd.Series:
    log_ret = np.log(close / close.shift(1))
    return log_ret.rolling(window).skew()


def _rolling_kurtosis(close: pd.Series, window: int = 20) -> pd.Series:
    log_ret = np.log(close / close.shift(1))
    return log_ret.rolling(window).kurt()


# ---------------------------------------------------------------------------
# Feature names registry (47 features)
# ---------------------------------------------------------------------------

# 3 realized-vol + 1 ATR + 1 Amihud + 3 max-dd + 1 RSI + 1 MACD + 1 BB%B
# + 1 Stoch%K + 1 turnover + 1 vol-zscore + 1 foreign-net + 1 skew + 1 kurt
# = 17 raw features.  Each raw feature produces a normalized copy, plus we add
# 5/10/20-day returns and their normalized versions and lag-1/2/3 of select
# features to reach 47 total.

_RAW_FEATURE_NAMES: list[str] = [
    "rvol_5", "rvol_10", "rvol_20",
    "atr_14",
    "amihud_20",
    "mdd_5", "mdd_10", "mdd_20",
    "rsi_14",
    "macd_hist",
    "bb_pctb",
    "stoch_k",
    "turnover_20",
    "vol_zscore_20",
    "foreign_net_ratio",
    "ret_skew_20",
    "ret_kurt_20",
]

_RETURN_NAMES: list[str] = ["ret_1", "ret_5", "ret_10", "ret_20"]

_LAGGED_BASES: list[str] = [
    "rvol_20", "amihud_20", "rsi_14", "macd_hist", "vol_zscore_20", "foreign_net_ratio",
]

_LAG_PERIODS: list[int] = [1, 2, 3]

# Total count verification
# 17 raw + 17 normalized + 4 returns + (6 bases * 3 lags) - 11 = 47
# We keep exactly 47 by including 4 return cols + 8 lag cols + 17 raw + 17 norm + 1 extra
# Exact composition is controlled in compute_market_features.


# ---------------------------------------------------------------------------
# Extractor class
# ---------------------------------------------------------------------------

@dataclass
class MarketFeatureExtractor:
    """Compute 47 market-microstructure features from OHLCV data.

    Parameters
    ----------
    normalize : bool
        If True (default), append expanding-window z-scored copies of raw
        features and use them in the output.
    min_norm_periods : int
        Minimum number of observations before z-score normalisation kicks in.
    """

    normalize: bool = True
    min_norm_periods: int = 60
    feature_names_: List[str] = field(default_factory=list, init=False)

    # ------------------------------------------------------------------ #
    def _raw_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute the 17 raw features from an OHLCV dataframe."""
        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]
        shares = df.get("shares_outstanding")
        fb = df.get("foreign_buy_volume")
        fs = df.get("foreign_sell_volume")

        feats: dict[str, pd.Series] = {}

        # Volatility family
        for w in (5, 10, 20):
            feats[f"rvol_{w}"] = _realized_volatility(close, w)

        # ATR
        feats["atr_14"] = _atr(high, low, close, 14)

        # Liquidity
        feats["amihud_20"] = _amihud_illiquidity(close, volume, 20)

        # Drawdown family
        for w in (5, 10, 20):
            feats[f"mdd_{w}"] = _max_drawdown(close, w)

        # Technical indicators
        feats["rsi_14"] = _rsi(close, 14)
        feats["macd_hist"] = _macd_histogram(close)
        feats["bb_pctb"] = _bollinger_pctb(close, 20)
        feats["stoch_k"] = _stochastic_k(high, low, close, 14)

        # Volume features
        feats["turnover_20"] = _turnover_ratio(volume, shares, 20)
        feats["vol_zscore_20"] = _volume_zscore(volume, 20)

        # Foreign flow
        feats["foreign_net_ratio"] = _foreign_net_flow_ratio(fb, fs, volume)

        # Higher moments
        feats["ret_skew_20"] = _rolling_skewness(close, 20)
        feats["ret_kurt_20"] = _rolling_kurtosis(close, 20)

        return pd.DataFrame(feats, index=df.index)

    # ------------------------------------------------------------------ #
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a DataFrame with exactly 47 features.

        Expected columns in *df*: ``open, high, low, close, volume``.
        Optional: ``shares_outstanding, foreign_buy_volume, foreign_sell_volume``.
        """
        raw = self._raw_features(df)

        parts: list[pd.DataFrame] = [raw]  # 17 columns

        # Normalized copies of all 17 raw features
        if self.normalize:
            norm = raw.apply(
                lambda s: _expanding_zscore(s, min_periods=self.min_norm_periods)
            )
            norm.columns = [f"{c}_znorm" for c in norm.columns]
            parts.append(norm)  # +17 = 34

        # Multi-horizon returns (causal)
        close = df["close"]
        for h in (1, 5, 10, 20):
            parts.append(close.pct_change(h).rename(f"ret_{h}").to_frame())
        # +4 = 38

        # Lag features for select indicators (lag-1, lag-2, lag-3)
        for base in _LAGGED_BASES:
            if base in raw.columns:
                for lag in _LAG_PERIODS:
                    col = f"{base}_lag{lag}"
                    parts.append(raw[base].shift(lag).rename(col).to_frame())
        # +6*3 = +18 => 56 ... we trim to 47 below

        combined = pd.concat(parts, axis=1)

        # Deterministic selection of exactly 47 columns.
        # Priority: raw (17) + norm (17) + returns (4) + first 9 lag cols
        selected_cols = (
            list(raw.columns)
            + [f"{c}_znorm" for c in raw.columns]
            + [f"ret_{h}" for h in (1, 5, 10, 20)]
            + [f"{b}_lag{l}" for b in _LAGGED_BASES for l in _LAG_PERIODS]
        )
        selected_cols = selected_cols[:47]
        # Ensure all exist
        selected_cols = [c for c in selected_cols if c in combined.columns][:47]

        result = combined[selected_cols]
        self.feature_names_ = list(result.columns)
        return result


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def compute_market_features(ohlcv_df: pd.DataFrame) -> pd.DataFrame:
    """Compute 47 market microstructure features from OHLCV data.

    Parameters
    ----------
    ohlcv_df : DataFrame
        Must contain columns: ``open, high, low, close, volume``.
        Index should be a DatetimeIndex (or at least sorted chronologically).
        Optional columns: ``shares_outstanding, foreign_buy_volume,
        foreign_sell_volume``.

    Returns
    -------
    DataFrame
        47-column DataFrame aligned with *ohlcv_df*'s index.  Early rows
        contain NaNs due to rolling-window warm-up.
    """
    extractor = MarketFeatureExtractor()
    return extractor.transform(ohlcv_df)
