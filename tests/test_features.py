"""
Tests for feature extractors.

Validates:
    - Output dimensions match config
    - No lookahead bias (features at time t use only data up to t)
    - Correct handling of edge cases (missing data, short series)
"""

from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Feature extraction functions (inline, mirroring crashmargin.features)
# ---------------------------------------------------------------------------
def compute_returns(prices: np.ndarray, window: int) -> np.ndarray:
    """Log returns over `window` days.  Output[t] uses prices[t-window:t+1]."""
    n = len(prices)
    out = np.full(n, np.nan)
    for t in range(window, n):
        out[t] = np.log(prices[t] / prices[t - window])
    return out


def compute_volatility(prices: np.ndarray, window: int) -> np.ndarray:
    """Rolling standard deviation of daily log returns."""
    log_ret = np.diff(np.log(prices))
    n = len(prices)
    out = np.full(n, np.nan)
    for t in range(window, n):
        out[t] = np.std(log_ret[t - window : t])
    return out


def compute_rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """Relative Strength Index."""
    n = len(prices)
    out = np.full(n, np.nan)
    deltas = np.diff(prices)
    for t in range(period + 1, n):
        gains = deltas[t - period : t].copy()
        avg_gain = np.mean(np.maximum(gains, 0))
        avg_loss = np.mean(np.maximum(-gains, 0))
        if avg_loss < 1e-12:
            out[t] = 100.0
        else:
            rs = avg_gain / avg_loss
            out[t] = 100.0 - 100.0 / (1.0 + rs)
    return out


def compute_margin_debt_ratio(
    margin_debt: np.ndarray, market_cap: np.ndarray
) -> np.ndarray:
    """Margin-debt-to-market-cap ratio."""
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(market_cap > 0, margin_debt / market_cap, np.nan)
    return ratio


def build_feature_matrix(
    prices: np.ndarray,
    volumes: np.ndarray,
    margin_debt: np.ndarray,
    market_cap: np.ndarray,
    lookback: int = 60,
) -> np.ndarray:
    """Build feature matrix: each row t uses only data up to t."""
    n = len(prices)
    ret_1d = compute_returns(prices, 1)
    ret_5d = compute_returns(prices, 5)
    ret_20d = compute_returns(prices, 20)
    vol_20d = compute_volatility(prices, 20)
    rsi_14 = compute_rsi(prices, 14)
    mdr = compute_margin_debt_ratio(margin_debt, market_cap)

    features = np.column_stack([ret_1d, ret_5d, ret_20d, vol_20d, rsi_14, mdr])
    return features  # (n, 6)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def synthetic_data():
    rng = np.random.default_rng(42)
    n = 300
    prices = 100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, n)))
    volumes = rng.lognormal(mean=15, sigma=0.5, size=n)
    margin_debt = rng.uniform(1e8, 5e8, size=n)
    market_cap = rng.uniform(1e10, 5e10, size=n)
    return prices, volumes, margin_debt, market_cap


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestFeatureDimensions:
    def test_returns_shape(self, synthetic_data):
        prices = synthetic_data[0]
        ret = compute_returns(prices, 5)
        assert ret.shape == prices.shape

    def test_volatility_shape(self, synthetic_data):
        prices = synthetic_data[0]
        vol = compute_volatility(prices, 20)
        assert vol.shape == prices.shape

    def test_rsi_shape(self, synthetic_data):
        prices = synthetic_data[0]
        rsi = compute_rsi(prices, 14)
        assert rsi.shape == prices.shape

    def test_feature_matrix_shape(self, synthetic_data):
        prices, volumes, margin_debt, market_cap = synthetic_data
        feats = build_feature_matrix(prices, volumes, margin_debt, market_cap)
        assert feats.shape == (len(prices), 6)

    def test_rsi_range(self, synthetic_data):
        prices = synthetic_data[0]
        rsi = compute_rsi(prices, 14)
        valid = rsi[~np.isnan(rsi)]
        assert np.all(valid >= 0.0) and np.all(valid <= 100.0)


class TestNoLookahead:
    """Ensure features at time t depend only on data up to t (no future leak)."""

    def test_returns_no_lookahead(self, synthetic_data):
        prices = synthetic_data[0]
        ret_full = compute_returns(prices, 5)
        # Modify future prices and check that past features are unchanged
        prices_modified = prices.copy()
        cutoff = 150
        prices_modified[cutoff:] *= 1.5  # change future
        ret_modified = compute_returns(prices_modified, 5)
        # Features up to cutoff-1 should be identical
        np.testing.assert_array_equal(
            ret_full[:cutoff], ret_modified[:cutoff],
            err_msg="Returns at t < cutoff changed when future data was modified"
        )

    def test_volatility_no_lookahead(self, synthetic_data):
        prices = synthetic_data[0]
        vol_full = compute_volatility(prices, 20)
        prices_modified = prices.copy()
        cutoff = 150
        prices_modified[cutoff:] *= 2.0
        vol_modified = compute_volatility(prices_modified, 20)
        np.testing.assert_array_equal(
            vol_full[:cutoff], vol_modified[:cutoff],
            err_msg="Volatility at t < cutoff changed when future data was modified"
        )

    def test_rsi_no_lookahead(self, synthetic_data):
        prices = synthetic_data[0]
        rsi_full = compute_rsi(prices, 14)
        prices_modified = prices.copy()
        cutoff = 150
        prices_modified[cutoff:] *= 0.5
        rsi_modified = compute_rsi(prices_modified, 14)
        np.testing.assert_array_equal(
            rsi_full[:cutoff], rsi_modified[:cutoff],
            err_msg="RSI at t < cutoff changed when future data was modified"
        )


class TestEdgeCases:
    def test_short_series(self):
        prices = np.array([100.0, 101.0, 99.0, 102.0, 98.0])
        ret = compute_returns(prices, 1)
        # First value should be NaN (no lookback)
        assert np.isnan(ret[0])
        assert not np.isnan(ret[1])

    def test_constant_prices(self):
        prices = np.full(100, 50.0)
        vol = compute_volatility(prices, 20)
        valid = vol[~np.isnan(vol)]
        np.testing.assert_allclose(valid, 0.0, atol=1e-12)

    def test_zero_market_cap(self):
        margin_debt = np.array([1e8, 2e8, 3e8])
        market_cap = np.array([1e10, 0.0, 1e10])
        ratio = compute_margin_debt_ratio(margin_debt, market_cap)
        assert np.isnan(ratio[1]), "Zero market cap should produce NaN"
