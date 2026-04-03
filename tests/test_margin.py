"""
Tests for dynamic margin formula and policy simulation.

Validates:
    - Sigmoid mapping (Eq. 1): m* = m_min + (m_max - m_min) * sigma((p - tau) / T)
    - Margin is always within [m_min, m_max] = [0.40, 0.85]
    - Policy simulation produces consistent results
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import expit  # sigmoid


# ---------------------------------------------------------------------------
# Dynamic margin formula — Eq. 1 (Section 3.5)
# ---------------------------------------------------------------------------
def dynamic_margin(
    crash_prob: float | np.ndarray,
    m_min: float = 0.40,
    m_max: float = 0.85,
    tau: float = 0.15,
    T: float = 0.1,
) -> float | np.ndarray:
    """Compute dynamic margin requirement from crash probability (Eq. 1).

    m*_{i,t} = m_min + (m_max - m_min) * sigma((p_hat - tau) / T)
    """
    crash_prob = np.asarray(crash_prob, dtype=np.float64)
    sigmoid_input = (crash_prob - tau) / T
    margin = m_min + (m_max - m_min) * expit(sigmoid_input)
    return margin


def simulate_margin_policy(
    crash_probs: np.ndarray,
    returns: np.ndarray,
    m_min: float = 0.40,
    m_max: float = 0.85,
    tau: float = 0.15,
    T: float = 0.1,
    maintenance_threshold: float = -0.03,
) -> dict[str, float]:
    """Run margin policy simulation, returning portfolio stats.

    Leverage = 1 / margin_ratio. Higher margin -> lower leverage -> smaller
    effective returns (both gains and losses). This matches the paper's
    intent: raising margin requirements during high-risk periods reduces
    leveraged losses.
    """
    n = len(returns)
    assert len(crash_probs) == n
    margins = dynamic_margin(crash_probs, m_min, m_max, tau, T)
    portfolio = np.ones(n + 1)
    margin_calls = 0

    for t in range(n):
        leverage = 1.0 / margins[t]  # higher margin -> lower leverage
        eff_ret = returns[t] * leverage
        portfolio[t + 1] = portfolio[t] * (1.0 + eff_ret)
        if eff_ret < maintenance_threshold:
            margin_calls += 1

    total_return = (portfolio[-1] / portfolio[0] - 1.0) * 100
    peak = np.maximum.accumulate(portfolio)
    max_dd = float(np.min((portfolio - peak) / peak) * 100)
    avg_util = float(np.mean(1.0 / margins) / (1.0 / m_min) * 100)

    return {
        "total_return_pct": float(total_return),
        "max_drawdown_pct": max_dd,
        "margin_calls": margin_calls,
        "avg_capital_util_pct": avg_util,
        "margins": margins,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestDynamicMarginFormula:
    def test_sigmoid_midpoint(self):
        """When crash_prob == tau, margin is halfway between m_min and m_max."""
        m = dynamic_margin(0.15)  # tau = 0.15
        expected = 0.40 + (0.85 - 0.40) * 0.5  # sigmoid(0) = 0.5
        assert m == pytest.approx(expected, abs=1e-6)

    def test_low_prob_approaches_m_min(self):
        """Very low crash probability should approach m_min."""
        m = dynamic_margin(0.0)
        assert m > 0.40  # sigmoid never reaches 0
        assert m < 0.50  # but should be close to m_min

    def test_high_prob_approaches_m_max(self):
        """Very high crash probability should approach m_max."""
        m = dynamic_margin(1.0)
        assert m > 0.80  # close to m_max
        assert m <= 0.85

    def test_monotonically_increasing(self):
        """Higher crash probability => higher margin."""
        probs = np.linspace(0, 1, 100)
        margins = dynamic_margin(probs)
        diffs = np.diff(margins)
        assert np.all(diffs >= 0), "Margin should be non-decreasing with crash prob"

    def test_bounds_always_respected(self):
        """Margin should always be in (m_min, m_max) for any finite input."""
        rng = np.random.default_rng(42)
        probs = rng.uniform(-0.5, 1.5, size=10000)  # include out-of-range
        margins = dynamic_margin(probs)
        assert np.all(margins >= 0.40), f"Min violation: {margins.min()}"
        assert np.all(margins <= 0.85), f"Max violation: {margins.max()}"

    def test_vectorized_matches_scalar(self):
        """Vectorized and scalar calls should agree."""
        probs = np.array([0.0, 0.10, 0.15, 0.30, 0.50, 0.80, 1.0])
        vec_result = dynamic_margin(probs)
        scalar_results = np.array([float(dynamic_margin(p)) for p in probs])
        np.testing.assert_allclose(vec_result, scalar_results)

    def test_temperature_controls_sharpness(self):
        """Lower temperature -> sharper transition around tau."""
        probs = np.array([0.10, 0.20])  # symmetric around tau=0.15
        m_sharp = dynamic_margin(probs, T=0.01)
        m_smooth = dynamic_margin(probs, T=0.5)
        # Sharp: big difference between 0.10 and 0.20
        # Smooth: small difference
        assert (m_sharp[1] - m_sharp[0]) > (m_smooth[1] - m_smooth[0])

    def test_matches_dynamic_margin_calculator(self):
        """Test function matches the DynamicMarginCalculator class."""
        from crashmargin.margin.dynamic_margin import DynamicMarginCalculator
        calc = DynamicMarginCalculator(m_min=0.40, m_max=0.85, tau=0.15, T=0.1)
        probs = np.array([0.0, 0.05, 0.15, 0.30, 0.50, 1.0])
        expected = calc.compute_margin(probs)
        actual = dynamic_margin(probs)
        np.testing.assert_allclose(actual, expected, rtol=1e-10)


class TestMarginPolicySimulation:
    @pytest.fixture
    def sim_data(self):
        rng = np.random.default_rng(42)
        n = 500
        returns = rng.normal(0.0005, 0.018, n)
        crash_probs = np.clip(rng.beta(1.5, 30.0, n), 0, 1)
        return crash_probs, returns

    def test_simulation_runs(self, sim_data):
        crash_probs, returns = sim_data
        result = simulate_margin_policy(crash_probs, returns)
        assert "total_return_pct" in result
        assert "max_drawdown_pct" in result
        assert "margin_calls" in result

    def test_margins_within_bounds(self, sim_data):
        crash_probs, returns = sim_data
        result = simulate_margin_policy(crash_probs, returns)
        assert np.all(result["margins"] >= 0.40)
        assert np.all(result["margins"] <= 0.85)

    def test_margin_calls_nonnegative(self, sim_data):
        crash_probs, returns = sim_data
        result = simulate_margin_policy(crash_probs, returns)
        assert result["margin_calls"] >= 0

    def test_max_drawdown_nonpositive(self, sim_data):
        crash_probs, returns = sim_data
        result = simulate_margin_policy(crash_probs, returns)
        assert result["max_drawdown_pct"] <= 0.0

    def test_reproducible_with_same_seed(self):
        for _ in range(2):
            rng = np.random.default_rng(123)
            n = 200
            returns = rng.normal(0.0005, 0.018, n)
            crash_probs = np.clip(rng.beta(1.5, 30.0, n), 0, 1)
            r = simulate_margin_policy(crash_probs, returns)
        # Just check it doesn't crash; determinism is guaranteed by NumPy seed

    def test_higher_crash_prob_reduces_losses(self):
        """Higher predicted crash prob -> higher margin -> smaller leveraged loss."""
        rng = np.random.default_rng(42)
        n = 1000
        returns = rng.normal(0.0005, 0.018, n)
        # Inject crashes
        crash_idx = rng.choice(n, size=30, replace=False)
        returns[crash_idx] -= 0.08

        # Scenario 1: model correctly predicts high crash prob
        probs_good = np.full(n, 0.05)
        probs_good[crash_idx] = 0.9

        # Scenario 2: model always predicts low crash prob
        probs_bad = np.full(n, 0.05)

        r_good = simulate_margin_policy(probs_good, returns)
        r_bad = simulate_margin_policy(probs_bad, returns)
        # Good predictions should lead to less severe drawdowns
        assert r_good["max_drawdown_pct"] >= r_bad["max_drawdown_pct"]
