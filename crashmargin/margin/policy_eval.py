"""
Margin Policy Evaluation (Section 5.5, Table 3).

Simulates portfolio performance under different margin management strategies
during crash and non-crash periods. Compares No Margin Control, Static 50%,
GARCH VaR 99%, and CrashMargin Dynamic strategies.

Paper targets (Table 3):
    | Strategy         | Avg Loss | Max Loss | Margin Calls | Utilization |
    |------------------|----------|----------|--------------|-------------|
    | No Margin        | -18.3%   | -37.1%   | N/A          | 100.0%      |
    | Static 50%       | -12.7%   | -24.8%   | 142          | 50.0%       |
    | GARCH VaR 99%    | -9.4%    | -19.2%   | 98           | 52.3%       |
    | CrashMargin      | -5.8%    | -13.5%   | 47           | 53.1%       |

CrashMargin achieves 54.3% loss reduction vs static and 66.9% margin call
reduction vs static.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class MarginStrategy(Enum):
    """Available margin management strategies."""

    NO_MARGIN = "no_margin"
    STATIC_50 = "static_50"
    GARCH_VAR = "garch_var"
    CRASHMARGIN = "crashmargin"


@dataclass
class PolicyMetrics:
    """Results from a margin policy simulation.

    Attributes:
        strategy: Name of the margin strategy used.
        avg_loss_crash: Average portfolio loss during crash periods (negative).
        max_loss: Maximum single-period portfolio loss (negative).
        margin_call_count: Number of margin calls triggered during simulation.
        capital_utilization: Average fraction of capital deployed.
        total_return: Cumulative portfolio return over the simulation.
        sharpe_ratio: Annualized Sharpe ratio (252 trading days).
    """

    strategy: str
    avg_loss_crash: float
    max_loss: float
    margin_call_count: int
    capital_utilization: float
    total_return: float = 0.0
    sharpe_ratio: float = 0.0


class MarginPolicySimulator:
    """Simulate portfolio with margin lending under different strategies (Section 5.5).

    Models a portfolio that can use margin lending to lever up. Different
    strategies control how much margin is permitted on each day. When the
    portfolio value drops below the margin maintenance threshold, a margin
    call is triggered and positions are forcibly reduced.

    Args:
        initial_capital: Starting portfolio value. Default: 1_000_000.
        maintenance_margin: Margin maintenance ratio that triggers calls.
            When equity / position_value < maintenance_margin, a margin
            call is issued. Default: 0.30.
        margin_call_threshold: Minimum equity ratio before forced liquidation.
            Default: 0.25.
        crash_threshold: Return threshold below which a day is considered
            a crash period for loss computation. Default: -0.02 (-2%).
    """

    def __init__(
        self,
        initial_capital: float = 1_000_000.0,
        maintenance_margin: float = 0.30,
        margin_call_threshold: float = 0.25,
        crash_threshold: float = -0.02,
    ):
        self.initial_capital = initial_capital
        self.maintenance_margin = maintenance_margin
        self.margin_call_threshold = margin_call_threshold
        self.crash_threshold = crash_threshold

    def _compute_garch_var(
        self,
        returns: np.ndarray,
        t: int,
        lookback: int = 20,
        confidence: float = 0.99,
    ) -> float:
        """Estimate VaR using a simplified GARCH(1,1) volatility model.

        Uses an exponentially weighted variance estimator as a lightweight
        GARCH proxy (avoids arch library dependency). The VaR at the given
        confidence level determines the maximum permissible exposure.

        Args:
            returns: Full return series, shape (n_days,).
            t: Current time index.
            lookback: Number of historical days for estimation.
            confidence: VaR confidence level (e.g., 0.99 for 99%).

        Returns:
            Margin ratio implied by the GARCH VaR estimate, in [0.30, 0.90].
        """
        from scipy.stats import norm

        start = max(0, t - lookback)
        hist = returns[start:t]
        if len(hist) < 10:
            return 0.50  # fallback to conservative static margin

        # Exponentially weighted variance (GARCH proxy, lambda=0.94)
        decay = 0.94
        weights = decay ** np.arange(len(hist) - 1, -1, -1)
        weights /= weights.sum()
        weighted_var = np.sum(weights * (hist - np.mean(hist)) ** 2)
        vol = np.sqrt(weighted_var)

        # VaR at confidence level
        z_score = norm.ppf(confidence)
        var_estimate = z_score * vol

        # Map VaR to margin ratio: higher VaR -> higher required margin
        # Scale so that typical vol maps to ~50% margin
        margin_ratio = np.clip(0.30 + var_estimate * 5.0, 0.30, 0.90)
        return float(margin_ratio)

    def simulate(
        self,
        returns: np.ndarray,
        crash_probs: np.ndarray | None = None,
        margin_strategy: str | MarginStrategy = MarginStrategy.CRASHMARGIN,
        dynamic_margin_calculator=None,
    ) -> dict:
        """Run a margin policy simulation over the return series.

        Args:
            returns: Daily portfolio returns, shape (n_days,).
            crash_probs: Predicted crash probabilities from the CrashMargin
                model, shape (n_days,). Required for CRASHMARGIN strategy.
            margin_strategy: One of the MarginStrategy enum values or its
                string name.
            dynamic_margin_calculator: Instance of DynamicMarginCalculator.
                Required for CRASHMARGIN strategy. If None and strategy is
                CRASHMARGIN, a default calculator is constructed.

        Returns:
            Dictionary with keys:
                - avg_loss_crash: Average loss during crash periods.
                - max_loss: Maximum single-period loss.
                - margin_call_count: Total margin calls triggered.
                - capital_utilization: Mean capital utilization ratio.
                - total_return: Cumulative return.
                - sharpe_ratio: Annualized Sharpe ratio.
                - equity_curve: Array of portfolio equity values.
                - daily_utilization: Array of daily utilization ratios.
        """
        if isinstance(margin_strategy, str):
            margin_strategy = MarginStrategy(margin_strategy)

        returns = np.asarray(returns, dtype=np.float64)
        n_days = len(returns)

        if crash_probs is not None:
            crash_probs = np.asarray(crash_probs, dtype=np.float64)

        # Lazy import to avoid circular dependency
        if margin_strategy == MarginStrategy.CRASHMARGIN:
            if dynamic_margin_calculator is None:
                from crashmargin.margin.dynamic_margin import DynamicMarginCalculator
                dynamic_margin_calculator = DynamicMarginCalculator()

        # Simulation state
        equity = self.initial_capital
        equity_curve = np.zeros(n_days)
        daily_utilization = np.zeros(n_days)
        margin_calls = 0
        crash_losses = []

        for t in range(n_days):
            # Determine margin ratio (fraction of equity used as collateral)
            if margin_strategy == MarginStrategy.NO_MARGIN:
                # Full leverage, no margin control
                margin_ratio = 0.0  # no collateral withheld
                leverage = 1.0 / max(1.0 - margin_ratio, 0.01)
            elif margin_strategy == MarginStrategy.STATIC_50:
                margin_ratio = 0.50
                leverage = 1.0 / margin_ratio
            elif margin_strategy == MarginStrategy.GARCH_VAR:
                margin_ratio = self._compute_garch_var(returns, t)
                leverage = 1.0 / margin_ratio
            elif margin_strategy == MarginStrategy.CRASHMARGIN:
                prob = crash_probs[t] if crash_probs is not None else 0.0
                margin_ratio = dynamic_margin_calculator.compute_margin(prob)
                leverage = 1.0 / margin_ratio
            else:
                raise ValueError(f"Unknown strategy: {margin_strategy}")

            # Capital utilization: fraction of total possible exposure used
            if margin_strategy == MarginStrategy.NO_MARGIN:
                utilization = 1.0
            else:
                # Utilization = 1/margin_ratio represents how aggressively
                # we can use capital; normalize to [0, 1] range
                utilization = 1.0 / leverage  # = margin_ratio
                utilization = 1.0 - (margin_ratio - self.maintenance_margin) / (
                    1.0 - self.maintenance_margin
                )
                utilization = np.clip(utilization, 0.0, 1.0)

            daily_utilization[t] = utilization

            # Apply return with leverage
            if margin_strategy == MarginStrategy.NO_MARGIN:
                daily_return = returns[t]
            else:
                # Leveraged return: equity portion earns the full return,
                # borrowed portion also earns return but costs are ignored
                # for simplicity in this evaluation
                daily_return = returns[t] * leverage

            # Update equity
            equity *= (1.0 + daily_return)
            equity_curve[t] = equity

            # Check margin call: equity falls below maintenance threshold
            if margin_strategy != MarginStrategy.NO_MARGIN:
                if daily_return < -self.maintenance_margin:
                    margin_calls += 1
                    # Forced de-leveraging: reset to initial margin
                    # (simulates selling to meet margin requirements)
                    equity *= 0.95  # 5% penalty for forced liquidation

            # Track crash period losses
            if returns[t] < self.crash_threshold:
                crash_losses.append(daily_return)

        # Compute summary metrics
        if len(crash_losses) > 0:
            avg_loss_crash = float(np.mean(crash_losses))
            max_loss = float(np.min(crash_losses))
        else:
            avg_loss_crash = 0.0
            max_loss = 0.0

        total_return = (equity_curve[-1] / self.initial_capital) - 1.0
        daily_returns = np.diff(equity_curve) / equity_curve[:-1]
        sharpe = (
            float(np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252))
            if np.std(daily_returns) > 0
            else 0.0
        )

        return {
            "strategy": margin_strategy.value,
            "avg_loss_crash": avg_loss_crash,
            "max_loss": max_loss,
            "margin_call_count": margin_calls,
            "capital_utilization": float(np.mean(daily_utilization)),
            "total_return": total_return,
            "sharpe_ratio": sharpe,
            "equity_curve": equity_curve,
            "daily_utilization": daily_utilization,
        }

    def compare_strategies(
        self,
        returns: np.ndarray,
        crash_probs: np.ndarray | None = None,
        dynamic_margin_calculator=None,
    ) -> dict[str, dict]:
        """Run all four strategies and return comparative results.

        Convenience method that simulates all strategies from Table 3 and
        computes relative improvement metrics.

        Args:
            returns: Daily portfolio returns, shape (n_days,).
            crash_probs: Predicted crash probabilities, shape (n_days,).
            dynamic_margin_calculator: DynamicMarginCalculator instance.

        Returns:
            Dictionary mapping strategy names to their metric dictionaries.
            Includes an additional 'relative_improvement' key with
            CrashMargin improvements vs other strategies.
        """
        results = {}
        for strategy in MarginStrategy:
            results[strategy.value] = self.simulate(
                returns=returns,
                crash_probs=crash_probs,
                margin_strategy=strategy,
                dynamic_margin_calculator=dynamic_margin_calculator,
            )

        # Compute relative improvements (Section 5.5)
        static_loss = results[MarginStrategy.STATIC_50.value]["avg_loss_crash"]
        cm_loss = results[MarginStrategy.CRASHMARGIN.value]["avg_loss_crash"]

        static_calls = results[MarginStrategy.STATIC_50.value]["margin_call_count"]
        cm_calls = results[MarginStrategy.CRASHMARGIN.value]["margin_call_count"]

        loss_reduction = (
            (abs(static_loss) - abs(cm_loss)) / abs(static_loss) * 100
            if static_loss != 0
            else 0.0
        )
        call_reduction = (
            (static_calls - cm_calls) / static_calls * 100
            if static_calls > 0
            else 0.0
        )

        results["relative_improvement"] = {
            "loss_reduction_vs_static_pct": loss_reduction,
            "margin_call_reduction_vs_static_pct": call_reduction,
            # Paper targets: 54.3% loss reduction, 66.9% call reduction
        }

        return results
