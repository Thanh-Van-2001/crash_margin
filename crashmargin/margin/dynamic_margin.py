"""
Dynamic Margin Computation (Section 3.5, Eq. 1).

Maps crash probabilities from the CrashMargin model to margin ratios
using a calibrated sigmoid mapping. Higher predicted crash probability
leads to higher required margin (more conservative lending), while low
crash probability allows more aggressive margin utilization.

Equation 1:
    m*_{i,t} = m_min + (m_max - m_min) * sigma((p_hat_{i,t} - tau) / T)

where sigma is the sigmoid function, tau is the threshold, and T is the
temperature controlling transition sharpness. Default parameters are
calibrated on the 2021 validation set (Section 4.1).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit  # sigmoid


class DynamicMarginCalculator:
    """Compute dynamic margin ratios from crash probabilities (Eq. 1).

    The mapping uses a sigmoid transfer function to smoothly interpolate
    between m_min (low-risk regime) and m_max (high-risk regime). The
    threshold tau and temperature T are calibrated to maximize economic
    utility on a validation set.

    Args:
        m_min: Minimum margin ratio (low crash probability). Default: 0.40.
        m_max: Maximum margin ratio (high crash probability). Default: 0.85.
        tau: Probability threshold for the sigmoid midpoint. Default: 0.15.
        T: Temperature controlling sigmoid sharpness. Default: 0.1.
    """

    def __init__(
        self,
        m_min: float = 0.40,
        m_max: float = 0.85,
        tau: float = 0.15,
        T: float = 0.1,
    ):
        self.m_min = m_min
        self.m_max = m_max
        self.tau = tau
        self.T = T

    def compute_margin(self, crash_prob: np.ndarray | float) -> np.ndarray:
        """Compute dynamic margin ratios from crash probabilities (Eq. 1).

        m*_{i,t} = m_min + (m_max - m_min) * sigma((p_hat - tau) / T)

        Args:
            crash_prob: Predicted crash probability, scalar or array with
                values in [0, 1].

        Returns:
            Margin ratios in [m_min, m_max]. Same shape as input.
        """
        crash_prob = np.asarray(crash_prob, dtype=np.float64)
        sigmoid_input = (crash_prob - self.tau) / self.T
        margin = self.m_min + (self.m_max - self.m_min) * expit(sigmoid_input)
        return margin

    def calibrate(
        self,
        val_probs: np.ndarray,
        val_labels: np.ndarray,
        val_returns: np.ndarray,
    ) -> tuple[float, float]:
        """Calibrate (tau, T) on the validation set (2021) to maximize utility.

        The calibration objective minimizes the sum of:
          - Average portfolio loss during crash periods (weighted by margin)
          - Penalty for excessive capital lock-up (unused margin capacity)

        Specifically, the economic utility is:
            U(tau, T) = -mean(loss_crash * margin_ratio)
                        - lambda * mean(margin_ratio_non_crash)

        where lambda = 0.1 balances crash protection vs capital efficiency.

        Args:
            val_probs: Model predicted crash probabilities on the validation
                set, shape (n_samples,).
            val_labels: Binary crash labels (1 = crash), shape (n_samples,).
            val_returns: Realized returns, shape (n_samples,).

        Returns:
            Tuple of calibrated (tau, T) values. Also updates self.tau
            and self.T in-place.
        """
        val_probs = np.asarray(val_probs, dtype=np.float64)
        val_labels = np.asarray(val_labels, dtype=np.float64)
        val_returns = np.asarray(val_returns, dtype=np.float64)

        crash_mask = val_labels == 1
        non_crash_mask = ~crash_mask

        capital_efficiency_weight = 0.1

        def objective(params: np.ndarray) -> float:
            tau_candidate, T_candidate = params
            # Temperature must be positive
            if T_candidate <= 1e-6:
                return 1e6

            sigmoid_input = (val_probs - tau_candidate) / T_candidate
            margins = self.m_min + (self.m_max - self.m_min) * expit(sigmoid_input)

            # Crash protection: portfolio loss scaled by (1 - margin)
            # Higher margin means less leverage, so loss is reduced
            if crash_mask.any():
                leveraged_loss = val_returns[crash_mask] * (1.0 - margins[crash_mask])
                crash_cost = np.mean(leveraged_loss)
            else:
                crash_cost = 0.0

            # Capital efficiency: penalize high margins during non-crash
            if non_crash_mask.any():
                idle_capital_cost = capital_efficiency_weight * np.mean(
                    margins[non_crash_mask]
                )
            else:
                idle_capital_cost = 0.0

            # Minimize crash loss (negative returns become positive cost)
            # and idle capital cost
            return crash_cost + idle_capital_cost

        # Grid search initialization for robustness
        best_result = None
        best_cost = np.inf

        for tau_init in [0.05, 0.10, 0.15, 0.20, 0.30]:
            for T_init in [0.03, 0.05, 0.10, 0.15, 0.20]:
                result = minimize(
                    objective,
                    x0=np.array([tau_init, T_init]),
                    method="Nelder-Mead",
                    bounds=None,
                    options={"maxiter": 1000, "xatol": 1e-5, "fatol": 1e-8},
                )
                if result.fun < best_cost:
                    best_cost = result.fun
                    best_result = result

        tau_opt, T_opt = best_result.x
        # Enforce temperature positivity
        T_opt = max(T_opt, 1e-4)

        self.tau = float(tau_opt)
        self.T = float(T_opt)

        return (self.tau, self.T)

    def __repr__(self) -> str:
        return (
            f"DynamicMarginCalculator("
            f"m_min={self.m_min}, m_max={self.m_max}, "
            f"tau={self.tau:.4f}, T={self.T:.4f})"
        )
