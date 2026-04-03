"""
SHAP Explainability for CrashMargin (Section 5.6).

Provides model-agnostic feature importance analysis using SHAP
(SHapley Additive exPlanations). Supports both DeepExplainer for
PyTorch models and KernelExplainer as a fallback.

Key findings from the paper (Section 5.6):
    Top-3 features by mean |SHAP|:
        1. realized_vol_5d:     0.142
        2. max_drawdown_20d:    0.118
        3. margin_debt_mcap:    0.097
    Margin features collectively account for 16.6% of total predictive power.
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import shap
except ImportError:
    shap = None

try:
    import matplotlib.pyplot as plt
    import matplotlib

    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


# Paper-reported top features and SHAP values (Section 5.6, Figure 8)
PAPER_TOP_FEATURES = [
    ("realized_vol_5d", 0.142),
    ("max_drawdown_20d", 0.118),
    ("margin_debt_mcap", 0.097),
    ("amihud_illiquidity_10d", 0.089),
    ("sentiment_neg_5d", 0.082),
    ("return_vol_ratio_20d", 0.076),
    ("margin_util_rate", 0.071),
    ("order_imbalance_5d", 0.065),
    ("turnover_shock_10d", 0.059),
    ("spread_mean_20d", 0.053),
    ("graph_centrality_sector", 0.048),
    ("news_volume_3d", 0.044),
    ("margin_concentration_hhi", 0.041),
    ("price_momentum_10d", 0.038),
    ("foreign_flow_net_5d", 0.035),
]


class CrashMarginExplainer:
    """SHAP-based explainability wrapper for CrashMargin models (Section 5.6).

    Automatically selects DeepExplainer for PyTorch neural network models
    and falls back to KernelExplainer for non-differentiable models or
    when DeepExplainer encounters compatibility issues.

    Args:
        feature_names: List of feature names corresponding to input columns.
            If None, features are named "feature_0", "feature_1", etc.
        use_deep: Whether to attempt DeepExplainer first. Default: True.
        n_background: Number of background samples for the explainer.
            Default: 100.
    """

    def __init__(
        self,
        feature_names: list[str] | None = None,
        use_deep: bool = True,
        n_background: int = 100,
    ):
        if shap is None:
            raise ImportError(
                "The 'shap' package is required for CrashMarginExplainer. "
                "Install it with: pip install shap"
            )

        self.feature_names = feature_names
        self.use_deep = use_deep
        self.n_background = n_background
        self._explainer = None
        self._shap_values = None

    def explain(
        self,
        model: Any,
        X: np.ndarray,
        background: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute SHAP values for the given inputs.

        Attempts DeepExplainer first (for PyTorch nn.Module models), then
        falls back to KernelExplainer if DeepExplainer fails or is disabled.

        Args:
            model: Trained model. Can be a PyTorch nn.Module or any callable
                that maps inputs to predictions.
            X: Input features to explain, shape (n_samples, n_features).
            background: Background dataset for the explainer. If None, a
                random subsample of X is used. Shape (n_background, n_features).

        Returns:
            SHAP values array of shape (n_samples, n_features). Positive
            values indicate features pushing toward crash prediction.
        """
        import torch

        X = np.asarray(X, dtype=np.float32)

        if background is None:
            n_bg = min(self.n_background, X.shape[0])
            indices = np.random.choice(X.shape[0], size=n_bg, replace=False)
            background = X[indices]

        # Set feature names if not provided
        if self.feature_names is None:
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        explainer_created = False

        # Attempt DeepExplainer for PyTorch models
        if self.use_deep and isinstance(model, torch.nn.Module):
            try:
                model.eval()
                bg_tensor = torch.tensor(background, dtype=torch.float32)
                self._explainer = shap.DeepExplainer(model, bg_tensor)
                X_tensor = torch.tensor(X, dtype=torch.float32)
                self._shap_values = self._explainer.shap_values(X_tensor)
                explainer_created = True
            except Exception:
                # Fall through to KernelExplainer
                explainer_created = False

        # Fallback to KernelExplainer
        if not explainer_created:
            if isinstance(model, torch.nn.Module):
                model.eval()

                def predict_fn(x: np.ndarray) -> np.ndarray:
                    with torch.no_grad():
                        t = torch.tensor(x, dtype=torch.float32)
                        output = model(t)
                        if output.ndim > 1:
                            output = output[:, -1]  # crash class probability
                        return output.numpy()
            else:
                predict_fn = model

            self._explainer = shap.KernelExplainer(predict_fn, background)
            self._shap_values = self._explainer.shap_values(X)

        # Ensure consistent output format
        if isinstance(self._shap_values, list):
            # Multi-class: take crash class (last class)
            self._shap_values = self._shap_values[-1]

        self._shap_values = np.asarray(self._shap_values)
        return self._shap_values

    def top_features(
        self,
        shap_values: np.ndarray | None = None,
        n: int = 15,
    ) -> list[tuple[str, float]]:
        """Rank features by mean absolute SHAP value.

        Args:
            shap_values: SHAP values array of shape (n_samples, n_features).
                If None, uses the values from the last call to explain().
            n: Number of top features to return. Default: 15.

        Returns:
            List of (feature_name, mean_abs_shap) tuples sorted by
            importance descending. Paper top-3: realized_vol_5d (0.142),
            max_drawdown_20d (0.118), margin_debt_mcap (0.097).
        """
        if shap_values is None:
            if self._shap_values is None:
                raise ValueError(
                    "No SHAP values available. Call explain() first or "
                    "provide shap_values directly."
                )
            shap_values = self._shap_values

        shap_values = np.asarray(shap_values)
        mean_abs = np.mean(np.abs(shap_values), axis=0)

        # Pair with feature names
        if self.feature_names is not None and len(self.feature_names) == len(mean_abs):
            feature_importance = list(zip(self.feature_names, mean_abs.tolist()))
        else:
            feature_importance = [
                (f"feature_{i}", float(v)) for i, v in enumerate(mean_abs)
            ]

        # Sort descending by importance
        feature_importance.sort(key=lambda x: x[1], reverse=True)

        return feature_importance[:n]

    def margin_feature_contribution(
        self,
        shap_values: np.ndarray | None = None,
        margin_feature_prefix: str = "margin_",
    ) -> float:
        """Compute the collective predictive contribution of margin features.

        From Section 5.6: margin features collectively account for 16.6%
        of total predictive power.

        Args:
            shap_values: SHAP values, shape (n_samples, n_features).
            margin_feature_prefix: Prefix identifying margin-related features.

        Returns:
            Fraction of total |SHAP| attributable to margin features.
        """
        if shap_values is None:
            if self._shap_values is None:
                raise ValueError("No SHAP values available. Call explain() first.")
            shap_values = self._shap_values

        shap_values = np.asarray(shap_values)
        mean_abs = np.mean(np.abs(shap_values), axis=0)
        total_shap = np.sum(mean_abs)

        if total_shap == 0:
            return 0.0

        if self.feature_names is None:
            return 0.0

        margin_shap = sum(
            mean_abs[i]
            for i, name in enumerate(self.feature_names)
            if name.startswith(margin_feature_prefix)
        )

        return float(margin_shap / total_shap)

    def generate_waterfall(
        self,
        shap_values: np.ndarray | None = None,
        instance_idx: int = 0,
        max_display: int = 15,
    ):
        """Generate a SHAP waterfall plot for a single prediction.

        Creates a waterfall chart showing how each feature contributes to
        pushing the prediction from the base value toward the final output
        for a specific instance.

        Args:
            shap_values: SHAP values, shape (n_samples, n_features).
            instance_idx: Index of the instance to explain.
            max_display: Maximum number of features to display.

        Returns:
            matplotlib.figure.Figure: The waterfall plot figure.
        """
        if not _HAS_MPL:
            raise ImportError(
                "matplotlib is required for waterfall plots. "
                "Install it with: pip install matplotlib"
            )

        if shap_values is None:
            if self._shap_values is None:
                raise ValueError("No SHAP values available. Call explain() first.")
            shap_values = self._shap_values

        shap_values = np.asarray(shap_values)
        instance_shap = shap_values[instance_idx]

        # Get base value from explainer if available
        if self._explainer is not None and hasattr(self._explainer, "expected_value"):
            base_value = self._explainer.expected_value
            if isinstance(base_value, (list, np.ndarray)):
                base_value = base_value[-1]  # crash class
        else:
            base_value = 0.0

        # Create SHAP Explanation object for the waterfall plot
        feature_names = self.feature_names or [
            f"feature_{i}" for i in range(len(instance_shap))
        ]

        explanation = shap.Explanation(
            values=instance_shap,
            base_values=base_value,
            feature_names=feature_names,
        )

        fig, ax = plt.subplots(figsize=(10, 8))
        plt.sca(ax)
        shap.plots.waterfall(explanation, max_display=max_display, show=False)
        plt.tight_layout()

        return fig

    def generate_summary_plot(
        self,
        shap_values: np.ndarray | None = None,
        X: np.ndarray | None = None,
        max_display: int = 15,
    ):
        """Generate a SHAP beeswarm summary plot.

        Args:
            shap_values: SHAP values, shape (n_samples, n_features).
            X: Original feature values for coloring, shape (n_samples, n_features).
            max_display: Maximum number of features to display.

        Returns:
            matplotlib.figure.Figure: The summary plot figure.
        """
        if not _HAS_MPL:
            raise ImportError("matplotlib is required for summary plots.")

        if shap_values is None:
            if self._shap_values is None:
                raise ValueError("No SHAP values available. Call explain() first.")
            shap_values = self._shap_values

        fig, ax = plt.subplots(figsize=(10, 8))
        plt.sca(ax)

        feature_names = self.feature_names or [
            f"feature_{i}" for i in range(shap_values.shape[1])
        ]

        shap.summary_plot(
            shap_values,
            features=X,
            feature_names=feature_names,
            max_display=max_display,
            show=False,
        )
        plt.tight_layout()

        return fig
