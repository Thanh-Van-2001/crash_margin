"""CrashMargin explainability module (Section 5.6).

SHAP-based feature importance analysis for interpreting crash predictions
and quantifying the contribution of margin lending features.
"""

from crashmargin.explainability.shap_explain import CrashMarginExplainer

__all__ = [
    "CrashMarginExplainer",
]
