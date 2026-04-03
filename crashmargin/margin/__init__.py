"""CrashMargin margin management module (Section 3.5 and Section 5.5).

Provides dynamic margin computation based on crash probabilities and
policy evaluation for comparing margin management strategies.
"""

from crashmargin.margin.dynamic_margin import DynamicMarginCalculator
from crashmargin.margin.policy_eval import MarginPolicySimulator

__all__ = [
    "DynamicMarginCalculator",
    "MarginPolicySimulator",
]
