"""
Focal Loss for crash prediction (Section 3.4).

Addresses severe class imbalance in crash prediction (4.4% crash rate)
by down-weighting well-classified examples and focusing training on
hard, misclassified samples.

    L = -alpha * (1 - p_hat)^gamma * log(p_hat)

With gamma=2 and alpha=0.8 for the minority crash class (Section 3.4).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss for binary crash prediction (Section 3.4, Eq. in text).

    Focal loss reduces the contribution of easy negatives (non-crash days)
    and emphasizes hard positives (missed crash predictions). This is
    critical given the 4.4% crash rate in the Vietnamese market dataset.

    L = -alpha_t * (1 - p_t)^gamma * log(p_t)

    where:
        p_t = p_hat     if y = 1 (crash)
        p_t = 1 - p_hat if y = 0 (no crash)
        alpha_t = alpha  if y = 1
        alpha_t = 1 - alpha if y = 0

    Args:
        gamma: Focusing parameter that reduces loss for well-classified
            examples. Higher gamma means more focus on hard examples.
            Default: 2.0 (Section 3.4).
        alpha: Weight for the positive (crash) class to further address
            class imbalance. Default: 0.8 (Section 3.4).
        reduction: Specifies the reduction to apply to output.
            'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float = 0.8,
        reduction: str = "mean",
    ):
        super().__init__()
        if gamma < 0:
            raise ValueError(f"gamma must be non-negative, got {gamma}")
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if reduction not in ("none", "mean", "sum"):
            raise ValueError(f"reduction must be 'none', 'mean', or 'sum', got {reduction}")

        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.

        Args:
            inputs: Raw logits (before sigmoid) of shape (batch,) or (batch, 1).
                Can also be pre-sigmoid probabilities if they contain values
                outside [0, 1], which are then clamped.
            targets: Binary labels (0 or 1) of shape matching inputs.
                1 = crash, 0 = no crash.

        Returns:
            Focal loss scalar (if reduction='mean' or 'sum') or tensor of
            shape matching inputs (if reduction='none').
        """
        # Flatten if needed
        inputs = inputs.view(-1)
        targets = targets.view(-1).float()

        # Apply sigmoid to get probabilities
        p = torch.sigmoid(inputs)

        # Numerical stability: clamp probabilities
        p = torch.clamp(p, min=1e-7, max=1.0 - 1e-7)

        # Binary cross-entropy terms
        # For y=1: -log(p), for y=0: -log(1-p)
        ce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction="none"
        )

        # p_t: probability of the true class
        p_t = p * targets + (1 - p) * (1 - targets)

        # Alpha weighting: alpha for crash class, (1-alpha) for non-crash
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Focal modulating factor: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma

        # Final focal loss: -alpha_t * (1 - p_t)^gamma * log(p_t)
        loss = alpha_t * focal_weight * ce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss

    def __repr__(self) -> str:
        return (
            f"FocalLoss(gamma={self.gamma}, alpha={self.alpha}, "
            f"reduction='{self.reduction}')"
        )
