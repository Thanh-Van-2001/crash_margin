"""
Tests for focal loss implementation (Section 3.4).

Validates the actual crashmargin.training.losses.FocalLoss:
    - Binary focal loss with sigmoid (not multiclass softmax)
    - Default: gamma=2.0, alpha=0.8 (paper Section 3.4)
    - Output is scalar and non-negative
    - Gradient flows correctly
    - gamma=0 degenerates to weighted binary cross-entropy
    - alpha weighting handles class imbalance
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from crashmargin.training.losses import FocalLoss


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestFocalLoss:
    @pytest.fixture
    def balanced_batch(self):
        torch.manual_seed(42)
        # Binary logits (B,) — matches CrashMarginModel output
        logits = torch.randn(32, requires_grad=True)
        targets = torch.cat([torch.zeros(16), torch.ones(16)])
        return logits, targets

    @pytest.fixture
    def imbalanced_batch(self):
        """~4.4% crash prevalence like in the paper."""
        torch.manual_seed(42)
        logits = torch.randn(256, requires_grad=True)
        targets = torch.zeros(256)
        targets[:11] = 1  # ~4.3%
        return logits, targets

    def test_default_params(self):
        """Default parameters match paper Section 3.4."""
        fl = FocalLoss()
        assert fl.gamma == 2.0
        assert fl.alpha == 0.8

    def test_output_is_scalar(self, balanced_batch):
        logits, targets = balanced_batch
        loss = FocalLoss()(logits, targets)
        assert loss.dim() == 0, "Loss should be scalar"

    def test_output_nonnegative(self, balanced_batch):
        logits, targets = balanced_batch
        loss = FocalLoss()(logits, targets)
        assert loss.item() >= 0.0, "Focal loss must be non-negative"

    def test_gradient_flows(self, balanced_batch):
        logits, targets = balanced_batch
        loss = FocalLoss()(logits, targets)
        loss.backward()
        assert logits.grad is not None, "Gradients did not flow to logits"
        assert not torch.all(logits.grad == 0), "All gradients are zero"

    def test_gamma_zero_is_weighted_bce(self):
        """Focal loss with gamma=0 should equal alpha-weighted BCE."""
        torch.manual_seed(42)
        logits = torch.randn(64, requires_grad=True)
        targets = torch.randint(0, 2, (64,)).float()

        alpha = 0.5
        fl = FocalLoss(alpha=alpha, gamma=0.0)(logits, targets)

        # Manual alpha-weighted BCE
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        expected = (alpha_t * bce).mean()

        assert torch.allclose(fl, expected, atol=1e-5), \
            f"Focal(gamma=0, alpha=0.5)={fl.item():.6f} vs weighted BCE={expected.item():.6f}"

    def test_higher_gamma_reduces_easy_examples(self):
        """Higher gamma should down-weight confident predictions."""
        torch.manual_seed(42)
        # Easy examples: high-confidence correct predictions
        logits = torch.tensor([5.0, 5.0, -5.0, -5.0])
        targets = torch.tensor([1.0, 1.0, 0.0, 0.0])
        loss_g0 = FocalLoss(gamma=0.0)(logits, targets)
        loss_g2 = FocalLoss(gamma=2.0)(logits, targets)
        loss_g5 = FocalLoss(gamma=5.0)(logits, targets)
        assert loss_g2 < loss_g0, "gamma=2 should give lower loss on easy examples"
        assert loss_g5 < loss_g2, "gamma=5 should give even lower loss"

    def test_alpha_weighting(self, imbalanced_batch):
        """Different alpha values should produce different losses."""
        logits, targets = imbalanced_batch
        loss_high_alpha = FocalLoss(alpha=0.9, gamma=0.0)(logits, targets)
        loss_low_alpha = FocalLoss(alpha=0.1, gamma=0.0)(logits, targets)
        assert not torch.allclose(loss_high_alpha, loss_low_alpha)

    def test_reduction_none(self, balanced_batch):
        logits, targets = balanced_batch
        loss = FocalLoss(reduction="none")(logits, targets)
        assert loss.shape == (32,), f"Expected per-sample loss, got {loss.shape}"

    def test_reduction_sum(self, balanced_batch):
        logits, targets = balanced_batch
        loss_sum = FocalLoss(reduction="sum")(logits, targets)
        loss_none = FocalLoss(reduction="none")(logits, targets)
        assert torch.allclose(loss_sum, loss_none.sum(), atol=1e-5)

    def test_invalid_params(self):
        with pytest.raises(ValueError):
            FocalLoss(gamma=-1.0)
        with pytest.raises(ValueError):
            FocalLoss(alpha=1.5)
        with pytest.raises(ValueError):
            FocalLoss(reduction="invalid")

    def test_repr(self):
        fl = FocalLoss()
        s = repr(fl)
        assert "gamma=2.0" in s
        assert "alpha=0.8" in s
