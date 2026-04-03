"""
Tests for the CrashMargin model forward pass (Section 3.3).

Validates against the actual CrashMarginModel architecture:
    - 4-modality inputs: market (B,20,47), sentiment (B,20,4),
      graph (N, node_dim) + dual adjacency, margin (B,7)
    - Output: raw logit of shape (batch,)
    - Sigmoid of logit is in [0, 1]
    - Gradient flow through all encoders
"""

from __future__ import annotations

import pytest
import torch

from crashmargin.model.crashmargin import CrashMarginModel


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
NUM_STOCKS = 20  # small graph for testing


@pytest.fixture
def model():
    torch.manual_seed(42)
    return CrashMarginModel(
        num_market_features=47,
        num_sentiment_features=4,
        num_margin_features=7,
        node_feature_dim=25,  # 20 sectors + 5 quintile bins
        hidden_dim=128,
        num_sectors=20,
        tft_num_heads=3,
        gat_num_heads=4,
        lookback=20,
        dropout=0.1,
        fusion_dropout=0.2,
    )


@pytest.fixture
def sample_batch():
    torch.manual_seed(42)
    batch_size = 8
    return {
        "market": torch.randn(batch_size, 20, 47),
        "sentiment": torch.randn(batch_size, 20, 4),
        "graph_features": torch.randn(NUM_STOCKS, 25),
        "adjacency": {
            "industry": torch.rand(NUM_STOCKS, NUM_STOCKS),
            "margin": torch.rand(NUM_STOCKS, NUM_STOCKS),
        },
        "margin_features": torch.randn(batch_size, 7),
        "sector_onehot": torch.zeros(batch_size, 20),
        "size_quintile": torch.zeros(batch_size, 1),
        "stock_indices": torch.arange(batch_size),
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestCrashMarginForward:
    def test_forward_runs(self, model, sample_batch):
        logit = model(**sample_batch)
        assert logit is not None

    def test_output_shape(self, model, sample_batch):
        logit = model(**sample_batch)
        assert logit.shape == (8,), f"Expected (8,), got {logit.shape}"

    def test_sigmoid_in_range(self, model, sample_batch):
        """Sigmoid of logit should be in [0, 1]."""
        logit = model(**sample_batch)
        prob = torch.sigmoid(logit)
        assert torch.all(prob >= 0.0), "Probabilities must be >= 0"
        assert torch.all(prob <= 1.0), "Probabilities must be <= 1"

    def test_batch_size_one(self, model):
        torch.manual_seed(0)
        batch = {
            "market": torch.randn(1, 20, 47),
            "sentiment": torch.randn(1, 20, 4),
            "graph_features": torch.randn(NUM_STOCKS, 25),
            "adjacency": {
                "industry": torch.rand(NUM_STOCKS, NUM_STOCKS),
                "margin": torch.rand(NUM_STOCKS, NUM_STOCKS),
            },
            "margin_features": torch.randn(1, 7),
            "stock_indices": torch.tensor([0]),
        }
        logit = model(**batch)
        assert logit.shape == (1,)

    def test_grad_flows(self, model, sample_batch):
        """Verify gradient flow through all four encoders."""
        logit = model(**sample_batch)
        loss = logit.sum()
        loss.backward()

        encoder_names = [
            "market_encoder", "sentiment_encoder",
            "graph_encoder", "margin_encoder",
            "fusion", "prediction_head",
        ]
        for enc_name in encoder_names:
            enc = getattr(model, enc_name)
            params_with_grad = 0
            total_params = 0
            for name, param in enc.named_parameters():
                if param.requires_grad:
                    total_params += 1
                    assert param.grad is not None, (
                        f"No gradient for {enc_name}.{name}"
                    )
                    if not torch.all(param.grad == 0):
                        params_with_grad += 1
            if total_params > 0:
                assert params_with_grad > 0, (
                    f"No non-zero gradients in {enc_name}"
                )

    def test_optional_static_covariates(self, model):
        """Model should work without sector_onehot and size_quintile."""
        torch.manual_seed(0)
        batch = {
            "market": torch.randn(4, 20, 47),
            "sentiment": torch.randn(4, 20, 4),
            "graph_features": torch.randn(NUM_STOCKS, 25),
            "adjacency": {
                "industry": torch.rand(NUM_STOCKS, NUM_STOCKS),
                "margin": torch.rand(NUM_STOCKS, NUM_STOCKS),
            },
            "margin_features": torch.randn(4, 7),
            "stock_indices": torch.arange(4),
        }
        logit = model(**batch)
        assert logit.shape == (4,)

    def test_no_stock_indices(self, model):
        """Without stock_indices, model takes first batch_size nodes."""
        torch.manual_seed(0)
        batch_size = 4
        batch = {
            "market": torch.randn(batch_size, 20, 47),
            "sentiment": torch.randn(batch_size, 20, 4),
            "graph_features": torch.randn(NUM_STOCKS, 25),
            "adjacency": {
                "industry": torch.rand(NUM_STOCKS, NUM_STOCKS),
                "margin": torch.rand(NUM_STOCKS, NUM_STOCKS),
            },
            "margin_features": torch.randn(batch_size, 7),
        }
        logit = model(**batch)
        assert logit.shape == (batch_size,)

    def test_deterministic_with_eval_mode(self, model, sample_batch):
        """Same input in eval mode should produce identical output."""
        model.eval()
        with torch.no_grad():
            out1 = model(**sample_batch)
            out2 = model(**sample_batch)
        torch.testing.assert_close(out1, out2)
