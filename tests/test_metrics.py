"""
Tests for evaluation metrics: AUROC, F1, DeLong test.

Validates:
    - AUROC matches sklearn
    - F1 handles edge cases (all-zero predictions)
    - DeLong test returns valid p-values
    - Metrics are deterministic
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats


# ---------------------------------------------------------------------------
# Metric functions (mirrors crashmargin.metrics)
# ---------------------------------------------------------------------------
def compute_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Area Under the ROC Curve."""
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y_true, y_score))


def compute_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    from sklearn.metrics import f1_score
    return float(f1_score(y_true, y_pred, zero_division=0))


def compute_balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    from sklearn.metrics import balanced_accuracy_score
    return float(balanced_accuracy_score(y_true, y_pred))


def delong_test(
    y_true: np.ndarray,
    y_score_a: np.ndarray,
    y_score_b: np.ndarray,
) -> dict[str, float]:
    """DeLong test for comparing two AUROCs on same sample.

    Returns z-statistic and two-sided p-value.
    """
    n = len(y_true)
    assert len(y_score_a) == n and len(y_score_b) == n

    auc_a = compute_auroc(y_true, y_score_a)
    auc_b = compute_auroc(y_true, y_score_b)

    # Placement values for structural components
    pos_mask = y_true == 1
    neg_mask = y_true == 0
    n_pos = pos_mask.sum()
    n_neg = neg_mask.sum()

    if n_pos == 0 or n_neg == 0:
        return {"z": 0.0, "p_value": 1.0, "auc_a": auc_a, "auc_b": auc_b}

    # Compute variance via Mann-Whitney U decomposition
    def _placement_values(y_true_arr, scores):
        """Compute placement values V10 and V01."""
        pos_scores = scores[pos_mask]
        neg_scores = scores[neg_mask]
        v10 = np.array([np.mean(s > neg_scores) for s in pos_scores])
        v01 = np.array([np.mean(pos_scores > s) for s in neg_scores])
        return v10, v01

    v10_a, v01_a = _placement_values(y_true, y_score_a)
    v10_b, v01_b = _placement_values(y_true, y_score_b)

    # Covariance matrix of (AUC_a, AUC_b)
    s10 = np.cov(v10_a, v10_b)
    s01 = np.cov(v01_a, v01_b)
    S = s10 / n_pos + s01 / n_neg

    # Variance of difference
    var_diff = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    if var_diff < 1e-15:
        return {"z": 0.0, "p_value": 1.0, "auc_a": auc_a, "auc_b": auc_b}

    z = (auc_a - auc_b) / np.sqrt(var_diff)
    p_value = 2.0 * stats.norm.sf(abs(z))

    return {"z": float(z), "p_value": float(p_value), "auc_a": auc_a, "auc_b": auc_b}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def binary_data():
    rng = np.random.default_rng(42)
    n = 1000
    y_true = rng.binomial(1, 0.044, size=n)  # ~4.4% prevalence (Section 4.1)
    # Good model: higher scores for positives
    y_score = rng.normal(0, 1, size=n) + y_true * 2.0
    y_score = 1.0 / (1.0 + np.exp(-y_score))  # sigmoid
    return y_true, y_score


@pytest.fixture
def two_model_data():
    rng = np.random.default_rng(42)
    n = 2000
    y_true = rng.binomial(1, 0.05, size=n)
    # Model A: better
    y_score_a = rng.normal(0, 1, size=n) + y_true * 2.5
    y_score_a = 1.0 / (1.0 + np.exp(-y_score_a))
    # Model B: worse
    y_score_b = rng.normal(0, 1, size=n) + y_true * 1.5
    y_score_b = 1.0 / (1.0 + np.exp(-y_score_b))
    return y_true, y_score_a, y_score_b


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestAUROC:
    def test_perfect_classifier(self):
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_score = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        assert compute_auroc(y_true, y_score) == pytest.approx(1.0)

    def test_random_classifier(self):
        rng = np.random.default_rng(42)
        y_true = rng.binomial(1, 0.5, size=10000)
        y_score = rng.random(10000)
        auroc = compute_auroc(y_true, y_score)
        assert 0.48 < auroc < 0.52, f"Random classifier AUROC should be ~0.5, got {auroc}"

    def test_auroc_in_range(self, binary_data):
        y_true, y_score = binary_data
        auroc = compute_auroc(y_true, y_score)
        assert 0.0 <= auroc <= 1.0

    def test_auroc_deterministic(self, binary_data):
        y_true, y_score = binary_data
        a1 = compute_auroc(y_true, y_score)
        a2 = compute_auroc(y_true, y_score)
        assert a1 == a2


class TestF1:
    def test_perfect_predictions(self):
        y_true = np.array([0, 1, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 1, 1])
        assert compute_f1(y_true, y_pred) == pytest.approx(1.0)

    def test_all_wrong(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 1, 0, 0])
        assert compute_f1(y_true, y_pred) == pytest.approx(0.0)

    def test_all_zero_predictions(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 0, 0])
        f1 = compute_f1(y_true, y_pred)
        assert f1 == pytest.approx(0.0), "All-zero predictions should give F1=0"

    def test_f1_in_range(self, binary_data):
        y_true, y_score = binary_data
        y_pred = (y_score >= 0.5).astype(int)
        f1 = compute_f1(y_true, y_pred)
        assert 0.0 <= f1 <= 1.0


class TestDeLong:
    def test_same_model_p_value(self, binary_data):
        """Comparing a model with itself should give p ~= 1."""
        y_true, y_score = binary_data
        result = delong_test(y_true, y_score, y_score)
        assert result["p_value"] == pytest.approx(1.0, abs=0.05)

    def test_different_models_significant(self, two_model_data):
        """Significantly different models should yield small p-value."""
        y_true, y_score_a, y_score_b = two_model_data
        result = delong_test(y_true, y_score_a, y_score_b)
        assert result["p_value"] < 0.05, \
            f"Expected significant difference, got p={result['p_value']:.4f}"

    def test_p_value_in_range(self, two_model_data):
        y_true, y_score_a, y_score_b = two_model_data
        result = delong_test(y_true, y_score_a, y_score_b)
        assert 0.0 <= result["p_value"] <= 1.0

    def test_symmetry(self, two_model_data):
        """DeLong(A,B) p-value should equal DeLong(B,A) p-value."""
        y_true, y_score_a, y_score_b = two_model_data
        r_ab = delong_test(y_true, y_score_a, y_score_b)
        r_ba = delong_test(y_true, y_score_b, y_score_a)
        assert r_ab["p_value"] == pytest.approx(r_ba["p_value"], abs=1e-10)

    def test_output_keys(self, two_model_data):
        y_true, y_score_a, y_score_b = two_model_data
        result = delong_test(y_true, y_score_a, y_score_b)
        assert set(result.keys()) == {"z", "p_value", "auc_a", "auc_b"}
