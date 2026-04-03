"""
Evaluation Metrics for CrashMargin (Section 4.3).

Classification metrics: AUROC (with DeLong test for significance),
F1 score, balanced accuracy, precision, and recall.

Economic metrics: average maximum portfolio loss during crash windows,
margin call frequency, and capital utilization efficiency.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def compute_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """Compute classification metrics for crash prediction (Section 4.3).

    Args:
        y_true: Binary ground truth labels (0/1), shape (n_samples,).
        y_prob: Predicted crash probabilities, shape (n_samples,).
        threshold: Decision threshold for converting probabilities to
            binary predictions. Default: 0.5.

    Returns:
        Dictionary containing:
            - auroc: Area under the ROC curve.
            - f1: F1 score.
            - balanced_accuracy: Mean of per-class recall.
            - precision: Positive predictive value.
            - recall: Sensitivity / true positive rate.
            - specificity: True negative rate.
            - n_positive: Number of positive (crash) samples.
            - n_negative: Number of negative (non-crash) samples.
    """
    y_true = np.asarray(y_true, dtype=np.int64).ravel()
    y_prob = np.asarray(y_prob, dtype=np.float64).ravel()

    # AUROC computation
    auroc = _compute_auroc(y_true, y_prob)

    # Binary predictions
    y_pred = (y_prob >= threshold).astype(np.int64)

    # Confusion matrix elements
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))

    # Precision
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    # Recall (sensitivity)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # Specificity
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # F1
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Balanced accuracy: mean of recall and specificity
    balanced_accuracy = (recall + specificity) / 2.0

    return {
        "auroc": float(auroc),
        "f1": float(f1),
        "balanced_accuracy": float(balanced_accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "n_positive": int(np.sum(y_true == 1)),
        "n_negative": int(np.sum(y_true == 0)),
    }


def _compute_auroc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute AUROC using the Mann-Whitney U statistic.

    Equivalent to sklearn.metrics.roc_auc_score but without the sklearn
    dependency. Uses the relationship: AUROC = U / (n_pos * n_neg).

    Args:
        y_true: Binary labels, shape (n_samples,).
        y_prob: Predicted probabilities, shape (n_samples,).

    Returns:
        AUROC value in [0, 1].
    """
    pos_scores = y_prob[y_true == 1]
    neg_scores = y_prob[y_true == 0]

    n_pos = len(pos_scores)
    n_neg = len(neg_scores)

    if n_pos == 0 or n_neg == 0:
        return 0.5  # undefined, return chance level

    # Mann-Whitney U statistic
    # Count how many positive scores exceed each negative score
    # Using sorted rank approach for O(n log n)
    all_scores = np.concatenate([pos_scores, neg_scores])
    all_labels = np.concatenate([np.ones(n_pos), np.zeros(n_neg)])

    # Sort by score
    order = np.argsort(all_scores)
    all_labels_sorted = all_labels[order]

    # Assign ranks (1-based), handling ties with average rank
    n = len(all_scores)
    ranks = np.empty(n, dtype=np.float64)
    all_scores_sorted = all_scores[order]

    i = 0
    while i < n:
        j = i
        while j < n and all_scores_sorted[j] == all_scores_sorted[i]:
            j += 1
        avg_rank = (i + j + 1) / 2.0  # 1-based average rank for the tie group
        ranks[i:j] = avg_rank
        i = j

    # Sum of ranks for positive class
    pos_rank_sum = np.sum(ranks[all_labels_sorted == 1])

    # AUROC = (R_pos - n_pos*(n_pos+1)/2) / (n_pos * n_neg)
    u_stat = pos_rank_sum - n_pos * (n_pos + 1) / 2.0
    auroc = u_stat / (n_pos * n_neg)

    return float(auroc)


def delong_test(
    auc1: float,
    auc2: float,
    y_true: np.ndarray,
    y_prob1: np.ndarray,
    y_prob2: np.ndarray,
) -> float:
    """DeLong test for comparing two AUROC values (Section 4.3).

    Tests the null hypothesis H0: AUC1 = AUC2 using the method of
    DeLong, DeLong, and Clarke-Pearson (1988). This non-parametric test
    accounts for the correlation between AUC estimates computed on the
    same dataset.

    Args:
        auc1: AUROC of model 1 (informational, recomputed internally).
        auc2: AUROC of model 2 (informational, recomputed internally).
        y_true: Binary ground truth labels, shape (n_samples,).
        y_prob1: Predicted probabilities from model 1, shape (n_samples,).
        y_prob2: Predicted probabilities from model 2, shape (n_samples,).

    Returns:
        Two-sided p-value for the null hypothesis AUC1 = AUC2.
        p < 0.05 indicates statistically significant difference.
    """
    y_true = np.asarray(y_true, dtype=np.int64).ravel()
    y_prob1 = np.asarray(y_prob1, dtype=np.float64).ravel()
    y_prob2 = np.asarray(y_prob2, dtype=np.float64).ravel()

    pos_mask = y_true == 1
    neg_mask = y_true == 0

    pos_scores1 = y_prob1[pos_mask]
    neg_scores1 = y_prob1[neg_mask]
    pos_scores2 = y_prob2[pos_mask]
    neg_scores2 = y_prob2[neg_mask]

    n_pos = len(pos_scores1)
    n_neg = len(neg_scores1)

    if n_pos < 2 or n_neg < 2:
        return 1.0  # cannot perform test

    # Structural components (placement values)
    # V_{10}(X_i): for each positive sample, fraction of negatives it exceeds
    v10_1 = np.array([np.mean(pos_scores1[i] > neg_scores1) +
                       0.5 * np.mean(pos_scores1[i] == neg_scores1)
                       for i in range(n_pos)])
    v10_2 = np.array([np.mean(pos_scores2[i] > neg_scores2) +
                       0.5 * np.mean(pos_scores2[i] == neg_scores2)
                       for i in range(n_pos)])

    # V_{01}(Y_j): for each negative sample, fraction of positives that exceed it
    v01_1 = np.array([np.mean(pos_scores1 > neg_scores1[j]) +
                       0.5 * np.mean(pos_scores1 == neg_scores1[j])
                       for j in range(n_neg)])
    v01_2 = np.array([np.mean(pos_scores2 > neg_scores2[j]) +
                       0.5 * np.mean(pos_scores2 == neg_scores2[j])
                       for j in range(n_neg)])

    # Covariance matrix of the two AUC estimates
    # S10: covariance from positive samples
    s10 = np.cov(v10_1, v10_2)  # 2x2 matrix

    # S01: covariance from negative samples
    s01 = np.cov(v01_1, v01_2)  # 2x2 matrix

    # Combined covariance of (AUC1, AUC2)
    S = s10 / n_pos + s01 / n_neg

    # Variance of AUC1 - AUC2
    var_diff = S[0, 0] + S[1, 1] - 2 * S[0, 1]

    if var_diff <= 0:
        return 1.0  # degenerate case

    # Recompute AUCs internally for consistency
    auc1_actual = _compute_auroc(y_true, y_prob1)
    auc2_actual = _compute_auroc(y_true, y_prob2)

    # Test statistic: z = (AUC1 - AUC2) / sqrt(var_diff)
    z = (auc1_actual - auc2_actual) / np.sqrt(var_diff)

    # Two-sided p-value
    p_value = 2.0 * stats.norm.sf(abs(z))

    return float(p_value)


def compute_economic_metrics(
    returns: np.ndarray,
    crash_probs: np.ndarray,
    crash_labels: np.ndarray,
    margin_ratios: np.ndarray,
    crash_threshold: float = -0.02,
) -> dict:
    """Compute economic evaluation metrics (Section 4.3, Table 3).

    Evaluates the economic impact of margin decisions on portfolio
    performance, particularly during crash periods.

    Args:
        returns: Daily portfolio returns, shape (n_days,).
        crash_probs: Predicted crash probabilities, shape (n_days,).
        crash_labels: True binary crash labels, shape (n_days,).
        margin_ratios: Applied margin ratios from the policy, shape (n_days,).
        crash_threshold: Return threshold for crash period identification.

    Returns:
        Dictionary containing:
            - avg_max_loss: Average of per-crash-window maximum losses.
            - margin_call_freq: Number of margin calls per 100 trading days.
            - capital_utilization: Mean portfolio utilization ratio.
            - loss_during_crash: Average leveraged loss during crash periods.
    """
    returns = np.asarray(returns, dtype=np.float64)
    crash_probs = np.asarray(crash_probs, dtype=np.float64)
    crash_labels = np.asarray(crash_labels, dtype=np.int64)
    margin_ratios = np.asarray(margin_ratios, dtype=np.float64)

    n_days = len(returns)

    # Leveraged returns: leverage = 1 / margin_ratio
    leverage = 1.0 / np.clip(margin_ratios, 0.01, 1.0)
    leveraged_returns = returns * leverage

    # Identify crash windows (contiguous blocks of crash days)
    crash_mask = crash_labels == 1
    crash_windows = _find_contiguous_blocks(crash_mask)

    # Per-window maximum loss
    window_max_losses = []
    for start, end in crash_windows:
        window_returns = leveraged_returns[start:end]
        # Cumulative return within window
        cum_return = np.cumprod(1.0 + window_returns) - 1.0
        max_loss = float(np.min(cum_return))
        window_max_losses.append(max_loss)

    avg_max_loss = float(np.mean(window_max_losses)) if window_max_losses else 0.0

    # Margin call frequency: days where leveraged loss exceeds maintenance
    maintenance_threshold = 0.30
    margin_call_days = np.sum(leveraged_returns < -maintenance_threshold)
    margin_call_freq = float(margin_call_days / n_days * 100) if n_days > 0 else 0.0

    # Capital utilization: inverse of margin ratio, normalized
    capital_utilization = float(np.mean(1.0 - margin_ratios + 0.30))
    capital_utilization = min(capital_utilization, 1.0)

    # Average loss during crash periods
    if crash_mask.any():
        loss_during_crash = float(np.mean(leveraged_returns[crash_mask]))
    else:
        loss_during_crash = 0.0

    return {
        "avg_max_loss": avg_max_loss,
        "margin_call_freq": margin_call_freq,
        "capital_utilization": capital_utilization,
        "loss_during_crash": loss_during_crash,
        "n_crash_windows": len(crash_windows),
        "n_crash_days": int(np.sum(crash_mask)),
    }


def _find_contiguous_blocks(mask: np.ndarray) -> list[tuple[int, int]]:
    """Find contiguous True blocks in a boolean array.

    Args:
        mask: Boolean array, shape (n,).

    Returns:
        List of (start, end) tuples for each contiguous block.
        End is exclusive (slice convention).
    """
    blocks = []
    in_block = False
    start = 0

    for i, val in enumerate(mask):
        if val and not in_block:
            start = i
            in_block = True
        elif not val and in_block:
            blocks.append((start, i))
            in_block = False

    if in_block:
        blocks.append((start, len(mask)))

    return blocks
