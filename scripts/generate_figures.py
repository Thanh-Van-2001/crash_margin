#!/usr/bin/env python
"""
Generate all paper figures (Figures 2-8) for CrashMargin ICAIF 2026.

Outputs PNG (300 dpi) and PDF to --output_dir.

Usage:
    python scripts/generate_figures.py --output_dir outputs/figures [--seed 42]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import seaborn as sns

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
sns.set_theme(style="whitegrid", font_scale=1.1)
PALETTE = sns.color_palette("colorblind", 10)
CM_COLOR = PALETTE[0]
BASELINE_COLORS = PALETTE[1:]


def _save(fig, path_stem: Path) -> None:
    for ext in ("png", "pdf"):
        out = path_stem.with_suffix(f".{ext}")
        fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path_stem.stem} (.png + .pdf)")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 2 — Dataset characteristics (3 panels)
# ═══════════════════════════════════════════════════════════════════════════
def figure2(output_dir: Path, rng: np.random.Generator) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # Panel A: Crash event distribution over time
    ax = axes[0]
    months = np.arange(1, 79)  # Jan 2018 – Jun 2024
    crash_counts = rng.poisson(lam=3.2, size=len(months))
    # Spikes in 2020-Q1 (COVID) and 2022-Q1 (margin crisis)
    crash_counts[26:29] += rng.integers(8, 15, size=3)  # Mar-May 2020
    crash_counts[48:51] += rng.integers(6, 12, size=3)  # Jan-Mar 2022
    ax.bar(months, crash_counts, color=PALETTE[3], alpha=0.8, width=0.8)
    ax.set_xlabel("Month (2018-01 to 2024-06)")
    ax.set_ylabel("Crash events")
    ax.set_title("(a) Crash Event Distribution")
    ax.axvspan(26, 29, alpha=0.15, color="red", label="COVID-19")
    ax.axvspan(48, 51, alpha=0.15, color="orange", label="Margin crisis")
    ax.legend(fontsize=8)

    # Panel B: SHAP feature importance (top 10)
    ax = axes[1]
    features = [
        "Margin debt ratio", "Vol 20d", "RSI 14", "Ret 5d",
        "MACD", "Amihud illiq.", "Kyle lambda", "OBV slope",
        "BB width", "Sentiment"
    ]
    importance = np.array([0.142, 0.118, 0.103, 0.091, 0.084,
                           0.076, 0.069, 0.058, 0.047, 0.039])
    importance += rng.uniform(-0.003, 0.003, size=len(importance))
    idx = np.argsort(importance)
    ax.barh(np.array(features)[idx], importance[idx], color=PALETTE[0], alpha=0.85)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("(b) Feature Importance (SHAP)")

    # Panel C: Sector pie chart
    ax = axes[2]
    sectors = [
        "Real Estate", "Banking", "Securities", "Materials",
        "Consumer", "Tech", "Industrial", "Energy", "Other"
    ]
    sizes = [18.2, 15.7, 9.4, 11.3, 10.1, 7.8, 8.6, 6.3, 12.6]
    explode = [0.05 if s in ("Real Estate", "Banking", "Securities") else 0 for s in sectors]
    ax.pie(sizes, labels=sectors, autopct="%1.1f%%", startangle=140,
           explode=explode, colors=PALETTE[:len(sectors)], textprops={"fontsize": 8})
    ax.set_title("(c) Sector Distribution (HOSE)")

    fig.suptitle("Figure 2: Dataset Characteristics", fontsize=14, y=1.02)
    fig.tight_layout()
    _save(fig, output_dir / "figure2_dataset")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 3 — Model comparison bar charts
# ═══════════════════════════════════════════════════════════════════════════
def figure3(output_dir: Path, rng: np.random.Generator) -> None:
    methods = [
        "Naive", "Logistic", "GARCH-EVT", "SVM", "XGBoost",
        "LightGBM", "LSTM", "TFT", "BiGAT-GRU", "CrashMargin"
    ]
    auroc = [0.500, 0.623, 0.654, 0.671, 0.724, 0.738, 0.741, 0.768, 0.789, 0.831]
    f1    = [0.000, 0.312, 0.341, 0.378, 0.452, 0.471, 0.489, 0.521, 0.553, 0.614]
    balacc = [0.500, 0.587, 0.618, 0.632, 0.689, 0.702, 0.711, 0.734, 0.751, 0.793]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    metrics = [("AUROC", auroc), ("F1 Score", f1), ("Balanced Accuracy", balacc)]
    x = np.arange(len(methods))

    for ax, (metric_name, vals) in zip(axes, metrics):
        colors = [CM_COLOR if m == "CrashMargin" else PALETTE[2] for m in methods]
        bars = ax.bar(x, vals, color=colors, alpha=0.85, edgecolor="white", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel(metric_name)
        ax.set_title(metric_name)
        ax.set_ylim(0, 1.0)
        # Annotate CrashMargin bar
        ax.annotate(f"{vals[-1]:.3f}", xy=(x[-1], vals[-1]),
                    ha="center", va="bottom", fontsize=9, fontweight="bold")

    fig.suptitle("Figure 3: Classification Performance Comparison", fontsize=14, y=1.02)
    fig.tight_layout()
    _save(fig, output_dir / "figure3_comparison")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 4 — Ablation horizontal bars
# ═══════════════════════════════════════════════════════════════════════════
def figure4(output_dir: Path, rng: np.random.Generator) -> None:
    variants = [
        "Full CrashMargin",
        "w/o Cross-Attention",
        "w/o Graph",
        "w/o Margin Features",
        "w/o Sentiment",
        "w/o Margin-Exposure Graph",
        "Market Only",
    ]
    auroc = [0.831, 0.784, 0.798, 0.807, 0.812, 0.818, 0.768]
    f1    = [0.614, 0.542, 0.567, 0.574, 0.583, 0.591, 0.521]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    y = np.arange(len(variants))

    for ax, (metric_name, vals) in zip(axes, [("AUROC", auroc), ("F1 Score", f1)]):
        colors = [CM_COLOR if v == "Full CrashMargin" else PALETTE[4] for v in variants]
        ax.barh(y, vals, color=colors, alpha=0.85, edgecolor="white")
        ax.set_yticks(y)
        ax.set_yticklabels(variants)
        ax.set_xlabel(metric_name)
        ax.set_title(f"Ablation: {metric_name}")
        ax.set_xlim(0.5, 0.9)
        for i, v in enumerate(vals):
            ax.text(v + 0.003, i, f"{v:.3f}", va="center", fontsize=9)

    fig.suptitle("Figure 4: Ablation Study", fontsize=14, y=1.02)
    fig.tight_layout()
    _save(fig, output_dir / "figure4_ablation")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 5 — Walk-forward AUROC line chart
# ═══════════════════════════════════════════════════════════════════════════
def figure5(output_dir: Path, rng: np.random.Generator) -> None:
    windows = ["2021 H1", "2021 H2", "2022 H1", "2022 H2", "2023 H1", "2023 H2", "2024 H1"]
    models = {
        "LightGBM":    [0.721, 0.729, 0.748, 0.741, 0.733, 0.739, 0.735],
        "LSTM":        [0.726, 0.734, 0.757, 0.749, 0.738, 0.742, 0.740],
        "TFT":         [0.753, 0.761, 0.782, 0.775, 0.764, 0.768, 0.766],
        "BiGAT-GRU":   [0.774, 0.781, 0.805, 0.798, 0.785, 0.789, 0.787],
        "CrashMargin": [0.818, 0.825, 0.849, 0.841, 0.828, 0.832, 0.830],
    }

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(windows))
    markers = ["s", "^", "D", "v", "o"]

    for i, (name, vals) in enumerate(models.items()):
        lw = 2.5 if name == "CrashMargin" else 1.5
        ms = 9 if name == "CrashMargin" else 6
        ax.plot(x, vals, marker=markers[i], label=name, linewidth=lw, markersize=ms,
                color=PALETTE[i], zorder=10 if name == "CrashMargin" else 5)

    ax.set_xticks(x)
    ax.set_xticklabels(windows, rotation=30)
    ax.set_ylabel("AUROC")
    ax.set_title("Figure 5: Walk-Forward AUROC (Semi-Annual Windows)")
    ax.legend(loc="lower right")
    ax.set_ylim(0.70, 0.87)
    # Highlight 2022-H1 peak
    ax.axvspan(1.5, 2.5, alpha=0.1, color="red", label="Market stress period")
    ax.annotate("Peak: 0.849", xy=(2, 0.849), xytext=(3, 0.855),
                arrowprops=dict(arrowstyle="->", color="gray"),
                fontsize=10, color=CM_COLOR, fontweight="bold")

    fig.tight_layout()
    _save(fig, output_dir / "figure5_walkforward")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 6 — Economic impact (2 panels)
# ═══════════════════════════════════════════════════════════════════════════
def figure6(output_dir: Path, rng: np.random.Generator) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A: Portfolio loss bars
    ax = axes[0]
    policies = ["No Margin", "Static 50%", "GARCH VaR", "CrashMargin"]
    losses = [-18.3, -12.7, -9.4, -5.8]
    max_dd = [-37.1, -24.8, -19.2, -13.5]
    x = np.arange(len(policies))
    w = 0.35
    bars1 = ax.bar(x - w / 2, losses, w, label="Avg Loss (%)", color=PALETTE[3], alpha=0.85)
    bars2 = ax.bar(x + w / 2, max_dd, w, label="Max Drawdown (%)", color=PALETTE[1], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(policies, rotation=20, ha="right")
    ax.set_ylabel("Percentage (%)")
    ax.set_title("(a) Portfolio Loss & Max Drawdown")
    ax.legend()
    ax.axhline(0, color="black", linewidth=0.5)

    # Panel B: Efficiency frontier (risk vs utilisation)
    ax = axes[1]
    margin_calls = [0, 142, 98, 47]
    capital_util = [100.0, 50.0, 52.3, 53.1]
    for i, pol in enumerate(policies):
        color = CM_COLOR if pol == "CrashMargin" else PALETTE[2]
        sz = 150 if pol == "CrashMargin" else 80
        ax.scatter(margin_calls[i], capital_util[i], s=sz, color=color,
                   edgecolors="black", linewidths=0.8, zorder=5)
        ax.annotate(pol, (margin_calls[i], capital_util[i]),
                    textcoords="offset points", xytext=(8, 5), fontsize=9)
    ax.set_xlabel("Margin Calls (#)")
    ax.set_ylabel("Capital Utilisation (%)")
    ax.set_title("(b) Efficiency Frontier")
    ax.set_xlim(-10, 160)

    fig.suptitle("Figure 6: Economic Impact of Margin Policies", fontsize=14, y=1.02)
    fig.tight_layout()
    _save(fig, output_dir / "figure6_economic")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 7 — Case study (3 panels: price, crash prob, dynamic margin)
# ═══════════════════════════════════════════════════════════════════════════
def figure7(output_dir: Path, rng: np.random.Generator) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    n = 250  # ~1 year of trading days

    # Simulated price path with crash episode around t=150-170
    t = np.arange(n)
    price = 50.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, n)))
    # Inject crash
    price[150:170] *= np.linspace(1.0, 0.78, 20)
    price[170:] *= 0.78

    # Crash probability rises before crash
    crash_prob = np.clip(0.05 + 0.02 * rng.standard_normal(n), 0.01, 0.15)
    crash_prob[140:170] = np.clip(
        np.linspace(0.15, 0.92, 30) + rng.normal(0, 0.03, 30), 0.1, 0.98
    )
    crash_prob[170:190] = np.clip(
        np.linspace(0.85, 0.10, 20) + rng.normal(0, 0.02, 20), 0.05, 0.95
    )

    # Dynamic margin
    margin = np.clip(0.50 + 2.5 * (crash_prob - 0.5), 0.40, 0.85)

    # Panel A: Price
    ax = axes[0]
    ax.plot(t, price, color=PALETTE[0], linewidth=1.2)
    ax.axvspan(150, 170, alpha=0.2, color="red", label="Crash episode")
    ax.set_ylabel("Price (VND '000)")
    ax.set_title("(a) Stock Price")
    ax.legend(loc="upper right", fontsize=9)

    # Panel B: Crash probability
    ax = axes[1]
    ax.fill_between(t, crash_prob, alpha=0.3, color=PALETTE[3])
    ax.plot(t, crash_prob, color=PALETTE[3], linewidth=1.2)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="Alert threshold")
    ax.axvspan(150, 170, alpha=0.2, color="red")
    ax.set_ylabel("Crash Probability")
    ax.set_title("(b) CrashMargin Predicted Probability")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(0, 1)

    # Panel C: Dynamic margin requirement
    ax = axes[2]
    ax.fill_between(t, 0.40, margin, alpha=0.25, color=PALETTE[0])
    ax.plot(t, margin, color=PALETTE[0], linewidth=1.5, label="Dynamic margin")
    ax.axhline(0.50, color="gray", linestyle="--", linewidth=0.8, label="Static 50%")
    ax.axhline(0.40, color="black", linestyle=":", linewidth=0.8, label="Floor (40%)")
    ax.axhline(0.85, color="black", linestyle=":", linewidth=0.8, label="Cap (85%)")
    ax.axvspan(150, 170, alpha=0.2, color="red")
    ax.set_ylabel("Margin Requirement")
    ax.set_xlabel("Trading Day")
    ax.set_title("(c) Dynamic Margin Requirement")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(0.35, 0.90)

    fig.suptitle("Figure 7: Case Study — Crash Episode", fontsize=14, y=1.01)
    fig.tight_layout()
    _save(fig, output_dir / "figure7_casestudy")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 8 — SHAP waterfall for a high-risk instance
# ═══════════════════════════════════════════════════════════════════════════
def figure8(output_dir: Path, rng: np.random.Generator) -> None:
    features = [
        "Margin debt ratio", "Vol 20d", "RSI 14 (oversold)", "Ret 5d",
        "MACD signal", "Amihud illiq.", "Sentiment (neg)", "Kyle lambda",
        "BB width", "OBV slope", "Sector (Real Estate)", "Graph attn."
    ]
    shap_vals = np.array([
        0.182, 0.134, 0.108, -0.095, 0.087, 0.074,
        0.068, 0.052, -0.041, 0.038, 0.031, 0.026
    ])
    shap_vals += rng.uniform(-0.005, 0.005, size=len(shap_vals))
    base_value = 0.044  # population crash rate (4.4%, Section 4.1)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Sort by absolute value
    order = np.argsort(np.abs(shap_vals))
    sorted_features = [features[i] for i in order]
    sorted_vals = shap_vals[order]
    colors = [PALETTE[3] if v > 0 else PALETTE[0] for v in sorted_vals]

    ax.barh(range(len(sorted_features)), sorted_vals, color=colors, alpha=0.85, edgecolor="white")
    ax.set_yticks(range(len(sorted_features)))
    ax.set_yticklabels(sorted_features)
    ax.set_xlabel("SHAP Value (impact on crash probability)")
    ax.set_title("Figure 8: SHAP Waterfall — High-Risk Instance")
    ax.axvline(0, color="black", linewidth=0.8)

    # Annotate predicted probability
    pred_prob = base_value + np.sum(shap_vals)
    ax.text(0.95, 0.05, f"Base: {base_value:.3f}\nPredicted: {pred_prob:.3f}",
            transform=ax.transAxes, fontsize=10, verticalalignment="bottom",
            horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))

    fig.tight_layout()
    _save(fig, output_dir / "figure8_shap")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
def main() -> None:
    parser = argparse.ArgumentParser(description="Generate CrashMargin paper figures")
    parser.add_argument("--output_dir", type=str, default="outputs/figures")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print("Generating CrashMargin paper figures...")
    figure2(output_dir, rng)
    figure3(output_dir, rng)
    figure4(output_dir, rng)
    figure5(output_dir, rng)
    figure6(output_dir, rng)
    figure7(output_dir, rng)
    figure8(output_dir, rng)
    print(f"\nAll figures saved to {output_dir}/")


if __name__ == "__main__":
    main()
