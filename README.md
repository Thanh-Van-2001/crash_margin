# CrashMargin

**A Multimodal Framework for Stock-Level Crash Forecasting with Dynamic Margin Risk Management**

*ICAIF 2026*

## Abstract

CrashMargin is a multimodal deep learning framework that (1) predicts stock-level
crash probability by fusing market microstructure data, PhoBERT-based Vietnamese
news sentiment, industry contagion graphs, and margin lending statistics;
(2) translates crash forecasts into dynamic, stock-level margin requirements via
a calibrated sigmoid mapping; and (3) provides SHAP-based risk decomposition for
regulatory transparency. Evaluated on 95 Vietnamese stocks (HOSE/HNX, 2018-2024)
with walk-forward retraining, CrashMargin achieves AUROC 0.831 and F1 0.614,
improving over the strongest baseline (BiGAT-GRU) by 5.3% AUROC and 11.0% F1.
Under simulated margin policy evaluation, the dynamic margin approach reduces
forced-liquidation losses by 54.3% relative to static margins while maintaining
comparable capital utilization.

## Architecture

```
Market Microstructure (20d, 47 TIs)        Vietnamese News Sentiment (PhoBERT)
        |                                         |
   TFT Encoder                              BiLSTM Encoder
   (3 heads, hidden=128, GRN)               (2-layer, hidden=64, dropout=0.15)
        |                                         |
Industry Graph + Margin-Exposure Graph      Margin Lending Data (7 features)
        |                                         |
   Dual-Graph GAT                           Margin Feature MLP
   (2-layer, 4 heads, hidden=64)            (FC 7->64->128)
        |                                         |
        +----> Cross-Modal Attention Fusion <-----+
                     |
              2-layer MLP (256, 128)
                     |
               Crash Logit -> sigmoid -> p_hat
                     |
         Dynamic Margin (Eq. 1, Section 3.5):
         m* = m_min + (m_max - m_min) * sigma((p - tau) / T)
         m_min=0.40, m_max=0.85, tau=0.15, T=0.1
```

## Installation

```bash
# Clone
git clone https://github.com/<your-org>/crashmargin.git
cd crashmargin

# Create environment
conda create -n crashmargin python=3.11 -y
conda activate crashmargin

# Install
pip install -e ".[dev]"
# or
pip install -r requirements.txt
```

## Experiments

```bash
# Table 1: Classification comparison (10 methods)
python experiments/run_classification.py --seed 42

# Table 2: Ablation study (7 variants)
python experiments/run_ablation.py --seed 42

# Table 3: Margin policy evaluation (4 policies)
python experiments/run_margin_eval.py --seed 42

# Figure 5: Walk-forward AUROC (7 windows)
python experiments/run_walkforward.py --seed 42

# All figures (Figures 2-8)
python scripts/generate_figures.py --output_dir outputs/figures

# Run all
make all
```

## Results

### Table 1: Classification Results (2022-2024)

| Method       | AUROC | F1    | Bal.Acc | Precision | Recall | DeLong p |
|:-------------|------:|------:|--------:|----------:|-------:|---------:|
| Naive        | 0.500 | 0.000 |   0.500 |       --- |    --- |   <0.001 |
| Logistic     | 0.623 | 0.312 |   0.587 |     0.401 |  0.255 |   <0.001 |
| GARCH-EVT    | 0.654 | 0.341 |   0.618 |     0.423 |  0.286 |   <0.001 |
| SVM          | 0.671 | 0.378 |   0.632 |     0.445 |  0.328 |   <0.001 |
| XGBoost      | 0.724 | 0.452 |   0.689 |     0.512 |  0.404 |   <0.001 |
| LightGBM     | 0.738 | 0.471 |   0.702 |     0.524 |  0.428 |   <0.001 |
| LSTM         | 0.741 | 0.489 |   0.711 |     0.534 |  0.451 |   <0.001 |
| TFT          | 0.768 | 0.521 |   0.734 |     0.567 |  0.483 |   <0.001 |
| BiGAT-GRU    | 0.789 | 0.553 |   0.751 |     0.598 |  0.515 |    0.003 |
| **CrashMargin** | **0.831** | **0.614** | **0.793** | **0.651** | **0.582** | --- |

### Table 2: Ablation Study

| Variant                  | AUROC | F1    | Delta AUROC |
|:-------------------------|------:|------:|------------:|
| Full CrashMargin         | 0.831 | 0.614 |         --- |
| w/o Graph                | 0.798 | 0.567 |      -0.033 |
| w/o Margin Features      | 0.807 | 0.574 |      -0.024 |
| w/o Margin-Exposure Graph| 0.818 | 0.591 |      -0.013 |
| w/o Sentiment            | 0.812 | 0.583 |      -0.019 |
| w/o Cross-Attention      | 0.784 | 0.542 |      -0.047 |
| Market Only (TFT)        | 0.768 | 0.521 |      -0.063 |

### Table 3: Margin Policy Evaluation (2022-2024)

| Policy       | Avg. Loss | Max Loss | Margin Calls | Capital Util. |
|:-------------|:---------:|:--------:|:------------:|:-------------:|
| No Margin    |   -18.3%  |  -37.1%  |     N/A      |     100%      |
| Static 50%   |   -12.7%  |  -24.8%  |     142      |     50.0%     |
| GARCH VaR 99%|    -9.4%  |  -19.2%  |      98      |     52.3%     |
| **CrashMargin** | **-5.8%** | **-13.5%** | **47** |   **53.1%**   |

## Tests

```bash
pytest tests/ -v
```

## Citation

```bibtex
@inproceedings{crashmargin2026,
  title     = {{CrashMargin}: A Multimodal Framework for Stock-Level Crash
               Forecasting with Dynamic Margin Risk Management},
  author    = {Anonymous},
  booktitle = {Proceedings of the 7th ACM International Conference on
               AI in Finance (ICAIF)},
  year      = {2026},
}
```

## License

MIT
