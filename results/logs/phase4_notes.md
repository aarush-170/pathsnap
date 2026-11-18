# Phase 4 notes: model comparison

Reproduce with `python -m scripts.train_models` (about 1 hour on a laptop CPU). Numbers come from
`model_comparison.csv` and `model_cv_folds.csv`. Figures: `results/figures/model_*.png`.

## Protocol (identical for all models)
- **Training data:** 1,336,027 candidate rows from labelled points on the 80 train trips, 26 features, target `is_true`.
- **Cross-validation:** 5-fold GroupKFold by trip, 16 trips per fold (`cv_folds.csv`). No trip ever appears in both the training and validation part of a fold.
- **Final evaluation:** a refit on all 80 train trips, then one evaluation on the 10 student_test trips (149,450 rows, 19,133 points).
- **Metrics:**
  - Observation-level **Top-1**: the highest-probability candidate per GPS point, compared with its label.
  - Row-level Macro F1, precision and recall at a 0.5 threshold.
  - Log loss and ROC-AUC.
  - **Route continuity**: the share of road changes in the Top-1 sequence that are physically connected.
  - Fit time, and inference time per 1,000 points.
- **Overfitting check:** each fold also scores a fixed sample of about 200k training rows.
- All configurations are untuned starting points (`results/models/<model>/metrics.json`). Seed 42.

## Results
| Model | Family | CV Top-1 (mean ± std) | CV Macro F1 | CV log loss | Test Top-1 | Test Macro F1 | Test log loss | Test continuity | Full fit (s) | Infer (ms / 1k pts) |
|---|---|---|---|---|---|---|---|---|---|---|
| XGBoost | Boosting | 0.8909 ± 0.0386 | 0.9359 | 0.0792 | **0.8941** | **0.9383** | 0.0689 | 0.828 | 34 | 14 |
| **LightGBM** | Boosting | **0.8915 ± 0.0389** | **0.9361** | **0.0769** | 0.8931 | 0.9378 | **0.0679** | **0.833** | 19 | 64 |
| Random Forest | Bagging | 0.8887 ± 0.0402 | 0.9351 | 0.0801 | 0.8912 | 0.9375 | 0.0702 | 0.817 | 311 | 44 |
| Logistic Regression | Linear | 0.8848 ± 0.0383 | 0.9285 | 0.1004 | 0.8860 | 0.9292 | 0.0898 | 0.801 | 4.5 | 2 |
| KNN (k = 25, 200k sample) | Instance-based | 0.8817 ± 0.0395 | 0.9291 | 0.1313 | 0.8855 | 0.9321 | 0.1097 | 0.795 | 0.1 | 1,862 |
| Linear SVM (200k sample) | Kernel (linear) | 0.8845 ± 0.0378 | 0.9287 | 0.1010 | 0.8845 | 0.9283 | 0.0919 | 0.807 | 13 | 7 |
| MLP 128-64 | Neural network | 0.8860 ± 0.0417 | 0.9311 | 0.0817 | 0.8831 | 0.9318 | 0.0722 | 0.824 | 182 | 16 |
| Gaussian NB | Probabilistic | 0.8628 ± 0.0483 | 0.8652 | 0.5981 | 0.8576 | 0.8571 | 0.6232 | 0.767 | 0.5 | 3 |
| *Rule baseline (no learning)* | | *0.856 (train)* | | | *0.858* | | | | | |

## Observations
- **Every learned model except Naive Bayes beats the rule baseline** (85.8% test Top-1). Gains run from 2.5 to 3.6 points.
- **Fold-to-fold spread (std about 0.04) is ten times larger than the gaps between the top models.** Some trips are much harder (folds 2 and 3 score about 0.84–0.86, folds 0 and 1 about 0.90–0.93). Grouping folds by trip is what exposes this; a row-level split would hide it.
- **Tree ensembles lead.** They capture interactions the linear models can't, for example "trust heading only when the car is moving".
- **Naive Bayes:** its log loss (0.60) is about 9× worse than the others. It assumes the features are independent, but many are strongly correlated (`dist` vs `dist_minus_min`, the context features), so it double-counts evidence and is overconfident.
- **KNN** is fast to fit but very slow to predict (1.9 s per 1,000 points), too slow for the API. It was trained on a 200k-row sample.
- **Route continuity** of the per-point choices is only 0.77–0.83: about 1 road change in 5 jumps between unconnected roads. This is the motivation for the HMM in Phase 6.

## Overfitting / underfitting (train-fold vs validation, Top-1)
| Model | Train Top-1 | CV Top-1 | Gap |
|---|---|---|---|
| Random Forest | 0.956 | 0.889 | 0.067: highest variance; deep trees memorise training trips |
| XGBoost | 0.951 | 0.891 | 0.060 |
| KNN | 0.921 | 0.882 | 0.039 |
| LightGBM | 0.929 | 0.892 | 0.038 |
| MLP | 0.896 | 0.886 | 0.010: early stopping at epoch 18 of 22 (`model_mlp_loss_curve.png`) |
| Logistic Regression | 0.883 | 0.885 | ≈ 0: underfits; too simple for feature interactions |
| Linear SVM | 0.883 | 0.885 | ≈ 0: same as logistic regression |
| Gaussian NB | 0.855 | 0.863 | ≈ 0: underfits; wrong independence assumption |

## Model selection
LightGBM and XGBoost are statistically tied on Top-1: 0.1 points on student_test, about 19 of 19,133 points, against a fold std of 3.9 points. The PLAN's criteria break the tie as follows:

| Criterion | LightGBM vs XGBoost |
|---|---|
| CV Top-1 (80 trips, more reliable than 10 test trips) | LightGBM higher in 3 of 5 folds; mean +0.06 points |
| Probability quality (log loss), which the HMM uses as emissions | **LightGBM better in all 5 folds** and on test |
| Generalisation gap | LightGBM 3.8 points vs XGBoost 6.0 |
| Route continuity (test) | LightGBM 0.833 vs 0.828 |
| Training time | LightGBM 19 s vs 34 s; matters for 50–100 Optuna trials |
| Inference time | XGBoost faster (14 vs 64 ms per 1k points); both are far below the API's needs |

**Decision (team, 2026-11-18): LightGBM** is the selected model. It goes forward to Bayesian optimisation (Phase 5) and the HMM (Phase 6). XGBoost is the documented runner-up.
