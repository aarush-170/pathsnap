# Phase 5 notes: Bayesian hyperparameter optimisation (LightGBM)

Reproduce with `python -m scripts.tune_optuna --trials 20 --timeout 2700`. Numbers come from `optuna_trials.csv`, `optuna_param_importance.csv`, `tuning_comparison.csv` and `configs/lightgbm_tuned.json`. Figures: `optuna_history.png`, `optuna_param_importance.png`, `optuna_slices.png`, `optuna_tuned_vs_untuned.png`.

## Answers to the report's questions (Deliverables §4)
1. **Hyperparameters searched (8):** learning_rate, n_estimators, num_leaves, min_child_samples, subsample, colsample_bytree, reg_lambda, reg_alpha. Fixed: subsample_freq = 1.
2. **Ranges:**

   | Hyperparameter | Range | Scale |
   |---|---|---|
   | learning_rate | 0.01–0.2 | log |
   | n_estimators | 100–800 | step 50 |
   | num_leaves | 15–255 | log |
   | min_child_samples | 10–500 | log |
   | subsample | 0.5–1.0 | linear |
   | colsample_bytree | 0.4–1.0 | linear |
   | reg_lambda | 0.001–10 | log |
   | reg_alpha | 0.001–10 | log |

   Log scales are used where the effect is multiplicative.
3. **Objective:** the mean validation **log loss** over the same 5 trip-grouped folds as Phase 4, minimised.
   - Log loss judges probability quality, and the HMM uses those probabilities as emissions.
   - It's also smoother than Top-1 accuracy, so the optimiser chases less noise.
   - Top-1 and Macro F1 were logged for every trial.
4. **Trials:** 14 within a 45-minute budget: 11 completed and 3 pruned by the MedianPruner. The sampler was TPE (seed 42) with 8 random start-up trials. **Trial 0 was the untuned Phase 4 configuration.**
5. **Best (trial 8):**

   | Hyperparameter | Value |
   |---|---|
   | learning_rate | 0.0127 |
   | n_estimators | 650 |
   | num_leaves | 228 |
   | min_child_samples | 188 |
   | subsample | 0.823 |
   | colsample_bytree | 0.560 |
   | reg_lambda | 0.130 |
   | reg_alpha | 0.0024 |

   CV log loss: 0.07617 (untuned 0.07683).

   In short: learn slowly with more trees and more complex trees, but force bigger leaves and feature subsampling to keep variance under control.
6. **Did tuning help?** Yes on every metric, but only slightly for the classifier alone. Same folds and the same test trips throughout:

| | Untuned (Phase 4) | Tuned (Optuna) |
|---|---|---|
| CV Macro F1 | 0.9361 | **0.9368** |
| CV Top-1 | 0.8915 ± 0.0389 | **0.8917** ± 0.0391 |
| CV log loss | 0.0769 | **0.0762** (better in all 5 folds) |
| Student-test Top-1 | 0.8931 | **0.8937** |
| Student-test Macro F1 | 0.9378 | **0.9392** |
| Student-test log loss | 0.0679 | **0.0671** |
| Training time (full train) | **19 s** | 61 s |
| Inference time (ms per 1,000 points) | **64** | 247 |
| **Student-test Top-1 after HMM** (best β, α) | 0.9404 | **0.9498** |

## Interpretation
- **Learning rate, min_child_samples and num_leaves explain about 90% of the variation in score** (fANOVA importance). The sampling and regularisation settings hardly mattered.
- **The classifier's own accuracy hardly moved (+0.06 points).** The untuned configuration was already near the best these 26 features can reach.
- **The gain shows up after the HMM: +0.94 points (94.0% → 95.0%).** Better-calibrated probabilities give the decoder better emissions, which confirms that choosing log loss as the objective was the right call. Most of the gain is on the hardest trip (trip 77: 83.3% → 88.1%).
- **The cost:** about 4× slower inference. That's 247 ms per 1,000 points, or about 1.7 s for the largest evaluator trip, which is fine for the API.
- **Decision:** the tuned LightGBM (`configs/lightgbm_tuned.json`) is the final classifier.
