# Phase 3 notes: feature engineering

Reproduce with `python -m scripts.build_features`. Numbers come from `feature_univariate.csv` and `rule_baselines.csv`. Figures: `feature_univariate_auc.png`, `feature_distributions.png`, `feature_correlation.png`.

## Problem framing
- Each row is a (GPS point, candidate arc) pair.
- The target is `is_true`: is this candidate the point's labelled arc? Duplicate arcs count as the same road.
- A trained model scores every candidate, and the highest score at each point is the prediction. That gives observation-level Top-1 accuracy, and the per-point scores become the HMM emission probabilities in Phase 6.
- Only rows from labelled points are used to train and evaluate (98.2% of train rows, 100% of student_test rows).
- Class balance: about 1 positive per 7–8 candidates.

## Features (`src/pathsnap/features.py`, 26 in total)
None of them use the reference route. The test `test_features_do_not_need_route` checks this, so the same code can run on evaluator trips in the API.

| Group | Features | Why (link to EDA / Phase 2) |
|---|---|---|
| Distance | `dist`, `rank`, `dist_minus_min`, `dist_ratio_min`, `dist_z`, `n_candidates`, `gap_1st_2nd` | GPS noise has a median of about 4.5 m, so the true arc is usually close. Distance *relative to the other candidates* matters more than absolute distance, because noise varies from trip to trip. `gap_1st_2nd` measures ambiguity. |
| Projection | `t`, `t_at_end`, `arc_len` | A point whose projection is clipped to an arc's end is past that arc. Arcs have a median length of about 37 m, so this happens often near intersections. |
| Heading | `heading_cos`, `heading_absdiff`, `heading_cos_prev`, `heading_cos_next`, `heading_rank`, `heading_valid` | About 95% of arcs have a reverse twin with identical geometry, so **only heading can tell the two directions of a street apart**. GPS heading is measured over points i−2…i+2, and over the single previous and next steps. Below 3 m of displacement it's treated as unknown (`heading_valid` = 0). |
| Context | `ctx_dist_mean`, `ctx_dist_max`, `ctx_minus_min`, `ctx_rank` | Distance from points i−2…i+2 to the arc, extended 30 m at each end. A road that fits the whole local trace is more likely than one that fits a single noisy point. |
| Motion | `speed`, `dt_prev`, `dt_next` | 24 trips have gaps over 60 s. Heading and context are less reliable across gaps and at low speed. |
| Network | `from_in_degree`, `to_out_degree`, `has_reverse_twin` | Intersections, where several roads meet, are where mistakes concentrate. |

## Univariate analysis (train, labelled points)
- **Strongest single features (AUC):** `dist_minus_min` 0.91, `rank` 0.90, `ctx_minus_min` 0.90, `dist_ratio_min` 0.90, `ctx_rank` 0.90, `dist` 0.86, heading features 0.83, `t_at_end` 0.83.
- **Weak on their own** (AUC about 0.50–0.54): `t`, `dt_prev`, `dt_next`, `heading_valid`, `speed`, the degree features. They are kept because they *modulate* how much to trust the others: heading at low speed, context across a gap. A tree model can use those interactions and the AUC can't show them.
- **Redundancy** (|Spearman| > 0.9):
  - `heading_cos` vs `heading_absdiff` (−1.0, a monotone transform)
  - `dist` vs `dist_minus_min` (0.92)
  - `ctx_dist_mean` vs `ctx_dist_max` and `ctx_minus_min` (0.94 / 0.91)

  They are kept for now because linear and tree models use them differently. The model-comparison phase can test dropping them.

## Rule baselines (no learning, observation-level Top-1)
| Rule | Train | Student_test |
|---|---|---|
| Nearest arc (`dist`) | 55.6% | 53.0% |
| Context only (`ctx_dist_mean`) | 54.7% | 52.5% |
| Context + heading | 81.7% | 82.4% |
| **Nearest + heading** (`−dist + 10·heading_cos`) | **85.6%** | **85.8%** |

- The nearest-arc rule scores only about 55% because the two directions of a street are always tied, so it picks a direction at random.
- Adding heading lifts it to about 86%. The trained classifiers in Phase 4 have to beat that.
- The upper bound is 1 − candidate miss rate: 99.4% on train, 99.9% on student_test.
