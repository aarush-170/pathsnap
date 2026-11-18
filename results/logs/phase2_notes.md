# Phase 2 notes: per-point labels and candidate generation

Reproduce with `python -m scripts.build_dataset --sweep` and then `python -m scripts.build_dataset`.
Numbers come from `candidate_sweep.csv`, `candidate_summary.csv` and `label_stats.csv`.

## Why labels must be built
`.route` is an ordered list of arcs, about 5 GPS points per arc. Nothing says which GPS point belongs to which arc. We need a per-point label to train a classifier and to compute observation-level Top-1 accuracy.

## Method: order-preserving alignment (`src/pathsnap/labels.py`)
- **States.** For each GPS point, the states are the route positions j (index into `.route`) whose arc lies within 50 m.
- **Constraint.** j can never decrease from one point to the next, because the vehicle doesn't go backwards along its own route.
- **Cost.**
  - the distance from the point to the arc, plus
  - λ = 0.1 × |progress along the route − straight-line GPS displacement| between consecutive labelled points.
- **Search.** Viterbi-style DP over these states, then backtracking.
- **Fallbacks:**
  - If a point has no allowed predecessor, a penalised restart (200 m-equivalent) is allowed.
  - Short runs of points (< 30) cut off by a large backward jump (> 10 route arcs) are unlabelled. This affects only trip 61, where 11 points at the start lie only near the *end* of a round-trip route.
- Points more than 50 m from every route arc stay unlabelled (−1).

**Why not snap each point to its nearest route arc?**
- Two-way streets are stored as two arcs with identical geometry, so nearest-arc snapping picks a direction arbitrarily.
- It also fails where the route revisits a street.

Across all 90 trips the aligned label differs from nearest-arc snapping for 5.9% of points on average. More than half of that (3.2%) is the reverse-direction twin. In trip 95, a drive into and back out of dead ends, it's 44.6% vs 44.4%.

The design went through two iterations:
1. A first version used a linear progress penalty, λ·(progress). It telescopes over a path, so it couldn't separate the two directions at a U-turn: a unit test on a synthetic out-and-back street failed.
2. Replacing it with the |progress − displacement| term fixed this (`tests/test_phase2.py`).

## Label coverage
| | train | student_test |
|---|---|---|
| GPS points | 208,417 | 19,160 |
| Labelled | 89.2% | 99.9% |
| Route arcs with ≥ 1 point (mean per trip) | about 91% (short arcs can fall between 1 Hz samples) | |

Most unlabelled train points come from trips 54, 33 and 02 (see `eda_notes.md`). Six trips keep small backward steps of 1–9 arcs, which is jitter around turns; trip 8 has one step of 30 arcs.

## Candidate generation (`src/pathsnap/spatial.py`, `candidates.py`)
- Arcs are densified every 10 m into a KD-tree. A ball query with radius R + 5 m is complete (it can't miss an arc within R), and exact point-to-segment distances are then computed for the hits.
- Candidates are ranked by distance. Ties are rounded to 1 cm and use the 'min' rank, so both directions of a two-way street are always kept or dropped together.
- Sweep over R ∈ {10…100 m} × K ∈ {1…16, all}:

| R = 50 m | K = 8 | **K = 12** | K = 16 |
|---|---|---|---|
| Train miss rate | 0.88% | **0.59%** | 0.48% |
| Student_test miss rate | 0.24% | **0.07%** | 0.03% |
| Mean candidates per labelled point (train / student_test) | 5.9 / 6.1 | **7.2 / 7.8** | 8.1 / 8.9 |

- **Choice: R = 50 m, K = 12.** Going to K = 16 removes only 0.1% more misses but adds about 1 candidate per point, which grows the HMM transition matrix quadratically.
- **Caveat:** with no K limit, the miss rate at R ≥ 50 m is 0 *by construction*, since labels only exist within 50 m. The meaningful numbers are the curves below 50 m and the K cutoffs.
- **Baseline:** K = 1 (keep only the nearest arc and its twin) misses the true arc for 11.7% of train and 10.5% of student_test points. This is an upper bound for a "nearest road" heuristic, and the classifier has to beat it.

## Output tables (`data/processed/`, rebuilt and not committed)
- `labels_{split}.parquet`: trip, point_idx, t, label_arc, route_pos, label_dist, nearest_route_arc, n_candidates, true_in_candidates
- `candidates_{split}.parquet`: trip, point_idx, arc, dist, t (fraction along the arc), rank, is_true. Duplicate arcs, i.e. the same (from, to) pair, count as the same road when setting `is_true`.
