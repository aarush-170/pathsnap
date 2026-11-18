# Phase 8 notes: exported artifacts and the shared inference pipeline

Reproduce with `python -m scripts.export_artifacts` (about 2 minutes: trains, saves, verifies). Add `--check-eval` for the evaluator run-check.

## Artifacts (`models/`)
| File | What | Size |
|---|---|---|
| `lightgbm_tuned.txt` | LightGBM booster (Optuna trial 8 parameters) trained on all 80 train trips: 1,336,027 rows, 26 features. Text format, so it loads on any machine with LightGBM. | 16.4 MB |
| `pipeline_config.json` | Feature list (checked against the code at load time), candidate settings (50 m, K = 12), HMM settings (β = 5, α = 0.5, 60 s gap break, Dijkstra cut-off), label-alignment settings for server-side scoring, training summary and library versions | 2 KB |

## One prediction path (`src/pathsnap/pipeline.py`)
`predict_trip(trip, artifacts)` runs these steps on a trip loaded **without its route**:
1. Projection
2. Candidates
3. Features
4. LightGBM probabilities
5. HMM / Viterbi
6. Route reconstruction (shortest-path fill between predicted arcs)

It returns the per-point arcs, the classifier-only arcs, the drivable route and timings for each stage. **The API calls this exact function.**

## Server-side scoring (`src/pathsnap/evaluation.py`)
`score_prediction(trip_with_route, point_arc, label_cfg)` reads the reference route only **after** prediction. It rebuilds the per-point labels with the same order-preserving alignment as Phase 2, and returns Top-1, route continuity and route precision/recall/F1. The reference arcs themselves are never returned.

## Verification on student_test (`results/logs/export_check.csv`)
- **Classifier parity:** the exported model's probabilities are **identical** to the Phase 5 refit (max difference 0.0).
- **End-to-end from raw files:** classifier-only **0.8937** (matches Phase 5), classifier + HMM **0.9493**.
  - Phase 6 reported 0.9498. The 0.05-point difference comes from the real pipeline also decoding the 27 student_test points that have no label (mostly trip 00000056). Those points were absent from the Phase 6 evaluation table, and their presence shifts a few Viterbi decisions nearby.
  - **0.9493 is the number to quote for the deployed system**, because it is exactly what the API computes.
- **Time per trip:** 0.2–2.6 s on a laptop CPU.

## Run-check on the 10 evaluator trips (`results/logs/export_eval_runcheck.csv`)
The pipeline was run once on each evaluator trip to make sure the demo works for all ten. This was **prediction only: the evaluator routes were not loaded and nothing was scored**, so no evaluator information influenced any model, threshold or setting.
- All 10 trips ran without errors in 0.4–3.5 s. Each produced a continuous route with 0 gaps.
- Trip 00000073 is decoded in 4 segments because of its 220 s GPS gap. Trip 00000015 has 21 points with no road within 50 m; they get no prediction rather than a forced one.
