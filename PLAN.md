# PathSnap — Project Plan

The work is in 10 phases. Each phase has its tasks, its outputs, and the report section it feeds. The rules from the guidelines apply to every phase:

- **No leakage.** Fit models on the 80 train trips. Compare models on the 10 student_test trips. Never touch evaluator trips except through the API at demo time.
- **Use only real numbers.** Every number, table and plot comes from a script in this repo and is saved to `results/`.
- **Seed everything.** Log configs and timings.

---

## 0. What the data actually looks like (first sanity check)

Results from a quick check of all 90 trips:

| | train (80) | student_test (10) |
|---|---|---|
| GPS points per trip | median 1,529 (361 – 19,313) | median 1,420 (262 – 3,958) |
| Road network per trip | median 9.6k nodes / 19.5k arcs (max 155k / 307k) | median 6.6k / 13.1k |
| Route length (arcs) | median 280 (59 – 2,266) | median 250 |
| NaN values / non-increasing timestamps | 0 / 0 | 0 / 0 |
| Route continuity (to_node[i] == from_node[i+1]) | ≈ 99.9 % | ≈ 99.97 % |
| Distance GPS point → nearest route arc | median ≈ 4.5 m, p95 ≈ 14 m | median ≈ 4.6 m, p95 ≈ 16 m |
| Typical arc length | median ≈ 37 m | median ≈ 29 m |
| Max time gap | up to 153,064 s (!) | up to 210 s |

**What this means for the design:**
- The **road network is different for each trip** and arc IDs are local to that trip. Every component works per trip.
- **There is no label for each GPS point.** `.route` is an ordered list of arcs, and one arc spans several points. We need to build those labels ourselves (Phase 2).
- Arcs are **directed**. The `.route` follows from→to, so the direction of travel is a useful feature.
- GPS noise is small, about 5 m on average. A **candidate radius of roughly 30–50 m** should cover almost every true arc. We'll confirm this with a candidate-miss-rate curve.
- **Anomalous trips need special handling:**
  - `00000002`, `00000054` and `00000033` have points 100–570 m from the route. The route may cover only part of the trip, or there may be an offset.
  - Some trips have huge time gaps, so the GPS has to be split into segments.
  - Route continuity is slightly below 100 % on a few trips.

  Document all of these in the EDA, and decide whether to drop the points or the trips.
- Trips are spread worldwide: train longitudes run from −100° to +137° and latitudes from −35° to +61°, across the Americas, Europe, Africa, Asia and Japan. The evaluator trips are just as spread out. So we use a **per-trip** metre projection (local equirectangular centred on each trip), not one fixed UTM zone. Never use Euclidean distance on degrees.

---

## Phase 1: EDA (feeds report §2). ✅ Done
`scripts/eda.py`, plus `src/pathsnap/io.py`, `geo.py`, `network.py` and `viz.py`. Findings and decisions are in `results/logs/eda_notes.md`.

- [x] Loaders for `.track/.nodes/.arcs/.route`, and a `Trip` object
- [x] Inventory: trips per partition, file sizes, row counts (`eda_file_inventory.csv`)
- [x] Missing or invalid values, checks that coordinates are plausible, timestamp ordering, histogram of time gaps
- [x] Trajectory stats: duration, length, speed distribution, geographic extent
- [x] Road network: nodes and arcs, connected components (scipy csgraph), isolated nodes, two-way arcs, duplicate arcs
- [x] Route: continuity, repeated arcs, GPS → route distance distribution
- [x] Anomaly table (`eda_anomalies.csv`) with a decision for each type
- [x] **Folium maps**: `results/maps/eda_train_00000095.html` and `eda_anomaly_train_00000054.html`
- [x] Figures saved to `results/figures/eda_*.png`

## Phase 2: Labels and candidate generation. ✅ Done
`labels.py`, `spatial.py`, `candidates.py`, `scripts/build_dataset.py`, `tests/test_phase2.py`. Notes for the report are in `results/logs/phase2_notes.md`.

- [x] **Per-point labels from the route.** Order-preserving DP alignment. The cost is distance to the arc plus λ·|route progress − GPS displacement|. Points more than 50 m from every route arc are left unlabelled, and short runs (< 30 points) cut off by a backward jump of more than 10 arcs are unlabelled too.
- [x] **Candidates.** Arcs are densified every 10 m into a KD-tree, then exact point-to-segment distances are computed. Keep arcs within R, cut off at distance rank K (the two directions of a street always share a rank).
- [x] **Choose R and K** from the sweep (`candidate_sweep.csv`, `candidates_miss_rate.png`): **R = 50 m, K = 12**, giving a miss rate of 0.59% on train and 0.07% on student_test, with about 7–8 candidates per point.
- [x] Output: `data/processed/{labels,candidates}_{split}.parquet`. Not committed; rebuild with `python -m scripts.build_dataset` (about 45 s).

## Phase 3: Feature engineering. ✅ Done
`src/pathsnap/features.py`, `src/pathsnap/metrics.py`, `scripts/build_features.py`, `tests/test_phase3.py`. Notes for the report are in `results/logs/phase3_notes.md`.

- [x] 26 features in 6 groups: distance, projection, heading, context (points i±2), motion, network. None of them read the route.
- [x] Feature tables in `data/processed/features_{split}.parquet` (1.36M train rows). Not committed; rebuild in about 20 s.
- [x] Univariate AUC, feature distributions, correlation heatmap
- [x] Rule baselines (no learning): nearest + heading gives **Top-1 85.6% on train and 85.8% on student_test**. This is the bar for Phase 4.

**Target:** binary, "is this candidate the true arc?". Trained only on rows from labelled points. The classifier's probabilities are normalised per point, both for Top-1 accuracy and as the HMM emission.

## Phase 4: Train and compare at least 5 models (feeds report §3). ✅ Done
**Result** (`results/logs/phase4_notes.md`): all 8 models trained.
- Top 3: LightGBM 0.8915 CV Top-1 / 0.8931 test, XGBoost 0.8909 / 0.8941, Random Forest 0.8887 / 0.8912.
- Every model except Naive Bayes beats the 0.858 rule baseline.
- **Selected: LightGBM** (team decision). It's tied with XGBoost on Top-1, but has better log loss in all 5 folds, a smaller overfitting gap and faster training.

`src/pathsnap/models.py` (model registry), `scripts/train_models.py` (one protocol for all models). Each model writes `results/models/<model>/` (metrics.json, cv_folds.csv, confusion_test.csv, MLP history.csv), and `--report` merges them into `results/logs/model_comparison.csv` and `results/figures/model_*.png`.

- 5-fold **GroupKFold by trip** on train (trip-to-fold map saved in `results/logs/cv_folds.csv`). Refit on all train trips, then evaluate once on student_test.
- Models (8 implemented; all untuned starting configs, recorded in metrics.json):
  1. Logistic Regression (linear)
  2. Gaussian Naive Bayes (probabilistic)
  3. KNN, distance-weighted, k = 25 (instance-based; trained on a 200k-row sample)
  4. Random Forest, 200 trees (bagging)
  5. LightGBM (boosting)
  6. XGBoost (boosting)
  7. MLP in PyTorch, 128-64 hidden, early stopping on held-out trips (neural; real epoch loss curves)
  8. Linear SVM with Platt calibration (kernel-based; trained on a 200k-row sample)
- Overfitting check: every fold also records train-fold Top-1, Macro F1 and log loss on a fixed sample of about 200k rows.
- Metrics:
  - CV and test Macro F1, Precision, Recall, Log Loss
  - **Observation-level Top-1 accuracy**
  - Candidate miss rate
  - Route continuity of the argmax route
  - Training time and inference time
  - Confusion matrices
- Plots:
  - CV Macro F1 bar chart with fold error bars
  - Test Top-1 and F1 chart
  - Log Loss chart
  - MLP train/validation loss curves only
- **Selection criteria:** test Top-1, then Macro F1, then probability quality (log loss), then speed and stability across folds.

## Phase 5: Bayesian hyperparameter optimisation (feeds report §4). ✅ Done
`scripts/tune_optuna.py`. Notes are in `results/logs/phase5_notes.md`.

- [x] Optuna TPE on LightGBM over 8 hyperparameters, with the same trip-grouped 5 folds. The objective is CV log loss. Ran 14 trials in a 45-minute budget (11 completed, 3 pruned); trial 0 was the untuned config.
- [x] Saved the trials CSV, history, parameter-importance and slice plots, and `configs/lightgbm_tuned.json`.
- [x] Tuned vs untuned: CV log loss 0.0769 → 0.0762 (better in all 5 folds), test Top-1 0.8931 → 0.8937, and **after the HMM 0.9404 → 0.9498**.

## Phase 6: HMM and Viterbi (feeds report §5). ✅ Done
`src/pathsnap/hmm.py`, `scripts/eval_hmm.py`, `tests/test_hmm.py`. Notes are in `results/logs/phase6_notes.md`.

**Result on student_test (tuned LightGBM):** Top-1 **0.8937 → 0.9498**, continuity 0.834 → 0.913. All 10 trips improved. Best β = 5, α = 0.5; only α·β matters.

Original design:

- **Hidden states:** the candidate arcs for each GPS point.
- **Emission:** log of the classifier probability, normalised per point.
- **Transition:** from candidate a at time t to b at t+1:
  - same arc → high probability;
  - otherwise use network distance d_net from a's position along a to b's position along b, found with Dijkstra on the directed arc graph and cut off at a distance limit;
  - p ∝ exp(−|d_net − d_gc| / β), where d_gc is the straight-line distance between the two GPS points;
  - unreachable within the limit → probability ≈ 0, with a small fallback.

  Cache Dijkstra results for each source arc.
- **Special cases:**
  - *Points with no candidates:* skip them in decoding, and fill them in from their neighbours afterwards.
  - *Large time gaps (> gap_break_seconds):* split the trip into segments and decode each one separately.
  - *All transitions impossible:* restart the chain at that point. Viterbi would otherwise return nothing for the rest of the trip.
- Tune β and the gap threshold on train and student_test only.
- Compare **classifier-only argmax** with **classifier + HMM** on Top-1 accuracy, continuity, and sequence-level metrics (precision/recall of route arcs, edit distance). Include worked examples where the HMM helped and where it made things worse.

## Phase 7: Folium visualisation (feeds report §6)
`scripts/make_maps.py`, output in `results/maps/*.html`

- Layers: raw GPS (blue dots), nearby road network (grey), reference route (green), predicted route (red), and points the model got wrong (orange). Add a legend, a LayerControl, and fit_bounds to the trip.

## Phase 8: Export artifacts
`scripts/export_artifacts.py`

- `models/classifier.joblib` (or `.pt`), `models/feature_config.json`, `models/hmm_config.json`, and any scaler. Keep them small enough for GitHub.

## Phase 9: API (Laptop 1) and UI (Laptop 2) (feeds report §7)
`api/main.py`, `ui/app.py`

**API (FastAPI), endpoints:**
- `GET /health`
- `GET /trips` lists the 10 evaluator trip IDs found in `data/eval/`
- `POST /predict {trip_id}`:
  1. load the trip's track, nodes and arcs
  2. generate candidates
  3. compute features
  4. run the classifier
  5. run Viterbi
  6. return the predicted arc sequence, its coordinates, the GPS points and timings
  7. **on the server only**, load `.route`, build labels, and return the accuracy, continuity and arc precision/recall

  The ground truth is never passed into the prediction pipeline. Every request is logged to the console so it can be shown to the evaluator.

**UI (Streamlit + streamlit-folium):**
- The API URL is configurable (Laptop 1's LAN IP).
- A dropdown of the 10 trips and a Run button.
- A map of GPS points vs predicted route, plus the metrics and timings.
- The UI never sees ground-truth arcs, only the metrics.

**Networking:**
- Run the API with `--host 0.0.0.0`.
- Allow port 8000 through the Windows firewall.
- Test across two machines on the same Wi-Fi beforehand, and try all 10 trip IDs.

## Phase 10: Report and submission
- [ ] PDF following Deliverables §2–7. Every table and figure comes from `results/`.
- [ ] Make the repo public, with a README containing exact run commands.
- [ ] AI-use disclosure, as required by the event policy.
- [ ] Every team member rehearses viva answers for each phase.

---

## Suggested order and checkpoints

| Checkpoint | Done when |
|---|---|
| **Phase-1 evaluation (before lunch)** | EDA, labels, candidates and features done; at least a baseline model trained; a Folium map exists |
| Mid | 5+ models compared; Optuna run complete; HMM working on student_test |
| Final | Artifacts exported; API and UI tested across two laptops on all trips; report PDF and public repo |

## Code layout

```
src/pathsnap/io.py          read trip files -> Trip(track, nodes, arcs, route)
src/pathsnap/geo.py         projection, haversine, point-segment distance, bearings
src/pathsnap/network.py     arc graph, adjacency, Dijkstra with cutoff
src/pathsnap/labels.py      monotonic DP alignment -> per-point true arc
src/pathsnap/candidates.py  spatial index + radius / top-K candidates
src/pathsnap/features.py    (point, candidate) feature table
src/pathsnap/hmm.py         emission / transition / Viterbi with gap handling
src/pathsnap/metrics.py     top-1, macro F1, log loss, miss rate, continuity, sequence metrics
src/pathsnap/pipeline.py    predict_trip(trip, artifacts) used by both scripts and API
```

`pipeline.predict_trip` is shared by the experiments and the API. This keeps the demo identical to what the report describes.
