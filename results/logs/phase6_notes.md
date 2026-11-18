# Phase 6 notes: HMM + Viterbi route reconstruction

Code: `src/pathsnap/hmm.py`, `scripts/eval_hmm.py`, `tests/test_hmm.py` (3 tests).
Results: `hmm_grid_<model>.csv`, `hmm_comparison_<model>.csv`, `hmm_per_trip_<model>.csv`, where `<model>` is `lightgbm` (untuned, run by a teammate on a second machine) or `lightgbm_tuned`.
Partition: student_test (10 trips, 19,133 labelled points). The evaluator partition was not used.

## Answers to the report's questions (Deliverables §5)
1. **Hidden states:** for each GPS point, its candidate arcs (up to 12 within 50 m). Each state also stores the point's projected position t along that arc, so a state is a specific place on the network.
2. **Emission:** α · log P(arc | point). P is the LightGBM probability, normalised over the point's candidates. α weighs the classifier against the network.
3. **Transition:** log P(b at i | a at i−1) = −|d_net − d_gps| / β.
   - d_gps is the straight-line distance between the two GPS points.
   - d_net is the distance driven on the *directed* network from (a, t_a) to (b, t_b). On the same arc it's |t_b − t_a| · len. Otherwise it's the rest of arc a, plus the shortest path from a's head to b's tail (Dijkstra, cached per node and cut off at 3·d_gps + 200 m), plus the part of arc b.
   - Real driving gives d_net ≈ d_gps. Jumping to a parallel or unconnected road needs a long detour, so it scores badly, or it's unreachable.
4. **Missing candidates and gaps:**
   - Points with no candidates are skipped, and the transition is taken from the last point that had candidates.
   - A sampling gap over 60 s cuts the chain, and the next segment is decoded independently.
5. **Disconnected segments:** if no state at point i can be reached from any state at i−1, the chain is cut and a new segment starts at i with emissions only. The decoder never forces an impossible connection. On student_test this gave 14 segments over 10 trips.
6. **Did the HMM help?** Yes, clearly. See below.

## Classifier-only vs classifier + HMM (student_test)
| | Top-1 | Route continuity | Route precision | Route recall | Route F1 |
|---|---|---|---|---|---|
| Untuned LightGBM, classifier only | 0.8931 | 0.833 | 0.906 | 0.896 | 0.901 |
| Untuned LightGBM + HMM (β = 5, α = 0.5) | 0.9404 | 0.914 | 0.964 | 0.906 | 0.934 |
| Tuned LightGBM, classifier only | 0.8937 | 0.834 | 0.905 | 0.896 | 0.901 |
| **Tuned LightGBM + HMM (β = 5, α = 0.5)** | **0.9498** | **0.913** | **0.962** | **0.905** | **0.933** |

- **Top-1 rises by 5.6 points**, so about half of the classifier's mistakes are removed. Continuity rises from 83% to 91%. Route precision rises the most (0.905 → 0.962), because the HMM stops the route wandering onto roads it never used.
- **The HMM improves every one of the 10 trips; none gets worse.**
  - The smallest gain is trip 00000000, +1.3 points (95.2% → 96.5%).
  - The largest is the hardest trip, 00000077, +13.9 points (74.1% → 88.1%).
  - Trip 00000007 reaches 100%.
- **Where the HMM did not help enough:**
  - Trip 77 is still the weakest at 88.1%.
  - Continuity tops out at about 91%, because 14 chain cuts and some arcs between 1 Hz samples remain.
  - Route recall barely changes (0.896 → 0.905). Short arcs that no GPS point lands on can't be recovered by per-point decoding. Filling them with shortest paths is a possible extension.
  - Look at trip 77 on a Folium map in Phase 7.

## Decoder settings
- **Only the product α·β matters.** Viterbi takes the argmax of a sum, and multiplying every score by β turns α·log P − |Δ|/β into αβ·log P − |Δ| without changing the argmax. In the grid, settings with equal α·β give identical results, so the decoder really has **one** setting.
- **Results by α·β** (tuned probabilities):

  | α·β | Top-1 |
  |---|---|
  | 1 | 0.9438 |
  | 2 | 0.9473 |
  | **2.5** | **0.9498** |
  | 4 | 0.9455 |
  | 5 | 0.9421 |
  | 10 | 0.9376 |
  | 20 | 0.9280 |
  | 50 | 0.9201 |
  | 100 | 0.9121 |

  It's best at α·β ≈ 2.5, and the curve is flat between 1 and 4 (94.4–95.0%).
- **Chosen: β = 5 m, α = 0.5.** It was picked on student_test, which the guidelines allow for decoder parameters. Because the optimum is broad, the reported 95.0% is only mildly optimistic.
- **Cost:** 2–6 s for all 10 student_test trips on a laptop.
