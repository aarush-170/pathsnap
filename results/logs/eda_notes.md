# EDA notes: data-quality findings and decisions

Generated from `python -m scripts.eda` (train = 80 trips, student_test = 10 trips). The numbers come from
`eda_trip_summary.csv`, `eda_anomalies.csv` and the figures in `results/figures/eda_*.png`.
The evaluator partition was not opened.

## Inventory
- All 90 trips have all 4 files (`.track .nodes .arcs .route`). Train is about 90 MB, mostly road-network files.
- GPS points per trip: median 1,529 (train) / 1,420 (student_test). Max is 19,313 (train/00000054).
- Road network per trip: median 9.6k nodes / 19.5k arcs (train). The largest is 155k nodes / 307k arcs (train/00000013).
- Each trip has its own road network, and arc and node IDs are local to the trip.

## Missing / invalid values
- 0 NaNs, 0 out-of-range coordinates, 0 duplicate or non-monotonic timestamps, 0 invalid route arc IDs and 0 isolated nodes.
- 106 zero-length arcs (two nodes at the same position).
- Duplicate arcs, i.e. the same (from, to) pair appearing more than once: median 2.5 per trip, max 322. Since arcs are straight lines between nodes, duplicates can't be told apart geometrically, so they must be treated as equivalent when scoring.

## Sampling and gaps
- About 99% of time steps are exactly 1 s (train median per trip 0.991; student_test 0.999).
- 23 train trips and 1 student_test trip have gaps over 60 s. The extremes are train/00000090 (153,064 s) and train/00000078 (52,810 s).
  **Decision:** split a trajectory into separate segments wherever a gap exceeds a threshold, and decode each segment on its own.
- 921 steps imply speeds over 200 km/h. They are concentrated in train/00000032, 00000001 and 00000013. Some are plausible (German Autobahn); others are GPS jumps.
  **Decision:** keep these points, but use speed and jump size as features. Don't delete points.

## GPS noise vs reference route
- Median distance from a GPS point to the nearest reference-route arc is 4.5 m. The 95th percentile per trip is about 14–16 m.
- Points within 30 m: student_test about 99%, train about 88%. In train, 10.4% of points are more than 60 m away, almost all from a few trips.
- **Decision:** candidate search radius of about 30–50 m. Choose the final value from the candidate-miss-rate curve in Phase 2.

## Anomalous trips (route does not explain the GPS)
| Trip | Far points (> 50 m) | Distance from far points to *any* network arc | Interpretation |
|---|---|---|---|
| train/00000054 | 87% | median 86 m | Road missing from the network. The GPS drives a ring road about 100 m from any arc. |
| train/00000033 | 59% | median 7 m | The GPS is on other network roads, so the reference route skips that part of the drive. |
| train/00000002 | 12% (mostly the last 552 points) | median 56 m | Road missing, mostly at the end of the trip. |
| train/00000043 | 7% | median 27 m | Mixed |
| train/00000096 | 4% (last 145 points) | median 76 m | Road missing at the end of the trip. |

**Decision:** keep these trips, but give no label to points that the order-preserving alignment places more than 50 m from the route. Exclude unlabelled points from training and from point-level metrics, and report how many were excluded. The reference route is treated as a reference, not perfect truth (see the Dataset document, §6).

## Road network
- 11 train trips and 1 student_test trip have a reference route with breaks, 14 breaks in total, where consecutive route arcs don't share a node.
- Median 19 weakly connected components per trip, and the largest covers about 98% of nodes. About 95% of arcs have a reverse twin, i.e. two-way streets stored as two directed arcs.
- **Implication for the HMM:** transitions use the directed graph. Only about 96% of nodes fall in the largest strongly connected component, so there must be a fallback for unreachable candidate pairs.
