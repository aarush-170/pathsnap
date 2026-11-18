# 2. Dataset Understanding and Exploratory Data Analysis

This first-review section covers the dataset structure, the preprocessing decisions made from the available data, and the features planned for Phase 3. EDA uses only `train` (80 trips) and `student_test` (10 trips). The evaluator partition was not opened or used.

## 2.1 Dataset structure

The raw data is grouped by trip. Each trip directory contains four headerless, whitespace-separated files. The row index is meaningful for the network files: a node's zero-based row number is its node ID, and an arc's zero-based row number is its arc ID. IDs are local to each trip's network.

| File | Rows and meaning | Role in the pipeline |
|---|---|---|
| `.track` | Longitude, latitude, timestamp in seconds | Ordered GPS observations of the vehicle |
| `.nodes` | Longitude, latitude | Road-network node coordinates |
| `.arcs` | Integer `from_node_id`, `to_node_id` pairs | Directed road segments connecting rows in `.nodes` |
| `.route` | Ordered integer arc IDs | Supplied reference route: an ordered sequence of rows from `.arcs` |

*Table 1. Raw trip files, columns, and the identifiers used to relate them. Source: `src/pathsnap/io.py` and Dataset Exploration, “Raw File Types.”*

The relationship is GPS point → nearby candidate arc → ordered route. The route is not a per-point label file: it is an ordered list of arcs, and the trip-level median is about five GPS observations per route arc. We therefore will not pair row *i* in `.track` with row *i* in `.route`. The planned label preparation is an order-preserving alignment of GPS observations to the route arc sequence. The EDA decision is to leave a point unlabelled when the alignment places it more than 50 m from the reference route; unlabelled points will be omitted from classifier training and point-level metrics.

### Inventory

All expected files are present in both analyzed partitions. The table reports exact byte and row totals from `eda_file_inventory.csv`; rows are summed across the trips in that split.

| Split | Trips | Files present | File type | Total size (bytes) | Total rows |
|---|---:|---:|---|---:|---:|
| train | 80 | 80 | `.track` | 6,872,352 | 208,417 |
| train | 80 | 80 | `.nodes` | 38,431,647 | 1,933,485 |
| train | 80 | 80 | `.arcs` | 44,267,751 | 3,876,904 |
| train | 80 | 80 | `.route` | 165,702 | 31,336 |
| student_test | 10 | 10 | `.track` | 625,911 | 19,160 |
| student_test | 10 | 10 | `.nodes` | 1,822,179 | 90,689 |
| student_test | 10 | 10 | `.arcs` | 1,902,157 | 185,041 |
| student_test | 10 | 10 | `.route` | 18,344 | 3,477 |

*Table 2. File inventory by split and file type. There are 320 files in train and 40 in student_test, with one file of each type per trip. Source: `results/logs/eda_file_inventory.csv`; aggregates are reproducible with `results/verify_section2.py`.*

Each trip has its own network, so node and arc IDs must never be interpreted across trip directories. The route sequence refers to arc IDs in that trip's `.arcs` file. In the analyzed data the per-trip median ratio of GPS observations to route arcs is 5.289222 in train and 4.634907 in student_test; neither value implies a direct one-to-one label mapping.

## 2.2 Preprocessing and data-quality decisions

`scripts/eda.py` loads the headerless files with explicit columns and numeric types, then checks missing values, coordinate plausibility, timestamp order, network references, route IDs, geometry, and connectivity. Geometric calculations use a trip-centred local equirectangular projection in metres. We do not use Euclidean distances in raw longitude/latitude degrees: degrees are angular units, not consistent ground distances. The implementation describes the projection error as below 0.1% over a 20 km scale and uses a Haversine helper for cross-checks. At the scale of these trips this is small relative to the tens-of-metres candidate search; the same projection is used for GPS points, nodes, and arc distances.

| Check over train + student_test | Count | Preprocessing decision |
|---|---:|---|
| NaN values | 0 | No missing-value imputation was needed in the inspected fields |
| Invalid coordinates | 0 | No coordinate rows were removed |
| Duplicate timestamps | 0 | Keep timestamp order as supplied |
| Non-monotonic timestamps | 0 | No timestamp reordering was needed |
| Invalid route arc IDs | 0 | All supplied route IDs index an arc in that trip |
| Isolated nodes | 0 | No isolated nodes were found by the network check |
| Zero-length arcs | 106 | Retain, but treat as zero geometric length |
| Route breaks | 14 | Preserve the ordered route; do not assume each adjacent pair is connected |
| Steps implying speed above 200 km/h | 921 | Keep the observations; use speed and jump size as planned signals |

*Table 3. Missing-value and structural checks across both analyzed partitions. Source: `results/figures/eda_data_quality.png`, `results/logs/eda_trip_summary.csv`, and `results/logs/eda_notes.md`.*

![Data-quality check counts for train and student_test.](../results/figures/eda_data_quality.png)

*Figure 1. Counts of missing and flagged structural values across train and student_test. Zero-count bars are checks that passed; zero-length arcs and route breaks are retained as observed data conditions. Source: `results/figures/eda_data_quality.png`.*

Sampling is nominally one observation per second. The median, across trips, of the fraction of exact one-second intervals is 0.990979 in train and 0.999323 in student_test. Gaps longer than 60 s occur in 23 train trips and 1 student_test trip; the longest observed gap is 153,064 s in train trip `00000090`. We will split trajectories at large gaps and decode each segment independently, while retaining a time-gap feature. The interval and speed plots show why deleting all high-speed transitions would be unsafe: some are GPS jumps, but some may be plausible high-speed driving.

![Sampling intervals and long-gap distribution.](../results/figures/eda_sampling_intervals.png)

*Figure 2. Sampling-interval counts and the distribution of intervals longer than 5 s, pooled across train and student_test. Source: `results/figures/eda_sampling_intervals.png`.*

![Instantaneous speed distribution.](../results/figures/eda_speed_distribution.png)

*Figure 3. Speed between consecutive GPS observations, clipped at 200 km/h for display. The 921 above-threshold steps are retained rather than deleted. Source: `results/figures/eda_speed_distribution.png` and `results/logs/eda_trip_summary.csv`.*

GPS-to-route distance provides the initial candidate-search scale. Across trips, the median of per-trip median distances is 4.498225 m in train and 4.593247 m in student_test. The corresponding medians of per-trip 95th-percentile distances are 14.216592 m and 15.768227 m. Most ordinary observations are close to the reference route, but a few trips are not. We therefore plan to start candidate generation in the approximately 30–50 m range and choose the final radius using the Phase 2 candidate-miss-rate curve, rather than treating the reference route as complete everywhere.

![GPS distance to the nearest reference-route arc.](../results/figures/eda_gps_to_route_distance.png)

*Figure 4. Pooled GPS-to-reference-route distances for train and student_test; the vertical guides mark 30 m and 50 m. Source: `results/figures/eda_gps_to_route_distance.png`.*

![Typical GPS trajectories over the road network and reference route.](../results/figures/eda_trajectories_typical.png)

*Figure 5. Three typical train trips and three typical student_test trips, selected by closeness to each split's median GPS-point count. Blue marks GPS observations, green shows the reference route, and grey shows nearby road-network arcs. Source: `results/figures/eda_trajectories_typical.png`.*

![GPS observations and route arcs per trip.](../results/figures/eda_points_per_trip.png)

*Figure 6. GPS-point counts per trip and the relationship between GPS observations and reference-route arc counts in train and student_test. Source: `results/figures/eda_points_per_trip.png`.*

The network checks show a large, mostly connected directed graph with some structural exceptions. The train median is 9,608 nodes and 19,526.5 arcs per trip; the median weak-component count is 20 and the median largest strongly connected component covers 0.964191 of nodes. About 95% of arcs have a reverse twin. Duplicate directed arcs occur in the data (median 2.5 per trip, maximum 322); since they have the same endpoints, we will treat geometrically identical duplicates as equivalent when scoring a candidate. A reverse twin is not assumed for every road. Directed connectivity also means some candidate transitions can be unreachable; the planned route decoder needs a fallback for those pairs. The route list has 14 breaks in total, where consecutive reference arcs do not share a node.

![Road-network size and connectivity per trip.](../results/figures/eda_network_connectivity.png)

*Figure 7. Node and arc counts, weakly connected components, and largest strongly connected component fraction per trip in train and student_test. Source: `results/figures/eda_network_connectivity.png`.*

![Road-arc length distribution.](../results/figures/eda_arc_length.png)

*Figure 8. Arc lengths computed between endpoint nodes in projected metres; the horizontal scale is logarithmic. Source: `results/figures/eda_arc_length.png`.*

### Trips where the reference route does not explain the GPS

The anomaly checks distinguish a route that skips part of the drive from a road that is absent from the supplied network. For train trip `00000033`, many GPS points are far from the reference route but close to another network arc: this indicates that the route skips roads used by the drive. For `00000054`, those points are also far from every network arc, indicating missing network geometry. Trips `00000002` and `00000096` show a similar missing-road pattern near the end. We keep these trips, but follow the EDA decision to leave points more than 50 m from the reference route unlabelled after order-preserving alignment. This avoids forcing a route-derived label onto points the route does not explain.

| Train trip | Points more than 50 m from route | Median distance from those points to any network arc | EDA interpretation |
|---|---:|---:|---|
| `00000054` | 87% | 86 m | Road missing from the network; GPS follows a ring road about 100 m from any arc |
| `00000033` | 59% | 7 m | GPS follows other network roads; reference route skips part of the drive |
| `00000002` | 12%, mostly the last 552 points | 56 m | Road missing, mostly at the end |
| `00000043` | 7% | 27 m | Mixed case |
| `00000096` | 4%, last 145 points | 76 m | Road missing at the end |

*Table 4. The five largest route-distance anomalies identified in train. Percentages and interpretations are reported as in the EDA notes; distances are medians for far points. Source: `results/logs/eda_notes.md` and `results/logs/eda_anomalies.csv`.*

![Trips with the largest distance from GPS to the reference route.](../results/figures/eda_trajectories_anomalies.png)

*Figure 9. The three trips with the highest fraction of GPS points more than 50 m from their reference route. Source: `results/figures/eda_trajectories_anomalies.png`.*

![Anomaly distance from the reference route over time.](../results/figures/eda_anomaly_distance_over_time.png)

*Figure 10. GPS-to-reference-route distance over each selected anomalous trip; the dashed line marks the 50 m EDA threshold. Source: `results/figures/eda_anomaly_distance_over_time.png`.*

The interactive map for the typical train example is available at [`eda_train_00000095.html`](../results/maps/eda_train_00000095.html). It has separate layers for nearby road arcs, reference route, and GPS observations, plus a visible legend and layer control. The browser screenshot requested for the report has not been captured in this draft because the available Windows browser automation targeted an inconsistent window.

### Assumptions and limitations

The supplied route is a reference, not independently verified, error-free ground truth. The GPS can be noisy, the network can omit roads, and the reference route can skip roads that appear in the network. Our planned 50 m unlabelled-point rule and candidate radius are therefore practical EDA decisions, not claims that all remaining labels are certainly correct. The detailed feature extraction and classifier have not been implemented yet.

## 2.3 Planned feature-engineering decisions

Phase 3 is planned work, not a completed implementation. Features will be created for each GPS-point/candidate-arc pair. The EDA signals below motivate the planned groups; they do not establish model performance.

| Planned feature group | Planned values from `PLAN.md` | EDA reason |
|---|---|---|
| Candidate geometry | Perpendicular distance; rank among the point's candidates; distance above the minimum candidate distance; within-point distance z-score | Typical GPS-to-route distances are in the low tens of metres, while the anomaly trips show that distance alone cannot make every point labelable. |
| Direction | Signed and absolute difference between smoothed GPS heading and directed arc bearing; reverse arcs use the opposite heading | `.arcs` is directed, and about 95% of arcs have reverse twins; direction can distinguish the two travel directions on the same geometry. |
| Motion and position on arc | Speed, acceleration, projection fraction along the arc, and arc length | The sampling is close to 1 Hz but contains long gaps and 921 steps above 200 km/h; arc lengths vary across the network. We retain these observations and expose the motion signal to the model. |
| Local network context | Candidate count for the point; node degrees; reverse-twin indicator | Network size and connectivity vary by trip, some routes are broken, and reverse twins are common but not universal. |
| Temporal context | Whether the candidate is also near the previous and next points; distance to the previous point's nearest candidate; time gap to the previous point | Most observations are regularly sampled, but gaps longer than 60 s occur. Large gaps will split decoding segments; local continuity remains useful inside a segment. |

*Table 5. Planned Phase 3 candidate features and their connection to observed PathSnap data. Feature names and target definition come from `PLAN.md`, Phase 3; EDA rationales cite the files listed in the final number trail.*

After order-preserving route alignment, the planned candidate target is binary: whether a candidate is the aligned reference arc for that observation. Candidate scores will be normalized per GPS point for Top-1 selection and as emissions to the planned HMM. Points more than 50 m from the reference route remain unlabelled and are excluded from training and point-level metrics, with the excluded count reported when the alignment is implemented. The final search radius will be selected from the Phase 2 candidate-miss-rate curve.

## Number trail

The values below include the data counts and thresholds stated in the prose and tables. Table 2 contains the exact byte and row totals by split and file type. Figure axes and plotted thresholds are sourced by the captions under Figures 1–10.

| Values used | Source file(s) |
|---|---|
| 80 train trips; 10 student_test trips; 4 raw files per trip; 320 train files and 40 student_test files; every per-type byte and row total in Table 2 | `results/logs/eda_file_inventory.csv`; `results/section2_inventory_summary.csv` |
| 3 track columns; 2 node columns; 2 arc columns; 1 route column; zero-based row IDs; about five GPS observations per route arc; train ratio 5.289222 and student_test ratio 4.634907 | `src/pathsnap/io.py`; Dataset Exploration, “Raw File Types”; `results/logs/eda_trip_summary.csv`; `results/logs/eda_split_summary.csv` |
| 0 NaNs; 0 invalid coordinates; 0 duplicate timestamps; 0 non-monotonic timestamps; 0 invalid route IDs; 0 isolated nodes; 106 zero-length arcs; 14 route breaks; 921 steps above 200 km/h | `results/figures/eda_data_quality.png`; `results/logs/eda_trip_summary.csv`; `results/logs/eda_notes.md` |
| Projection error bound 0.1% over 20 km | `src/pathsnap/geo.py` |
| Per-trip median fractions of exact one-second intervals: 0.990979 (train), 0.999323 (student_test); gaps over 60 s in 23 train trips and 1 student_test trip; maximum gap 153,064 s in train/00000090 | `results/logs/eda_split_summary.csv`; `results/logs/eda_trip_summary.csv`; `results/logs/eda_notes.md` |
| Per-trip median GPS-to-route distance 4.498225 m (train), 4.593247 m (student_test); per-trip median 95th-percentile distance 14.216592 m (train), 15.768227 m (student_test); candidate range 30–50 m; alignment cutoff 50 m | `results/logs/eda_split_summary.csv`; `results/logs/eda_notes.md`; `scripts/eda.py` |
| Train median network size 9,608 nodes and 19,526.5 arcs per trip; 20 weak components per trip; largest SCC fraction 0.964191; approximately 95% reverse twins; duplicate arcs median 2.5 per trip, maximum 322 | `results/logs/eda_split_summary.csv`; `results/logs/eda_notes.md`; `src/pathsnap/network.py` |
| Anomaly rows: 00000054 (87%, 86 m, ring road about 100 m from arcs); 00000033 (59%, 7 m); 00000002 (12%, last 552 points, 56 m); 00000043 (7%, 27 m); 00000096 (4%, last 145 points, 76 m) | `results/logs/eda_notes.md`; `results/logs/eda_anomalies.csv` |
| Figure sample counts: 3 typical trips in each split and 3 worst anomalies; plotted thresholds 5 s, 30 m, 50 m, and 200 km/h | `scripts/eda.py`; Figures 2–5 and 9–10; captions above |
| Planned features and binary candidate target | `PLAN.md`, Phase 3 |

*Table 6. Number registry for the section. Exact split inventory byte/row values appear in Table 2; chart tick values are visible in the referenced source figures.*
