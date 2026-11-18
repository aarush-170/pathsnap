# Phase 7 notes: geographical visualisation (Folium)

Reproduce with `python -m scripts.make_maps` (add `--all` for all 10 student_test trips). This uses the tuned LightGBM + HMM (β = 5, α = 0.5).
Code: `scripts/make_maps.py`, `src/pathsnap/viz.py` (`trip_map`), `src/pathsnap/hmm.py` (`reconstruct_route`).

## Interactive maps (`results/maps/route_<trip>.html`)
Three representative student_test trips:

| Trip | Why chosen | Top-1 classifier → HMM | Mismatched points | Fixed by HMM | Newly wrong |
|---|---|---|---|---|---|
| 00000077 | hardest trip | 74.1% → 88.1% | 1,012 → 467 | 654 | 109 |
| 00000040 | typical | 95.3% → 97.7% | 141 → 70 | 82 | 11 |
| 00000007 | best | 94.7% → 100.0% | 72 → 0 | 72 | 0 |

**Each map has** (Deliverables §6):
- **Layers you can toggle:**
  - road network near the trip: grey
  - reference route: green, wide
  - HMM-predicted route: red, dashed
  - classifier-only route: purple, dotted, **off by default**, so you can switch between before and after the HMM
  - GPS observations: blue dots
  - points still mismatched: orange
- **Markers and legend:** start and end markers, a legend, and a stats box (Top-1, mismatches and continuity, before and after the HMM).
- **Framing:** the map is fitted to the trip on an OpenStreetMap base map.

The predicted routes are made **drivable**. Consecutive predicted arcs are joined by the shortest connecting path, which fills in short arcs that fall between 1 Hz samples, and a break is shown if no path exists. All three routes reconstructed with **0 breaks**.

## Static figures for the PDF (`results/figures/map_<trip>.png`)
Each figure has three panels: the full trip with the HMM route, then a zoom on the 40-point stretch where the HMM fixed the most classifier mistakes, shown first with the classifier-only route and then with the HMM route.

## What the maps show
- **Where errors happen.** On trip 77 most errors cluster where the **vehicle is stopped at a junction**. A dense blob of GPS points builds up while it waits, the heading is unknown (`heading_valid` = 0), and every road meeting at the junction is equally close. The classifier wanders onto the crossing road; the HMM keeps most of those points on the connected route.
- **Where the HMM fails or hurts.** Some points directly on the junction remain wrong. On trip 77 the HMM also made 109 points wrong that the classifier had right, mostly at the moment of turning, when the decoder switches road a few seconds early or late. That is a limitation of decoding at 1 Hz with a single transition-tolerance setting.
- **Where the HMM helps.** Classifier mistakes that are isolated jumps onto a parallel or crossing road disappear completely (trip 7 goes to 100%).

## For the report
- Include at least one map as a screenshot: open `results/maps/route_00000077.html` in a browser, zoom into a junction, and switch on the classifier-only layer for a before/after view. Or use `results/figures/map_00000077.png`.
- The map-generation code is in the repo (`scripts/make_maps.py`), as required.
