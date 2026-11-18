"""Recompute compact Section 2 tables from committed EDA CSVs.

Run from the repository root with: python results/verify_section2.py
Outputs results/section2_inventory_summary.csv and section2_key_numbers.csv.
These summarize committed EDA evidence; raw-data regeneration requires data/raw.
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "results" / "logs"
OUT = ROOT / "results"

inv = pd.read_csv(LOG / "eda_file_inventory.csv")
summary = (inv.groupby(["split", "file"], sort=True)
          .agg(trips=("trip", "nunique"), files_exist=("exists", "sum"),
               bytes=("bytes", "sum"), rows=("rows", "sum"))
          .reset_index())
summary.to_csv(OUT / "section2_inventory_summary.csv", index=False)

trips = pd.read_csv(LOG / "eda_trip_summary.csv")
keys = {
    "n_points": "GPS points per trip",
    "n_nodes": "network nodes per trip",
    "n_arcs": "network arcs per trip",
    "duplicate_arcs": "duplicate directed arcs per trip",
    "two_way_frac": "fraction of arcs with reverse twin per trip",
    "n_weak_components": "weak components per trip",
    "largest_scc_frac": "largest SCC node fraction per trip",
    "route_breaks": "route breaks per trip",
    "frac_points_gt_50m": "fraction GPS >50 m from route per trip",
    "gps_route_dist_median_m": "median point-to-route distance per trip (m)",
    "gps_route_dist_p95_m": "95th percentile point-to-route distance per trip (m)",
    "frac_dt_eq_1": "fraction 1-second intervals per trip",
    "max_gap_s": "maximum sampling gap per trip (s)",
    "speed_jumps_gt_200kmh": "steps above 200 km/h per trip",
    "zero_length_arcs": "zero-length arcs per trip",
}
rows = []
for col, label in keys.items():
    if col not in trips:
        continue
    for split, group in trips.groupby("split", sort=True):
        s = group[col].dropna()
        rows.append({"split": split, "metric": label, "source_column": col,
                     "count": len(s), "sum": s.sum(), "median": s.median(),
                     "min": s.min(), "max": s.max()})
pd.DataFrame(rows).to_csv(OUT / "section2_key_numbers.csv", index=False)
print(f"Wrote {OUT / 'section2_inventory_summary.csv'}")
print(f"Wrote {OUT / 'section2_key_numbers.csv'}")
