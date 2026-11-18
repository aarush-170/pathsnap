"""Compare a saved EDA rerun with the committed result artifacts.

Run from the repository root immediately after `python -m scripts.eda`.
"""
from io import BytesIO
from pathlib import Path
import re
import subprocess

import pandas as pd
from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "section2_eda_semantic_verification.csv"

paths = [
    *(Path("results/logs") / f for f in (
        "eda_anomalies.csv", "eda_file_inventory.csv", "eda_split_summary.csv", "eda_trip_summary.csv")),
    *(Path("results/figures") / f for f in (
        "eda_anomaly_distance_over_time.png", "eda_arc_length.png", "eda_data_quality.png",
        "eda_gps_to_route_distance.png", "eda_network_connectivity.png", "eda_points_per_trip.png",
        "eda_sampling_intervals.png", "eda_speed_distribution.png", "eda_trajectories_anomalies.png",
        "eda_trajectories_typical.png")),
    *(Path("results/maps") / f for f in ("eda_anomaly_train_00000054.html", "eda_train_00000095.html")),
]
rows = []
for rel in paths:
    committed = subprocess.check_output(["git", "show", f"HEAD:{rel.as_posix()}"], cwd=ROOT)
    rerun_path = ROOT / rel
    rerun = rerun_path.read_bytes()
    if rel.suffix == ".csv":
        pd.testing.assert_frame_equal(pd.read_csv(BytesIO(committed)), pd.read_csv(BytesIO(rerun)),
                                      check_dtype=False, check_exact=False, rtol=1e-12, atol=1e-12)
        method = "CSV values"
    elif rel.suffix == ".png":
        a = Image.open(BytesIO(committed)).convert("RGBA")
        b = Image.open(BytesIO(rerun)).convert("RGBA")
        if a.size != b.size or ImageChops.difference(a, b).getbbox() is not None:
            raise AssertionError(f"Rendered pixels differ: {rel}")
        method = "rendered pixels"
    else:
        pattern = rb"\b([a-z][a-z_]*_)[0-9a-f]{32}"
        a = re.sub(pattern, rb"\1<generated-id>", committed)
        b = re.sub(pattern, rb"\1<generated-id>", rerun)
        if a != b:
            raise AssertionError(f"HTML differs beyond generated map IDs: {rel}")
        method = "HTML after generated Folium ID normalization"
    rows.append({"file": rel.as_posix(), "comparison": method, "match": True,
                 "committed_bytes": len(committed), "rerun_bytes": len(rerun)})

pd.DataFrame(rows).to_csv(OUT, index=False)
print(f"PASS: {len(rows)} output artifacts match; details saved to {OUT}")
