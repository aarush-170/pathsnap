"""Phase 2: per-point labels + candidate generation.

  python -m scripts.build_dataset --sweep   # candidate radius / K sweep -> choose values for config.yaml
  python -m scripts.build_dataset           # build final tables using config.yaml candidates.radius_m / max_k

Outputs
  --sweep: results/logs/candidate_sweep.csv, results/figures/candidates_miss_rate.png
  build:   data/processed/labels_{split}.parquet       one row per GPS point
           data/processed/candidates_{split}.parquet   one row per (point, candidate arc), with is_true
           results/logs/label_stats.csv                per-trip label / candidate statistics
           results/logs/candidate_summary.csv          per-split candidate miss rate and set sizes
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathsnap.candidates import generate_candidates, trip_geometry
from pathsnap.config import load_config
from pathsnap.io import iter_trips
from pathsnap.labels import align_route, canonical_arc_ids

RADII = [10, 15, 20, 25, 30, 40, 50, 75, 100]
KS = [1, 2, 4, 6, 8, 10, 12, 16, 10_000]


def label_trip(trip, cfg):
    lc = cfg["labels"]
    return align_route(trip, max_dist=lc.get("max_dist_m", 50.0), lam=lc.get("progress_weight", 0.02))


def mark_true(cands: pd.DataFrame, trip, labels) -> pd.Series:
    canon = canonical_arc_ids(trip.arcs)
    lab = labels.label_arc[cands.point_idx.to_numpy()]
    is_true = (lab >= 0) & (canon[cands.arc.to_numpy()] == canon[np.where(lab >= 0, lab, 0)])
    return pd.Series(is_true, index=cands.index)


# --------------------------------------------------------------------------------------- sweep
def sweep(cfg, splits):
    rows = []
    for split, d in splits.items():
        acc = {(r, k): dict(miss=0, cands=0, empty=0) for r in RADII for k in KS}
        n_lab = 0
        t0 = time.time()
        for trip in iter_trips(d, split=split):
            labels = label_trip(trip, cfg)
            c = generate_candidates(trip, radius=max(RADII))
            c["is_true"] = mark_true(c, trip, labels)
            labelled = np.where(labels.label_arc >= 0)[0]
            n_lab += len(labelled)
            tr = c[c.is_true].groupby("point_idx").agg(d=("dist", "min"), r=("rank", "min"))
            true_d = np.full(len(trip.track), np.inf)
            true_r = np.full(len(trip.track), 10 ** 9)
            true_d[tr.index] = tr.d
            true_r[tr.index] = tr.r
            true_d, true_r = true_d[labelled], true_r[labelled]
            p, dist, rank = c.point_idx.to_numpy(), c.dist.to_numpy(), c["rank"].to_numpy()
            is_lab = labels.label_arc[p] >= 0
            for r in RADII:
                for k in KS:
                    m = is_lab & (dist <= r) & (rank < k)
                    cnt = np.bincount(p[m], minlength=len(trip.track))[labelled]
                    a = acc[(r, k)]
                    a["miss"] += int(((true_d > r) | (true_r >= k)).sum())
                    a["cands"] += int(cnt.sum())
                    a["empty"] += int((cnt == 0).sum())
        for (r, k), a in acc.items():
            rows.append(dict(split=split, radius_m=r, max_k=("all" if k >= 10_000 else k),
                             miss_rate=a["miss"] / n_lab, mean_candidates=a["cands"] / n_lab,
                             frac_points_no_candidates=a["empty"] / n_lab, labelled_points=n_lab))
        print(f"  {split}: swept in {time.time() - t0:.1f}s ({n_lab} labelled points)")
    df = pd.DataFrame(rows)
    out = ROOT / "results" / "logs" / "candidate_sweep.csv"
    df.to_csv(out, index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    tr = df[df.split == "train"]
    for k, g in tr.groupby("max_k", sort=False):
        axes[0].plot(g.radius_m, g.miss_rate * 100, marker="o", ms=3, label=f"K={k}")
    axes[0].set_xlabel("candidate radius (m)"); axes[0].set_ylabel("candidate miss rate (%)")
    axes[0].set_yscale("log"); axes[0].set_title("Train: true arc missing from candidate set")
    axes[0].legend(ncol=3, fontsize=8)
    for r, g in tr.groupby("radius_m"):
        axes[1].plot(g.mean_candidates, g.miss_rate * 100, marker="o", ms=3, label=f"R={r} m")
    axes[1].set_xscale("log"); axes[1].set_yscale("log")
    axes[1].set_xlabel("mean candidates per labelled point"); axes[1].set_ylabel("miss rate (%)")
    axes[1].set_title("Train: miss rate vs candidate-set size (points = K values)")
    axes[1].legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(ROOT / "results" / "figures" / "candidates_miss_rate.png", dpi=150)
    plt.close(fig)
    pd.set_option("display.width", 200)
    piv = df.pivot_table(index=["split", "radius_m"], columns="max_k", values="miss_rate", sort=False)
    print("\nMiss rate (%):\n", (piv * 100).round(3).to_string())
    piv = df.pivot_table(index=["split", "radius_m"], columns="max_k", values="mean_candidates", sort=False)
    print("\nMean candidates per point:\n", piv.round(1).to_string())


# --------------------------------------------------------------------------------------- build
def build(cfg, splits):
    radius = float(cfg["candidates"]["radius_m"])
    max_k = cfg["candidates"].get("max_k")
    out_dir = cfg["paths"]["processed"]
    out_dir.mkdir(parents=True, exist_ok=True)
    stats, summary = [], []
    for split, d in splits.items():
        lab_frames, cand_frames = [], []
        t0 = time.time()
        for trip in iter_trips(d, split=split):
            g = trip_geometry(trip)
            labels = label_trip(trip, cfg)
            c = generate_candidates(trip, radius=radius, max_rank=max_k, geom=g)
            c["is_true"] = mark_true(c, trip, labels)
            c.insert(0, "trip", trip.trip_id)
            n = len(trip.track)
            lab = pd.DataFrame({"trip": trip.trip_id, "point_idx": np.arange(n, dtype=np.int32),
                                "t": trip.track.t.values, "label_arc": labels.label_arc,
                                "route_pos": labels.route_pos, "label_dist": labels.label_dist,
                                "nearest_route_arc": labels.nearest_arc})
            ncand = np.bincount(c.point_idx, minlength=n)
            has_true = np.bincount(c.point_idx[c.is_true], minlength=n) > 0
            lab["n_candidates"] = ncand
            lab["true_in_candidates"] = has_true
            lab_frames.append(lab)
            cand_frames.append(c)

            m = labels.label_arc >= 0
            a, b = labels.label_arc[m], labels.nearest_arc[m]
            twin = (trip.arcs[a, 0] == trip.arcs[b, 1]) & (trip.arcs[a, 1] == trip.arcs[b, 0])
            stats.append(dict(split=split, trip=trip.trip_id, n_points=n, labelled_frac=m.mean(),
                              route_arcs=len(trip.route),
                              route_arcs_with_points=len(np.unique(labels.route_pos[m])) / len(trip.route),
                              label_monotonic=bool(np.all(np.diff(labels.route_pos[m]) >= 0)),
                              unlabelled_short_runs=labels.dropped_short_runs,
                              differs_from_nearest=float(np.mean(a != b)) if m.any() else np.nan,
                              differs_reverse_twin=float(np.mean((a != b) & twin)) if m.any() else np.nan,
                              mean_candidates=ncand[m].mean() if m.any() else np.nan,
                              miss_rate=float(np.mean(~has_true[m])) if m.any() else np.nan,
                              points_no_candidates=int((ncand == 0).sum())))
        L = pd.concat(lab_frames, ignore_index=True)
        C = pd.concat(cand_frames, ignore_index=True)
        L.to_parquet(out_dir / f"labels_{split}.parquet", index=False)
        C.to_parquet(out_dir / f"candidates_{split}.parquet", index=False)
        lm = L.label_arc >= 0
        summary.append(dict(split=split, radius_m=radius, max_k=max_k, trips=L.trip.nunique(), points=len(L),
                            labelled_points=int(lm.sum()), labelled_frac=float(lm.mean()),
                            candidate_rows=len(C), positives=int(C.is_true.sum()),
                            mean_candidates_per_point=float(L.n_candidates[lm].mean()),
                            candidate_miss_rate=float((~L.true_in_candidates[lm]).mean()),
                            points_no_candidates=int((L.n_candidates == 0).sum())))
        print(f"  {split}: {len(L)} points, {len(C)} candidate rows in {time.time() - t0:.1f}s")
    pd.DataFrame(stats).to_csv(ROOT / "results" / "logs" / "label_stats.csv", index=False)
    s = pd.DataFrame(summary)
    s.to_csv(ROOT / "results" / "logs" / "candidate_summary.csv", index=False)
    print(s.T.to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true", help="run the radius/K sweep instead of building tables")
    args = ap.parse_args()
    cfg = load_config()
    splits = {"train": cfg["paths"]["raw_train"], "student_test": cfg["paths"]["raw_student_test"]}
    (ROOT / "results" / "logs").mkdir(parents=True, exist_ok=True)
    (ROOT / "results" / "figures").mkdir(parents=True, exist_ok=True)
    sweep(cfg, splits) if args.sweep else build(cfg, splits)


if __name__ == "__main__":
    main()
