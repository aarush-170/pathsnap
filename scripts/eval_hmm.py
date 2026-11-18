"""Phase 6: classifier-only vs classifier + HMM/Viterbi.

  python -m scripts.eval_hmm --preds results/models/lightgbm/pred_student_test.parquet --grid
  python -m scripts.eval_hmm --preds results/models/lightgbm_tuned/pred_student_test.parquet --beta 10 --alpha 1

`--preds` is a table (trip, point_idx, arc, p) of classifier probabilities for the student_test
candidates. With --grid, several (beta, alpha) settings are compared.
Outputs (tag = name of the folder holding the predictions)
  results/logs/hmm_grid_<tag>.csv, hmm_comparison_<tag>.csv, hmm_per_trip_<tag>.csv
"""
from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from pathsnap.candidates import trip_geometry
from pathsnap.config import load_config
from pathsnap.hmm import HMMParams, RoadGraph, decode_trip
from pathsnap.io import load_trip
from pathsnap.labels import canonical_arc_ids
from pathsnap.metrics import route_continuity

LOG = ROOT / "results" / "logs"


def prepare(cfg, split, preds_path):
    C = pd.read_parquet(cfg["paths"]["processed"] / f"candidates_{split}.parquet")
    L = pd.read_parquet(cfg["paths"]["processed"] / f"labels_{split}.parquet")
    pr = pd.read_parquet(preds_path)
    pr["trip"] = pr.trip.astype(str)
    D = C.merge(pr[["trip", "point_idx", "arc", "p"]], on=["trip", "point_idx", "arc"], how="inner")
    trips = {}
    raw = cfg["paths"]["raw_" + split] if "raw_" + split in cfg["paths"] else cfg["paths"]["raw_student_test"]
    for tid, d in D.groupby("trip"):
        trip = load_trip(raw / tid, split=split)
        g = trip_geometry(trip)
        trips[tid] = dict(trip=trip, P=g.P, ts=trip.track.t.to_numpy(), graph=RoadGraph(trip.arcs, g.N),
                          cands=d.sort_values(["point_idx", "arc"]).reset_index(drop=True),
                          labels=L[L.trip == tid].set_index("point_idx"),
                          canon=canonical_arc_ids(trip.arcs))
    return trips


def score_trip(info, pred: pd.Series) -> dict:
    """Point-level and route-level metrics of a per-point arc prediction for one trip."""
    lab = info["labels"]
    canon = info["canon"]
    lab = lab[lab.label_arc >= 0]
    common = pred.index.intersection(lab.index)
    hit = canon[pred.loc[common].to_numpy()] == canon[lab.loc[common, "label_arc"].to_numpy()]
    true_set = set(canon[info["trip"].route])
    pred_set = set(canon[pred.to_numpy()])
    tp = len(true_set & pred_set)
    return dict(points=len(lab), correct=int(hit.sum()), top1=hit.sum() / max(len(lab), 1),
                continuity=route_continuity(info["trip"].arcs, pred.sort_index().to_numpy()),
                route_precision=tp / max(len(pred_set), 1), route_recall=tp / max(len(true_set), 1))


def classifier_only(info) -> pd.Series:
    c = info["cands"]
    top = c.sort_values(["point_idx", "p", "rank"], ascending=[True, False, True]).drop_duplicates("point_idx")
    return top.set_index("point_idx").arc


def run(trips, params: HMMParams | None):
    rows, t0, segs = [], time.perf_counter(), 0
    for tid, info in trips.items():
        if params is None:
            pred = classifier_only(info)
        else:
            pred, diag = decode_trip(info["graph"], info["P"], info["ts"], info["cands"],
                                     info["cands"].p.to_numpy(), params)
            segs += diag["segments"]
        rows.append(dict(trip=tid, **score_trip(info, pred)))
    per = pd.DataFrame(rows)
    tot = dict(top1=per.correct.sum() / per.points.sum(), continuity=per.continuity.mean(),
               route_precision=per.route_precision.mean(), route_recall=per.route_recall.mean(),
               seconds=time.perf_counter() - t0, segments=segs)
    tot["route_f1"] = 2 * tot["route_precision"] * tot["route_recall"] / (tot["route_precision"] + tot["route_recall"])
    return tot, per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--split", default="student_test")
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--beta", type=float, default=10.0)
    ap.add_argument("--alpha", type=float, default=1.0)
    args = ap.parse_args()
    cfg = load_config()
    tag = Path(args.preds).parent.name
    trips = prepare(cfg, args.split, ROOT / args.preds)
    print(f"{len(trips)} trips, {sum(len(t['cands']) for t in trips.values()):,} candidate rows")

    base, base_per = run(trips, None)
    print(f"classifier-only: top1 {base['top1']:.4f} continuity {base['continuity']:.4f} "
          f"route P/R {base['route_precision']:.3f}/{base['route_recall']:.3f}")

    settings = list(itertools.product([2, 5, 10, 20, 50], [0.5, 1.0, 2.0])) if args.grid else [(args.beta, args.alpha)]
    grid = []
    best = None
    for beta, alpha in settings:
        res, per = run(trips, HMMParams(beta=beta, alpha=alpha, gap_break_s=cfg["hmm"].get("gap_break_seconds", 60)))
        grid.append(dict(beta=beta, alpha=alpha, **res))
        print(f"HMM beta={beta:>4} alpha={alpha:>3}: top1 {res['top1']:.4f} continuity {res['continuity']:.4f} "
              f"route P/R {res['route_precision']:.3f}/{res['route_recall']:.3f} segments {res['segments']} "
              f"({res['seconds']:.1f}s)", flush=True)
        if best is None or res["top1"] > best[0]["top1"]:
            best = (res, per, beta, alpha)
    G = pd.DataFrame(grid)
    G.to_csv(LOG / f"hmm_grid_{tag}.csv", index=False)
    res, per, beta, alpha = best
    comp = pd.DataFrame([dict(method="classifier only", **base),
                         dict(method=f"classifier + HMM (beta={beta}, alpha={alpha})", **res)])
    comp.to_csv(LOG / f"hmm_comparison_{tag}.csv", index=False)
    pt = base_per.merge(per, on="trip", suffixes=("_clf", "_hmm"))
    pt["top1_gain"] = pt.top1_hmm - pt.top1_clf
    pt.to_csv(LOG / f"hmm_per_trip_{tag}.csv", index=False)
    pd.set_option("display.width", 200)
    print("\n", comp.round(4).to_string(index=False))
    print("\nper trip:\n", pt[["trip", "points_clf", "top1_clf", "top1_hmm", "top1_gain", "continuity_clf",
                              "continuity_hmm"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
