"""Phase 7: Folium maps (and static PNGs for the report) of predicted vs reference routes.

  python -m scripts.make_maps                                  # default 3 representative student_test trips
  python -m scripts.make_maps --trips 00000077 00000009
  python -m scripts.make_maps --all

Each interactive map (results/maps/route_<trip>.html) has toggleable layers:
  road network (grey) | reference route (green) | HMM-predicted route (red, dashed)
  classifier-only route (purple, off by default) | GPS points (blue) | points still mismatched (orange)
plus a legend, start/end markers and a stats box. Predicted routes are made drivable by inserting the
shortest connecting path between consecutive predicted arcs (hmm.reconstruct_route).
Each static figure (results/figures/map_<trip>.png) shows the full trip and a zoom on the stretch where
the HMM fixed most classifier-only mistakes.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection

from pathsnap.config import load_config
from pathsnap.hmm import HMMParams, decode_trip, reconstruct_route
from pathsnap.viz import COLORS, trip_map
from scripts.eval_hmm import classifier_only, prepare, score_trip

MAPS, FIG = ROOT / "results" / "maps", ROOT / "results" / "figures"
DEFAULT_TRIPS = ["00000077", "00000040", "00000007"]   # hardest, typical, best (see phase6_notes.md)


def hmm_params(cfg) -> HMMParams:
    h = cfg["hmm"]
    return HMMParams(beta=h["beta"], alpha=h["alpha"], gap_break_s=h["gap_break_seconds"],
                     route_factor=h["route_factor"], route_slack_m=h["route_slack_m"])


def wrong_points(info, pred: pd.Series) -> np.ndarray:
    lab = info["labels"]
    lab = lab[lab.label_arc >= 0]
    common = pred.index.intersection(lab.index)
    canon = info["canon"]
    bad = canon[pred.loc[common].to_numpy()] != canon[lab.loc[common, "label_arc"].to_numpy()]
    return np.asarray(common[bad])


def static_figure(tid, info, clf_route, hmm_route, wrong_clf, wrong_hmm, s_clf, s_hmm, out):
    P, g, arcs = info["P"], info["graph"], info["trip"].arcs
    N = np.vstack([info["trip"].nodes[:, 0], info["trip"].nodes[:, 1]]).T  # unused; projection below
    from pathsnap.candidates import trip_geometry
    NM = trip_geometry(info["trip"]).N

    def segs(ids):
        ids = [a for a in ids if a >= 0]
        return np.stack([NM[arcs[ids, 0]], NM[arcs[ids, 1]]], 1) if ids else np.zeros((0, 2, 2))

    fixed = np.setdiff1d(wrong_clf, wrong_hmm)
    if len(fixed):   # zoom on the 40-point window with the most fixes
        idx = np.sort(fixed)
        counts = [np.sum((idx >= i) & (idx < i + 40)) for i in idx]
        c0 = idx[int(np.argmax(counts))]
        win = P[c0:c0 + 40]
    else:
        win = P[: min(len(P), 40)]
    lo, hi = win.min(0) - 60, win.max(0) + 60

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.8))
    titles = [f"Full trip: HMM route (Top-1 {s_hmm['top1']:.1%})",
              f"Zoom: classifier only (Top-1 {s_clf['top1']:.1%})", f"Zoom: classifier + HMM (Top-1 {s_hmm['top1']:.1%})"]
    for k, ax in enumerate(axes):
        full = k == 0
        a_lo, a_hi = (P.min(0) - 150, P.max(0) + 150) if full else (lo, hi)
        net = segs(range(len(arcs)))
        keep = ((net[:, :, 0].max(1) >= a_lo[0]) & (net[:, :, 0].min(1) <= a_hi[0]) &
                (net[:, :, 1].max(1) >= a_lo[1]) & (net[:, :, 1].min(1) <= a_hi[1]))
        ax.add_collection(LineCollection(net[keep], colors=COLORS["network"], linewidths=0.5, alpha=0.6,
                                         label="road network"))
        ax.add_collection(LineCollection(segs(info["trip"].route), colors=COLORS["route"], linewidths=4.5,
                                         alpha=0.55, label="reference route"))
        route = hmm_route if k != 1 else clf_route
        ax.add_collection(LineCollection(segs(route), colors=COLORS["baseline"] if k == 1 else COLORS["pred"],
                                         linewidths=1.6, label="classifier-only route" if k == 1 else "HMM route"))
        ax.scatter(P[:, 0], P[:, 1], s=4 if full else 10, c=COLORS["gps"], alpha=0.5, lw=0, label="GPS", zorder=3)
        wp = wrong_clf if k == 1 else wrong_hmm
        ax.scatter(P[wp, 0], P[wp, 1], s=10 if full else 28, facecolors="none", edgecolors=COLORS["wrong"],
                   lw=1.0, label="mismatched point", zorder=4)
        if full:
            ax.add_patch(plt.Rectangle(lo, *(hi - lo), fill=False, ec="k", lw=1, ls="--"))
        ax.set_xlim(a_lo[0], a_hi[0]); ax.set_ylim(a_lo[1], a_hi[1]); ax.set_aspect("equal")
        ax.set_title(titles[k], fontsize=10); ax.set_xticks([]); ax.set_yticks([])
    h1, l1 = axes[1].get_legend_handles_labels()
    h2, l2 = axes[2].get_legend_handles_labels()
    hs = dict(zip(l1 + l2, h1 + h2))
    fig.legend(hs.values(), hs.keys(), loc="lower center", ncol=6, fontsize=9)
    fig.suptitle(f"student_test trip {tid}: dashed box = zoom where the HMM fixed {len(fixed)} classifier mistakes",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default="results/models/lightgbm_tuned/pred_student_test.parquet")
    ap.add_argument("--trips", nargs="*", default=DEFAULT_TRIPS)
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    MAPS.mkdir(parents=True, exist_ok=True)
    trips = prepare(cfg, "student_test", ROOT / args.preds)
    params = hmm_params(cfg)
    rows = []
    for tid in (sorted(trips) if args.all else args.trips):
        info = trips[tid]
        clf = classifier_only(info)
        hmm, diag = decode_trip(info["graph"], info["P"], info["ts"], info["cands"], info["cands"].p.to_numpy(), params)
        s_clf, s_hmm = score_trip(info, clf), score_trip(info, hmm)
        clf_route = reconstruct_route(info["graph"], clf.sort_index().to_numpy())
        hmm_route = reconstruct_route(info["graph"], hmm.sort_index().to_numpy())
        w_clf, w_hmm = wrong_points(info, clf), wrong_points(info, hmm)
        stats = (f"Top-1: classifier {s_clf['top1']:.1%} &rarr; HMM <b>{s_hmm['top1']:.1%}</b><br>"
                 f"Mismatched points: {len(w_clf)} &rarr; <b>{len(w_hmm)}</b> of {s_hmm['points']}<br>"
                 f"Continuity: {s_clf['continuity']:.1%} &rarr; {s_hmm['continuity']:.1%}"
                 f" &nbsp;|&nbsp; HMM segments: {diag['segments']}")
        m = trip_map(info["trip"], predicted_route=hmm_route, wrong_points=w_hmm, baseline_route=clf_route,
                     predicted_label="HMM-predicted route", title=f"PathSnap: student_test trip {tid}",
                     stats_html=stats)
        out = MAPS / f"route_{tid}.html"
        m.save(out)
        static_figure(tid, info, clf_route, hmm_route, w_clf, w_hmm, s_clf, s_hmm, FIG / f"map_{tid}.png")
        rows.append(dict(trip=tid, top1_clf=s_clf["top1"], top1_hmm=s_hmm["top1"], wrong_clf=len(w_clf),
                         wrong_hmm=len(w_hmm), fixed=len(np.setdiff1d(w_clf, w_hmm)),
                         broken=len(np.setdiff1d(w_hmm, w_clf)), route_gaps=hmm_route.count(-1),
                         map=str(out.relative_to(ROOT))))
        print(f"{tid}: top1 {s_clf['top1']:.4f} -> {s_hmm['top1']:.4f}, mismatches {len(w_clf)} -> {len(w_hmm)} "
              f"(fixed {rows[-1]['fixed']}, newly wrong {rows[-1]['broken']}), route gaps {rows[-1]['route_gaps']}")
    pd.DataFrame(rows).to_csv(ROOT / "results" / "logs" / "maps_summary.csv", index=False)


if __name__ == "__main__":
    main()
