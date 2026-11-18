"""Phase 3: feature tables for every (GPS point, candidate arc) pair, plus feature analysis.

  python -m scripts.build_features            # build tables + analysis
  python -m scripts.build_features --analyse  # analysis only (tables already built)

Needs data/processed/{labels,candidates}_{split}.parquet from scripts.build_dataset.
Outputs
  data/processed/features_{split}.parquet   trip, point_idx, arc, is_true, labelled, <FEATURES>
  results/logs/feature_univariate.csv       per-feature ROC-AUC and single-feature Top-1 (train)
  results/logs/rule_baselines.csv           simple hand-made scoring rules (no learning), both splits
  results/figures/feature_*.png
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
from sklearn.metrics import roc_auc_score

from pathsnap.candidates import trip_geometry
from pathsnap.config import load_config
from pathsnap.features import FEATURES, trip_features
from pathsnap.io import load_trip
from pathsnap.metrics import top1_accuracy

SPLITS = ("train", "student_test")


def build(cfg):
    proc = cfg["paths"]["processed"]
    raw = {"train": cfg["paths"]["raw_train"], "student_test": cfg["paths"]["raw_student_test"]}
    for split in SPLITS:
        t0 = time.time()
        C = pd.read_parquet(proc / f"candidates_{split}.parquet")
        L = pd.read_parquet(proc / f"labels_{split}.parquet", columns=["trip", "point_idx", "label_arc"])
        frames = []
        for tid, c in C.groupby("trip", sort=True):
            trip = load_trip(raw[split] / tid, split=split, load_route=False)   # features never see the route
            c = c.reset_index(drop=True)
            F = trip_features(trip, trip_geometry(trip), c)
            lab = L[L.trip == tid].set_index("point_idx").label_arc
            out = pd.concat([c[["trip", "point_idx", "arc", "is_true"]], F], axis=1)
            out["labelled"] = (lab.reindex(c.point_idx).to_numpy() >= 0)
            frames.append(out)
        D = pd.concat(frames, ignore_index=True)
        D.to_parquet(proc / f"features_{split}.parquet", index=False)
        print(f"  {split}: {len(D):,} rows x {len(FEATURES)} features, "
              f"{D.labelled.mean():.1%} rows from labelled points, {time.time() - t0:.1f}s")


def analyse(cfg):
    proc = cfg["paths"]["processed"]
    figs, logs = ROOT / "results" / "figures", ROOT / "results" / "logs"
    data = {s: pd.read_parquet(proc / f"features_{s}.parquet") for s in SPLITS}
    lab_pts = {s: int((pd.read_parquet(proc / f"labels_{s}.parquet", columns=["label_arc"]).label_arc >= 0).sum())
               for s in SPLITS}
    tr = data["train"][data["train"].labelled].copy()
    y = tr.is_true.to_numpy()

    # Feature sanity: missing / infinite values
    bad = {f: int((~np.isfinite(tr[f])).sum()) for f in FEATURES}
    print("non-finite values per feature (train):", {k: v for k, v in bad.items() if v} or "none")

    # Univariate: ROC-AUC (direction-free: max(auc, 1-auc)) and Top-1 when ranking by this feature alone
    lower_is_better = {"dist", "rank", "dist_minus_min", "dist_ratio_min", "dist_z", "heading_absdiff",
                       "heading_rank", "ctx_dist_mean", "ctx_dist_max", "ctx_minus_min", "ctx_rank"}
    rows = []
    for f in FEATURES:
        x = tr[f].to_numpy()
        auc = roc_auc_score(y, x) if np.nanstd(x) > 0 else 0.5
        tmp = tr[["trip", "point_idx", "rank", "is_true"]].copy()
        tmp["s"] = -x if f in lower_is_better else x
        top1 = top1_accuracy(tmp, "s", lab_pts["train"])
        rows.append(dict(feature=f, auc=auc, auc_abs=max(auc, 1 - auc), direction=("lower" if auc < 0.5 else "higher"),
                         top1_train_alone=top1, mean_true=x[y].mean(), mean_false=x[~y].mean()))
    U = pd.DataFrame(rows).sort_values("auc_abs", ascending=False)
    U.to_csv(logs / "feature_univariate.csv", index=False)
    print("\nUnivariate (train, labelled points):\n", U.round(4).to_string(index=False))

    # Hand-made scoring rules (no learning): baselines the trained models must beat
    rules = {
        "nearest arc (dist)": lambda d: -d.dist,
        "nearest + heading (dist - 10*cos)": lambda d: -d.dist + 10 * d.heading_cos,
        "context (ctx_dist_mean)": lambda d: -d.ctx_dist_mean,
        "context + heading": lambda d: -d.ctx_dist_mean + 10 * d.heading_cos,
    }
    rb = []
    for s in SPLITS:
        d = data[s][data[s].labelled].copy()
        for name, fn in rules.items():
            d["s"] = fn(d)
            rb.append(dict(split=s, rule=name, top1=top1_accuracy(d, "s", lab_pts[s])))
    RB = pd.DataFrame(rb).pivot(index="rule", columns="split", values="top1")[list(SPLITS)]
    RB.to_csv(logs / "rule_baselines.csv")
    print("\nRule baselines, observation-level Top-1:\n", RB.round(4).to_string())

    # Figures: AUC bar, distributions of the strongest features, correlation heatmap
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(U.feature, U.auc_abs, color=["#4c72b0" if a >= 0.5 else "#c44e52" for a in U.auc])
    ax.axvline(0.5, c="k", lw=0.8, ls="--")
    ax.invert_yaxis(); ax.set_xlim(0.45, 1.0)
    ax.set_xlabel("ROC-AUC of the single feature (direction-free)")
    ax.set_title("Univariate separability: true vs other candidates (train)")
    fig.tight_layout(); fig.savefig(figs / "feature_univariate_auc.png", dpi=150); plt.close(fig)

    top = U.feature.head(8).tolist()
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    samp = tr.sample(min(len(tr), 300_000), random_state=cfg["seed"])
    for ax, f in zip(axes.ravel(), top):
        x = samp[f]
        lo, hi = np.nanpercentile(x, [0.5, 99.5])
        bins = np.linspace(lo, hi, 50) if hi > lo else 10
        ax.hist(x[~samp.is_true], bins=bins, alpha=0.6, density=True, label="other candidates", color="#c44e52")
        ax.hist(x[samp.is_true], bins=bins, alpha=0.6, density=True, label="true arc", color="#4c72b0")
        ax.set_title(f, fontsize=10)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Feature distributions: true arc vs other candidates (train, labelled points)")
    fig.tight_layout(); fig.savefig(figs / "feature_distributions.png", dpi=150); plt.close(fig)

    corr = samp[FEATURES].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(FEATURES))); ax.set_xticklabels(FEATURES, rotation=90, fontsize=8)
    ax.set_yticks(range(len(FEATURES))); ax.set_yticklabels(FEATURES, fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Spearman correlation")
    ax.set_title("Feature correlation (train)")
    fig.tight_layout(); fig.savefig(figs / "feature_correlation.png", dpi=150); plt.close(fig)

    hi_corr = [(a, b, round(corr.loc[a, b], 3)) for i, a in enumerate(FEATURES) for b in FEATURES[i + 1:]
               if abs(corr.loc[a, b]) > 0.9]
    print("\n|Spearman| > 0.9 pairs:", hi_corr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyse", action="store_true", help="skip building, only run the analysis")
    args = ap.parse_args()
    cfg = load_config()
    if not args.analyse:
        build(cfg)
    analyse(cfg)


if __name__ == "__main__":
    main()
