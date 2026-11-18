"""Phase 4: train and compare candidate classifiers.

  python -m scripts.train_models                      # all models, then the comparison report
  python -m scripts.train_models --models lightgbm mlp
  python -m scripts.train_models --report             # only rebuild table + plots from saved results
  python -m scripts.train_models --sample 0.05        # quick smoke test on 5% of trips' points

Protocol (identical for every model)
  * fit data: train split, rows from labelled points only (results/logs/phase3_notes.md)
  * 5-fold GroupKFold by trip on the 80 train trips (fold assignment saved to results/logs/cv_folds.csv)
  * refit on all train trips, evaluate once on the 10 student_test trips
  * metrics: observation-level Top-1 accuracy (argmax of P(is_true) per point), row-level Macro F1,
    precision, recall (threshold 0.5), log loss, ROC-AUC, route continuity of the Top-1 sequence,
    training and inference time
Outputs
  results/models/<model>/metrics.json, cv_folds.csv, confusion_test.csv, (mlp) history.csv
  results/logs/model_comparison.csv, results/figures/model_*.png
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (confusion_matrix, f1_score, log_loss, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import GroupKFold

from pathsnap.config import load_config
from pathsnap.features import FEATURES
from pathsnap.io import load_trip
from pathsnap.metrics import route_continuity, top1_predictions
from pathsnap.models import registry, timed

MODELS_DIR = ROOT / "results" / "models"
LOG, FIG = ROOT / "results" / "logs", ROOT / "results" / "figures"


def load_split(cfg, split, sample=None, seed=42):
    df = pd.read_parquet(cfg["paths"]["processed"] / f"features_{split}.parquet")
    df = df[df.labelled].reset_index(drop=True)
    if sample:
        pts = df[["trip", "point_idx"]].drop_duplicates().sample(frac=sample, random_state=seed)
        df = df.merge(pts, on=["trip", "point_idx"]).reset_index(drop=True)
    df["rank"] = df["rank"].astype(float)
    return df


def evaluate(df: pd.DataFrame, p: np.ndarray, arcs_by_trip: dict | None = None) -> dict:
    y = df.is_true.to_numpy()
    p = np.clip(p, 1e-7, 1 - 1e-7)
    yhat = p >= 0.5
    d = df[["trip", "point_idx", "rank", "arc", "is_true"]].copy()
    d["p"] = p
    top = top1_predictions(d, "p")
    out = dict(
        top1=float(top.is_true.mean()),
        macro_f1=float(f1_score(y, yhat, average="macro")),
        precision_macro=float(precision_score(y, yhat, average="macro", zero_division=0)),
        recall_macro=float(recall_score(y, yhat, average="macro", zero_division=0)),
        f1_pos=float(f1_score(y, yhat)),
        log_loss=float(log_loss(y, p, labels=[False, True])),
        roc_auc=float(roc_auc_score(y, p)),
        n_points=int(len(top)), n_rows=int(len(df)),
    )
    if arcs_by_trip is not None:
        cont = [route_continuity(arcs_by_trip[t], g.sort_values("point_idx").arc.to_numpy())
                for t, g in top.groupby("trip")]
        out["route_continuity"] = float(np.nanmean(cont))
    return out


def run_model(spec, tr, te, folds, arcs_te, out_dir: Path):
    X, y, g = tr[FEATURES].to_numpy(np.float32), tr.is_true.to_numpy(), tr.trip.to_numpy()
    fold_rows = []
    for k in sorted(folds.fold.unique()):
        val_trips = set(folds.trip[folds.fold == k])
        vm = np.isin(g, list(val_trips))
        m = spec.build()
        _, t_fit = timed(m.fit, X[~vm], y[~vm], g[~vm])
        p_va, t_inf = timed(m.predict_proba, X[vm])
        r = evaluate(tr[vm], p_va)
        # training-fold fit quality (for over/underfitting), on a fixed ~200k-row sample of whole points
        tr_fit = tr[~vm]
        keys = tr_fit[["trip", "point_idx"]].drop_duplicates()
        frac = min(1.0, 200_000 / max(len(tr_fit), 1))
        keys = keys.sample(frac=frac, random_state=0)
        sel = tr_fit.reset_index().merge(keys, on=["trip", "point_idx"])["index"].to_numpy()
        r_tr = evaluate(tr.loc[sel], m.predict_proba(X[sel]))
        fold_rows.append(dict(fold=k, **r, train_top1=r_tr["top1"], train_macro_f1=r_tr["macro_f1"],
                              train_log_loss=r_tr["log_loss"], fit_s=t_fit, infer_s=t_inf,
                              epochs=len(m.history) if m.history else None))
        print(f"    fold {k}: top1={r['top1']:.4f} macroF1={r['macro_f1']:.4f} logloss={r['log_loss']:.4f} "
              f"(train top1={r_tr['top1']:.4f}) fit={t_fit:.1f}s", flush=True)
    cv = pd.DataFrame(fold_rows)
    cv.to_csv(out_dir / "cv_folds.csv", index=False)

    # refit on all train trips, evaluate on student_test
    m = spec.build()
    _, t_fit = timed(m.fit, X, y, g)
    Xte = te[FEATURES].to_numpy(np.float32)
    p_te, t_inf = timed(m.predict_proba, Xte)
    test = evaluate(te, p_te, arcs_te)
    cm = confusion_matrix(te.is_true.to_numpy(), p_te >= 0.5, labels=[False, True])
    pd.DataFrame(cm, index=["true_other", "true_correct"], columns=["pred_other", "pred_correct"]) \
        .to_csv(out_dir / "confusion_test.csv")
    if m.history:
        pd.DataFrame(m.history).to_csv(out_dir / "history.csv", index=False)
    pd.DataFrame({"trip": te.trip, "point_idx": te.point_idx, "arc": te.arc, "p": p_te}) \
        .to_parquet(out_dir / "pred_student_test.parquet", index=False)

    metrics = dict(
        model=spec.name, family=spec.family, config=spec.config, note=spec.note,
        cv={c: dict(mean=float(cv[c].mean()), std=float(cv[c].std()))
            for c in ["top1", "macro_f1", "precision_macro", "recall_macro", "log_loss", "roc_auc",
                      "train_top1", "train_macro_f1", "train_log_loss", "fit_s", "infer_s"]},
        test=test, test_fit_s=t_fit, test_infer_s=t_inf,
        test_infer_ms_per_1k_points=1e6 * t_inf / test["n_points"],   # seconds -> ms, per 1,000 points
    )
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    return metrics


# ------------------------------------------------------------------------------------ report
def report():
    rows, folds_all, hist = [], [], {}
    for mdir in sorted(MODELS_DIR.iterdir()):
        f = mdir / "metrics.json"
        if not f.exists():
            continue
        m = json.loads(f.read_text())
        cv, te = m["cv"], m["test"]
        rows.append({
            "model": m["model"], "family": m["family"],
            "config": json.dumps(m["config"]),
            "cv_top1": cv["top1"]["mean"], "cv_top1_std": cv["top1"]["std"],
            "cv_macro_f1": cv["macro_f1"]["mean"], "cv_macro_f1_std": cv["macro_f1"]["std"],
            "cv_log_loss": cv["log_loss"]["mean"], "cv_train_top1": cv["train_top1"]["mean"],
            "test_top1": te["top1"], "test_macro_f1": te["macro_f1"], "test_precision": te["precision_macro"],
            "test_recall": te["recall_macro"], "test_log_loss": te["log_loss"], "test_roc_auc": te["roc_auc"],
            "test_route_continuity": te.get("route_continuity"),
            "fit_s_full_train": m["test_fit_s"], "infer_ms_per_1k_points": m["test_infer_ms_per_1k_points"],
            "note": m.get("note", ""),
        })
        c = pd.read_csv(mdir / "cv_folds.csv")
        c.insert(0, "model", m["model"])
        folds_all.append(c)
        if (mdir / "history.csv").exists():
            hist[m["model"]] = pd.read_csv(mdir / "history.csv")
    if not rows:
        print("no model results yet")
        return
    T = pd.DataFrame(rows).sort_values("test_top1", ascending=False)
    T.to_csv(LOG / "model_comparison.csv", index=False)
    pd.concat(folds_all).to_csv(LOG / "model_cv_folds.csv", index=False)
    pd.set_option("display.width", 220)
    show = ["model", "family", "cv_top1", "cv_top1_std", "cv_macro_f1", "cv_log_loss", "test_top1",
            "test_macro_f1", "test_log_loss", "test_route_continuity", "fit_s_full_train",
            "infer_ms_per_1k_points"]
    print(T[show].round(4).to_string(index=False))

    rb = pd.read_csv(LOG / "rule_baselines.csv").set_index("rule")
    rule_te = rb.loc["nearest + heading (dist - 10*cos)", "student_test"]
    rule_tr = rb.loc["nearest + heading (dist - 10*cos)", "train"]
    order = T.model.tolist()
    F = pd.concat(folds_all)
    col = plt.cm.tab10(np.arange(len(order)) % 10)

    # 1. CV macro F1 and CV top-1 with fold spread
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, met, lab in [(axes[0], "macro_f1", "CV Macro F1 (row level)"),
                         (axes[1], "top1", "CV Top-1 accuracy (observation level)")]:
        means = [F[F.model == m][met].mean() for m in order]
        stds = [F[F.model == m][met].std() for m in order]
        ax.bar(order, means, yerr=stds, capsize=4, color=col)
        for i, m in enumerate(order):
            ax.scatter([i] * 5, F[F.model == m][met], s=8, c="k", zorder=3)
        lo = min(F[met].min(), rule_tr if met == "top1" else 1) - 0.02
        ax.set_ylim(max(lo, 0), 1.0)
        if met == "top1":
            ax.axhline(rule_tr, ls="--", c="r", lw=1, label=f"rule baseline (train) {rule_tr:.3f}")
            ax.legend(loc="lower right", fontsize=8)
        ax.set_title(f"{lab}: mean ± std over 5 trip-grouped folds", fontsize=10)
        ax.tick_params(axis="x", rotation=30)
    fig.tight_layout(); fig.savefig(FIG / "model_cv_comparison.png", dpi=150); plt.close(fig)

    # 2. student_test top-1 and macro F1
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(order))
    ax.bar(x - 0.2, T.test_top1, 0.4, label="Top-1 accuracy", color="#4c72b0")
    ax.bar(x + 0.2, T.test_macro_f1, 0.4, label="Macro F1", color="#dd8452")
    ax.axhline(rule_te, ls="--", c="r", lw=1, label=f"rule baseline Top-1 {rule_te:.3f}")
    for i, (a, b) in enumerate(zip(T.test_top1, T.test_macro_f1)):
        ax.text(i - 0.2, a + 0.003, f"{a:.3f}", ha="center", fontsize=7)
        ax.text(i + 0.2, b + 0.003, f"{b:.3f}", ha="center", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(order, rotation=30)
    ax.set_ylim(max(min(T.test_macro_f1.min(), T.test_top1.min(), rule_te) - 0.05, 0), 1.0)
    ax.set_title("Student-test performance (10 trips)"); ax.legend(loc="upper right", fontsize=8, ncol=3)
    fig.tight_layout(); fig.savefig(FIG / "model_test_comparison.png", dpi=150); plt.close(fig)

    # 3. log loss
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x - 0.2, T.cv_log_loss, 0.4, label="CV (mean)", color="#8172b2")
    ax.bar(x + 0.2, T.test_log_loss, 0.4, label="student_test", color="#64b5cd")
    ax.set_xticks(x); ax.set_xticklabels(order, rotation=30); ax.set_yscale("log")
    ax.set_ylabel("log loss (lower is better, log scale)"); ax.set_title("Probability quality: log loss")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "model_log_loss.png", dpi=150); plt.close(fig)

    # 4. speed vs accuracy
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for i, r in enumerate(T.itertuples()):
        ax.scatter(r.infer_ms_per_1k_points, r.test_top1, s=60, color=col[i])
        ax.annotate(r.model, (r.infer_ms_per_1k_points, r.test_top1), textcoords="offset points",
                    xytext=(5, 4), fontsize=8)
    ax.set_xscale("log"); ax.set_xlabel("inference time per 1,000 GPS points (ms, log scale)")
    ax.set_ylabel("student-test Top-1"); ax.set_title("Accuracy vs inference cost")
    fig.tight_layout(); fig.savefig(FIG / "model_speed_vs_accuracy.png", dpi=150); plt.close(fig)

    # 5. confusion matrices (row level, threshold 0.5)
    n = len(order)
    cols_ = min(4, n)
    rows_ = int(np.ceil(n / cols_))
    fig, axes = plt.subplots(rows_, cols_, figsize=(3.3 * cols_, 3.0 * rows_), squeeze=False)
    for ax, mname in zip(axes.ravel(), order):
        cm = pd.read_csv(MODELS_DIR / mname / "confusion_test.csv", index_col=0).to_numpy()
        cmn = cm / cm.sum(1, keepdims=True)
        ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, f"{v:,}\n({cmn[i, j]:.1%})", ha="center", va="center", fontsize=7,
                    color="white" if cmn[i, j] > 0.6 else "black")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["pred other", "pred correct"], fontsize=7)
        ax.set_yticks([0, 1]); ax.set_yticklabels(["other", "correct"], fontsize=7)
        ax.set_title(mname, fontsize=9)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle("Student-test confusion matrices (candidate rows, threshold 0.5)", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "model_confusion_matrices.png", dpi=150); plt.close(fig)

    # 6. MLP training curves (only model with epoch histories)
    for mname, h in hist.items():
        fig, ax = plt.subplots(figsize=(6.5, 4))
        ax.plot(h.epoch, h.train_loss, marker="o", ms=3, label="train loss")
        ax.plot(h.epoch, h.val_loss, marker="o", ms=3, label="validation loss (held-out trips)")
        best = h.val_loss.idxmin()
        ax.axvline(h.epoch[best], ls=":", c="k", lw=0.8, label=f"best epoch {h.epoch[best]}")
        ax.set_xlabel("epoch"); ax.set_ylabel("binary cross-entropy"); ax.legend(fontsize=8)
        ax.set_title(f"{mname}: training history (final fit on all train trips)")
        fig.tight_layout(); fig.savefig(FIG / f"model_{mname}_loss_curve.png", dpi=150); plt.close(fig)
    print(f"\nwrote {LOG / 'model_comparison.csv'} and results/figures/model_*.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", help="subset of model names (default: all)")
    ap.add_argument("--report", action="store_true", help="only rebuild the comparison from saved results")
    ap.add_argument("--sample", type=float, default=None, help="fraction of points to use (smoke test)")
    args = ap.parse_args()
    cfg = load_config()
    seed = cfg["seed"]
    for p in (MODELS_DIR, LOG, FIG):
        p.mkdir(parents=True, exist_ok=True)
    if args.report:
        report()
        return

    tr = load_split(cfg, "train", args.sample, seed)
    te = load_split(cfg, "student_test", args.sample, seed)
    trips = np.array(sorted(tr.trip.unique()))
    gkf = GroupKFold(n_splits=cfg["cv"]["n_splits"])
    tg = pd.DataFrame({"trip": trips})
    tg["fold"] = -1
    for k, (_, vi) in enumerate(gkf.split(trips, groups=trips)):
        tg.loc[vi, "fold"] = k
    if not args.sample:
        tg.to_csv(LOG / "cv_folds.csv", index=False)
    arcs_te = {t: load_trip(cfg["paths"]["raw_student_test"] / t, load_route=False).arcs for t in te.trip.unique()}
    print(f"train rows {len(tr):,} ({tr.trip.nunique()} trips), test rows {len(te):,}; "
          f"fold sizes (trips): {tg.fold.value_counts().sort_index().tolist()}")

    reg = registry(seed)
    names = args.models or list(reg)
    for name in names:
        spec = reg[name]
        out_dir = MODELS_DIR / name if not args.sample else Path(tempfile.mkdtemp(prefix=f"pathsnap_smoke_{name}_"))
        out_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        print(f"\n== {name} ({spec.family})", flush=True)
        m = run_model(spec, tr, te, tg, arcs_te, out_dir)
        print(f"   CV top1 {m['cv']['top1']['mean']:.4f} ± {m['cv']['top1']['std']:.4f} | "
              f"test top1 {m['test']['top1']:.4f}, macroF1 {m['test']['macro_f1']:.4f}, "
              f"logloss {m['test']['log_loss']:.4f}, continuity {m['test'].get('route_continuity', float('nan')):.4f} "
              f"| {time.time() - t0:.0f}s total", flush=True)
    if not args.sample:
        report()


if __name__ == "__main__":
    main()
