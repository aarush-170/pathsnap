"""Phase 5: Bayesian hyperparameter optimisation of the selected model (LightGBM) with Optuna.

  python -m scripts.tune_optuna                    # search (default 40 trials / 100 min) + final comparison
  python -m scripts.tune_optuna --trials 60 --timeout 7200
  python -m scripts.tune_optuna --final-only       # skip search, use the best params already saved
  python -m scripts.tune_optuna --sample 0.03 --trials 4   # smoke test

Design
  sampler    TPE (Tree-structured Parzen Estimator), seed 42 -- a Bayesian method: it models which
             regions of the search space give good scores and samples where improvement is likely
  objective  mean validation LOG LOSS over the same 5 trip-grouped folds as Phase 4 (minimise).
             Log loss is a proper scoring rule: it rewards accurate probabilities, which the HMM uses as
             emission scores, and it is smoother than Top-1 (fewer ties/noise for the optimiser).
             Top-1 and Macro F1 are recorded for every trial too.
  pruning    MedianPruner: after each fold, a trial whose running mean log loss is worse than the median
             of earlier trials at the same fold is stopped early (saves time on hopeless settings)
  trial 0    the untuned Phase 4 configuration, so the search starts from (and is compared with) it
  data       train split only (80 trips). student_test is used once, after tuning, for the comparison.
Outputs
  results/logs/optuna_trials.csv, configs/lightgbm_tuned.json, results/logs/tuning_comparison.csv,
  results/models/lightgbm_tuned/ (same files as Phase 4), results/figures/optuna_*.png
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd

from pathsnap.config import load_config
from pathsnap.features import FEATURES
from pathsnap.io import load_trip
from pathsnap.models import LGBMModel, ModelSpec, registry
from scripts.train_models import MODELS_DIR, evaluate, load_split, run_model

LOG, FIG = ROOT / "results" / "logs", ROOT / "results" / "figures"
BEST_PATH = ROOT / "configs" / "lightgbm_tuned.json"

SEARCH_SPACE = {  # documented in the report: name -> (type, low, high, scale)
    "learning_rate": ("float", 0.01, 0.2, "log"),
    "n_estimators": ("int", 100, 800, "step 50"),
    "num_leaves": ("int", 15, 255, "log"),
    "min_child_samples": ("int", 10, 500, "log"),
    "subsample": ("float", 0.5, 1.0, "linear"),
    "colsample_bytree": ("float", 0.4, 1.0, "linear"),
    "reg_lambda": ("float", 1e-3, 10.0, "log"),
    "reg_alpha": ("float", 1e-3, 10.0, "log"),
}
FIXED = dict(subsample_freq=1)


def suggest(trial):
    p = {}
    for name, (kind, lo, hi, scale) in SEARCH_SPACE.items():
        if kind == "int":
            p[name] = trial.suggest_int(name, lo, hi, step=50 if scale == "step 50" else 1, log=(scale == "log"))
        else:
            p[name] = trial.suggest_float(name, lo, hi, log=(scale == "log"))
    return {**p, **FIXED}


def make_objective(tr, folds, seed):
    X, y, g = tr[FEATURES].to_numpy(np.float32), tr.is_true.to_numpy(), tr.trip.to_numpy()
    masks = [np.isin(g, folds.trip[folds.fold == k].to_numpy()) for k in sorted(folds.fold.unique())]

    def objective(trial):
        params = suggest(trial)
        lls, top1s, f1s, t0 = [], [], [], time.time()
        for k, vm in enumerate(masks):
            m = LGBMModel(params, seed=seed).fit(X[~vm], y[~vm])
            r = evaluate(tr[vm], m.predict_proba(X[vm]))
            lls.append(r["log_loss"]); top1s.append(r["top1"]); f1s.append(r["macro_f1"])
            trial.report(float(np.mean(lls)), step=k)
            if trial.should_prune():
                trial.set_user_attr("folds_done", k + 1)
                raise optuna.TrialPruned()
        trial.set_user_attr("cv_top1", float(np.mean(top1s)))
        trial.set_user_attr("cv_top1_std", float(np.std(top1s, ddof=1)))
        trial.set_user_attr("cv_macro_f1", float(np.mean(f1s)))
        trial.set_user_attr("cv_log_loss_std", float(np.std(lls, ddof=1)))
        trial.set_user_attr("folds_done", len(masks))
        trial.set_user_attr("seconds", time.time() - t0)
        return float(np.mean(lls))

    return objective


def search(tr, folds, cfg, n_trials, timeout, sample):
    seed = cfg["seed"]
    study = optuna.create_study(direction="minimize", study_name="lightgbm_pathsnap",
                                sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=8),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1))
    untuned = {k: v for k, v in registry(seed)["lightgbm"].config.items() if k in SEARCH_SPACE}
    untuned.setdefault("reg_alpha", 1e-3)   # Phase 4 used LightGBM's default 0; 1e-3 is the nearest in range
    study.enqueue_trial(untuned)          # trial 0 = Phase 4 configuration
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def log_cb(study, t):
        state = t.state.name.lower()
        val = f"{t.value:.5f}" if t.value is not None else "-"
        best = study.best_value if any(s.state.name == "COMPLETE" for s in study.trials) else float("nan")
        print(f"  trial {t.number:3d} {state:8s} logloss={val} top1={t.user_attrs.get('cv_top1', float('nan')):.4f} "
              f"folds={t.user_attrs.get('folds_done')} | best={best:.5f}", flush=True)

    study.optimize(make_objective(tr, folds, seed), n_trials=n_trials, timeout=timeout, callbacks=[log_cb])
    df = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs", "state", "duration"))
    df["duration_s"] = df.pop("duration").dt.total_seconds()
    if not sample:
        df.to_csv(LOG / "optuna_trials.csv", index=False)
        best = {**study.best_params, **FIXED}
        BEST_PATH.write_text(json.dumps(dict(params=best, cv_log_loss=study.best_value,
                                             best_trial=study.best_trial.number,
                                             cv_top1=study.best_trial.user_attrs.get("cv_top1"),
                                             n_trials=len(study.trials),
                                             n_complete=int((df.state == "COMPLETE").sum()),
                                             n_pruned=int((df.state == "PRUNED").sum()),
                                             search_space={k: list(v) for k, v in SEARCH_SPACE.items()},
                                             objective="mean 5-fold trip-grouped validation log loss",
                                             sampler="TPE seed 42", pruner="MedianPruner"), indent=2))
        plots(study, df)
    print(f"\nbest trial {study.best_trial.number}: logloss {study.best_value:.5f}, params {study.best_params}")
    return study


def plots(study, df):
    comp = df[df.state == "COMPLETE"].sort_values("number")
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.scatter(comp.number, comp.value, s=18, label="completed trial")
    pr = df[df.state == "PRUNED"]
    if len(pr):
        ax.scatter(pr.number, [comp.value.max()] * len(pr), marker="x", c="grey", s=18, label="pruned (stopped early)")
    ax.plot(comp.number, comp.value.cummin(), c="r", lw=1.5, label="best so far")
    ax.axhline(comp.value.iloc[0] if comp.number.iloc[0] == 0 else np.nan, ls="--", c="k", lw=0.8,
               label="trial 0 = untuned config")
    ax.set_xlabel("trial"); ax.set_ylabel("mean CV log loss (lower is better)")
    ax.set_title("Optuna TPE optimisation history (LightGBM)"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "optuna_history.png", dpi=150); plt.close(fig)

    try:
        imp = optuna.importance.get_param_importances(study)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.barh(list(imp)[::-1], list(imp.values())[::-1], color="#4c72b0")
        ax.set_xlabel("relative importance (fANOVA)"); ax.set_title("Which hyperparameters mattered")
        fig.tight_layout(); fig.savefig(FIG / "optuna_param_importance.png", dpi=150); plt.close(fig)
        pd.Series(imp, name="importance").to_csv(LOG / "optuna_param_importance.csv")
    except Exception as e:  # needs enough completed trials
        print("param importance skipped:", e)

    names = list(SEARCH_SPACE)
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.5))
    for ax, n in zip(axes.ravel(), names):
        ax.scatter(comp[f"params_{n}"], comp.value, s=14, c=comp.number, cmap="viridis")
        if SEARCH_SPACE[n][3] == "log":
            ax.set_xscale("log")
        ax.set_title(n, fontsize=9); ax.tick_params(labelsize=7)
    axes[0, 0].set_ylabel("CV log loss"); axes[1, 0].set_ylabel("CV log loss")
    fig.suptitle("Each hyperparameter vs CV log loss (colour = trial number)")
    fig.tight_layout(); fig.savefig(FIG / "optuna_slices.png", dpi=150); plt.close(fig)


def final_comparison(cfg, tr, te, folds):
    best = json.loads(BEST_PATH.read_text())["params"]
    seed = cfg["seed"]
    spec = ModelSpec("lightgbm_tuned", "Boosting", best, lambda: LGBMModel(best, seed=seed),
                     "Optuna TPE best trial (configs/lightgbm_tuned.json)")
    out = MODELS_DIR / "lightgbm_tuned"
    out.mkdir(parents=True, exist_ok=True)
    arcs_te = {t: load_trip(cfg["paths"]["raw_student_test"] / t, load_route=False).arcs for t in te.trip.unique()}
    print("\n== refit tuned LightGBM with the Phase 4 protocol")
    tuned = run_model(spec, tr, te, folds, arcs_te, out)
    base = json.loads((MODELS_DIR / "lightgbm" / "metrics.json").read_text())
    rows = []
    for label, m in [("untuned (Phase 4)", base), ("tuned (Optuna)", tuned)]:
        rows.append({"model": label, "cv_macro_f1": m["cv"]["macro_f1"]["mean"],
                     "cv_top1": m["cv"]["top1"]["mean"], "cv_top1_std": m["cv"]["top1"]["std"],
                     "cv_log_loss": m["cv"]["log_loss"]["mean"], "cv_train_top1": m["cv"]["train_top1"]["mean"],
                     "test_top1": m["test"]["top1"], "test_macro_f1": m["test"]["macro_f1"],
                     "test_log_loss": m["test"]["log_loss"], "test_route_continuity": m["test"]["route_continuity"],
                     "fit_s_full_train": m["test_fit_s"], "infer_ms_per_1k_points": m["test_infer_ms_per_1k_points"]})
    T = pd.DataFrame(rows)
    T.to_csv(LOG / "tuning_comparison.csv", index=False)
    pd.set_option("display.width", 220)
    print(T.round(4).to_string(index=False))

    metrics = [("cv_macro_f1", "CV Macro F1"), ("test_top1", "Test Top-1"), ("test_macro_f1", "Test Macro F1"),
               ("test_log_loss", "Test log loss"), ("infer_ms_per_1k_points", "Inference ms / 1k points")]
    fig, axes = plt.subplots(1, len(metrics), figsize=(16, 3.6))
    for ax, (c, lab) in zip(axes, metrics):
        bars = ax.bar(["untuned", "tuned"], T[c], color=["#9e9e9e", "#2ca02c"])
        ax.bar_label(bars, fmt="%.4f" if T[c].max() < 10 else "%.1f", fontsize=8)
        lo, hi = T[c].min(), T[c].max()
        if c not in ("infer_ms_per_1k_points", "test_log_loss"):
            ax.set_ylim(lo - 0.01, min(1.0, hi + 0.01))
        ax.set_title(lab, fontsize=9)
    fig.suptitle("LightGBM before vs after Bayesian optimisation (same folds, same test trips)", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "optuna_tuned_vs_untuned.png", dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--timeout", type=int, default=6000, help="seconds")
    ap.add_argument("--sample", type=float, default=None)
    ap.add_argument("--final-only", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    tr = load_split(cfg, "train", args.sample, cfg["seed"])
    te = load_split(cfg, "student_test", args.sample, cfg["seed"])
    folds = pd.read_csv(LOG / "cv_folds.csv", dtype={"trip": str})
    print(f"train rows {len(tr):,}, folds from results/logs/cv_folds.csv (same as Phase 4)")
    if not args.final_only:
        search(tr, folds, cfg, args.trials, args.timeout, args.sample)
    if not args.sample:
        final_comparison(cfg, tr, te, folds)


if __name__ == "__main__":
    main()
