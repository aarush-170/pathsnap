"""Phase 8: train the final model on all 80 train trips and export everything the API needs.

  python -m scripts.export_artifacts                 # train, save, verify on student_test
  python -m scripts.export_artifacts --verify-only   # re-check saved artifacts
  python -m scripts.export_artifacts --check-eval    # also run the pipeline on evaluator trips:
                                                     # inference + timing ONLY, no scoring
Outputs: models/lightgbm_tuned.txt, models/pipeline_config.json, results/logs/export_check.csv
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from pathsnap.config import load_config
from pathsnap.evaluation import score_prediction
from pathsnap.features import FEATURES
from pathsnap.io import list_trips, load_trip
from pathsnap.pipeline import load_artifacts, predict_trip

MODELS = ROOT / "models"
LOG = ROOT / "results" / "logs"


def train_and_save(cfg):
    import lightgbm as lgb
    tuned = json.loads((ROOT / cfg["model"]["params_file"]).read_text())
    params = tuned["params"]
    tr = pd.read_parquet(cfg["paths"]["processed"] / "features_train.parquet")
    tr = tr[tr.labelled]
    X, y = tr[FEATURES].to_numpy(np.float32), tr.is_true.to_numpy()
    t0 = time.perf_counter()
    clf = lgb.LGBMClassifier(**params, random_state=cfg["seed"], verbose=-1).fit(X, y)
    fit_s = time.perf_counter() - t0
    MODELS.mkdir(exist_ok=True)
    clf.booster_.save_model(str(MODELS / "lightgbm_tuned.txt"))
    st = json.loads((ROOT / "results" / "models" / "lightgbm_tuned" / "metrics.json").read_text())["test"]
    hmm = pd.read_csv(LOG / "hmm_comparison_lightgbm_tuned.csv").iloc[1]
    pipe_cfg = dict(
        model_file="lightgbm_tuned.txt", model="LightGBM (Optuna-tuned, Phase 5 trial %d)" % tuned["best_trial"],
        lightgbm_params=params, seed=cfg["seed"], features=FEATURES,
        candidates=dict(radius_m=cfg["candidates"]["radius_m"], max_k=cfg["candidates"]["max_k"]),
        hmm=dict(beta=cfg["hmm"]["beta"], alpha=cfg["hmm"]["alpha"], gap_break_seconds=cfg["hmm"]["gap_break_seconds"],
                 route_factor=cfg["hmm"]["route_factor"], route_slack_m=cfg["hmm"]["route_slack_m"]),
        labels=dict(max_dist_m=cfg["labels"]["max_dist_m"], progress_weight=cfg["labels"]["progress_weight"]),
        training=dict(trips=int(tr.trip.nunique()), rows=int(len(tr)), positives=int(y.sum()), fit_seconds=round(fit_s, 1),
                      trained_at=dt.datetime.now().isoformat(timespec="seconds"),
                      python=platform.python_version(), lightgbm=lgb.__version__),
        student_test_reference=dict(classifier_top1=st["top1"], hmm_top1=float(hmm.top1),
                                    hmm_route_continuity=float(hmm.continuity)),
    )
    (MODELS / "pipeline_config.json").write_text(json.dumps(pipe_cfg, indent=2))
    size = (MODELS / "lightgbm_tuned.txt").stat().st_size / 1e6
    print(f"trained on {len(tr):,} rows from {tr.trip.nunique()} trips in {fit_s:.1f}s; model file {size:.1f} MB")
    return clf


def verify(cfg):
    art = load_artifacts(MODELS)
    # 1) classifier parity: same probabilities as the Phase 5 refit on student_test candidates
    te = pd.read_parquet(cfg["paths"]["processed"] / "features_student_test.parquet")
    te = te[te.labelled]
    p_new = art.booster.predict(te[FEATURES].to_numpy(np.float32))
    ref = pd.read_parquet(ROOT / "results" / "models" / "lightgbm_tuned" / "pred_student_test.parquet")
    diff = np.abs(p_new - ref.p.to_numpy()).max()
    print(f"classifier parity vs Phase 5 predictions: max |dp| = {diff:.2e}")

    # 2) end-to-end from RAW files (route-free trip) + server-side scoring
    rows = []
    for d in list_trips(cfg["paths"]["raw_student_test"]):
        trip = load_trip(d, split="student_test", load_route=False)
        out = predict_trip(trip, art)
        full = load_trip(d, split="student_test", load_route=True)
        s_hmm = score_prediction(full, out["point_arc"], art.config["labels"])
        s_clf = score_prediction(full, out["classifier_point_arc"], art.config["labels"])
        rows.append(dict(trip=trip.trip_id, points=out["n_points"], candidates=out["n_candidates"],
                         correct_hmm=s_hmm["correct_points"], correct_clf=s_clf["correct_points"],
                         labelled=s_hmm["labelled_points"], top1_clf=s_clf["top1_accuracy"],
                         top1_hmm=s_hmm["top1_accuracy"], continuity=s_hmm["route_continuity"],
                         route_f1=s_hmm["route_f1"], route_gaps=out["route_gaps"], **out["timings"]))
        print(f"  {trip.trip_id}: top1 {s_clf['top1_accuracy']:.4f} -> {s_hmm['top1_accuracy']:.4f} "
              f"in {out['timings']['total_s']:.1f}s", flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(LOG / "export_check.csv", index=False)
    tot_h = R.correct_hmm.sum() / R.labelled.sum()
    tot_c = R.correct_clf.sum() / R.labelled.sum()
    ref_cfg = art.config["student_test_reference"]
    print(f"end-to-end from raw files: classifier {tot_c:.4f} (Phase 5: {ref_cfg['classifier_top1']:.4f}), "
          f"HMM {tot_h:.4f} (Phase 6: {ref_cfg['hmm_top1']:.4f}); max trip time {R.total_s.max():.1f}s")
    return R


def check_eval(cfg):
    """Robustness check for the demo: the pipeline must run on every evaluator trip.
    Prediction only -- the evaluator routes are NOT loaded and nothing is scored."""
    art = load_artifacts(MODELS)
    rows = []
    for d in list_trips(cfg["paths"]["raw_eval"]):
        trip = load_trip(d, split="eval", load_route=False)
        out = predict_trip(trip, art)
        rows.append(dict(trip=trip.trip_id, points=out["n_points"], candidates=out["n_candidates"],
                         points_without_candidates=out["points_without_candidates"],
                         route_arcs=len(out["route"]), route_gaps=out["route_gaps"],
                         hmm_segments=out["hmm"]["segments"], total_s=round(out["timings"]["total_s"], 2)))
        print(f"  eval {trip.trip_id}: ok, {out['n_points']} points, {out['timings']['total_s']:.1f}s", flush=True)
    pd.DataFrame(rows).to_csv(LOG / "export_eval_runcheck.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--check-eval", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    if not args.verify_only:
        train_and_save(cfg)
    verify(cfg)
    if args.check_eval:
        check_eval(cfg)


if __name__ == "__main__":
    main()
