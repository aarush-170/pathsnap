"""End-to-end inference: raw trip files -> per-point road predictions -> drivable route.

This is the ONLY prediction path, used by the experiments (export check) and by the API. It never
reads the reference route: `predict_trip` works on a Trip loaded with load_route=False.

Artifacts (models/):
  lightgbm_tuned.txt      LightGBM booster (text format, portable across machines)
  pipeline_config.json    feature list, candidate settings, HMM settings, training summary
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .candidates import generate_candidates, trip_geometry
from .features import FEATURES, trip_features
from .hmm import HMMParams, RoadGraph, decode_trip, reconstruct_route
from .io import Trip


@dataclass
class Artifacts:
    booster: object
    config: dict

    @property
    def hmm(self) -> HMMParams:
        h = self.config["hmm"]
        return HMMParams(beta=h["beta"], alpha=h["alpha"], gap_break_s=h["gap_break_seconds"],
                         route_factor=h["route_factor"], route_slack_m=h["route_slack_m"])


def load_artifacts(model_dir: str | Path) -> Artifacts:
    import lightgbm as lgb
    model_dir = Path(model_dir)
    cfg = json.loads((model_dir / "pipeline_config.json").read_text())
    if cfg["features"] != FEATURES:
        raise ValueError("feature list in pipeline_config.json does not match pathsnap.features.FEATURES")
    booster = lgb.Booster(model_file=str(model_dir / cfg["model_file"]))
    return Artifacts(booster, cfg)


def predict_trip(trip: Trip, art: Artifacts, use_hmm: bool = True) -> dict:
    """Run the full pipeline on one trip. Returns predictions, the reconstructed route and timings."""
    t = {}
    t0 = time.perf_counter()
    geom = trip_geometry(trip)
    c = art.config["candidates"]
    cands = generate_candidates(trip, radius=c["radius_m"], max_rank=c["max_k"], geom=geom)
    t["candidates_s"] = time.perf_counter() - t0

    t1 = time.perf_counter()
    F = trip_features(trip, geom, cands)
    t["features_s"] = time.perf_counter() - t1

    t2 = time.perf_counter()
    prob = art.booster.predict(F[FEATURES].to_numpy(np.float32)) if len(F) else np.array([])
    cands = cands.assign(p=prob)
    t["classifier_s"] = time.perf_counter() - t2

    t3 = time.perf_counter()
    graph = RoadGraph(trip.arcs, geom.N)
    clf = (cands.sort_values(["point_idx", "p", "rank"], ascending=[True, False, True])
           .drop_duplicates("point_idx").set_index("point_idx").arc)
    if use_hmm and len(cands):
        pred, diag = decode_trip(graph, geom.P, trip.track.t.to_numpy(), cands, prob, art.hmm)
    else:
        pred, diag = clf, dict(segments=0, breaks=0, gap_breaks=0)
    route = reconstruct_route(graph, pred.sort_index().to_numpy())
    t["hmm_s"] = time.perf_counter() - t3
    t["total_s"] = time.perf_counter() - t0

    n = len(trip.track)
    point_arc = np.full(n, -1, dtype=np.int64)
    point_arc[pred.index.to_numpy()] = pred.to_numpy()
    clf_arc = np.full(n, -1, dtype=np.int64)
    clf_arc[clf.index.to_numpy()] = clf.to_numpy()
    return dict(trip_id=trip.trip_id, n_points=n, n_candidates=int(len(cands)),
                points_without_candidates=int((point_arc < 0).sum()),
                point_arc=point_arc, classifier_point_arc=clf_arc, route=route,
                route_gaps=int(sum(1 for a in route if a < 0)), hmm=diag, timings=t)
