"""Server-side scoring of a prediction against the reference route.

Kept separate from pipeline.py on purpose: the reference route is read here, AFTER prediction, and
never flows into the prediction. The API calls predict_trip() on a route-free trip, then
score_prediction() with the route, and returns only the metrics (never the reference arcs).
"""
from __future__ import annotations

import numpy as np

from .io import Trip
from .labels import align_route, canonical_arc_ids
from .metrics import route_continuity


def score_prediction(trip_with_route: Trip, point_arc: np.ndarray, label_cfg: dict) -> dict:
    """Top-1 accuracy (labelled points), route continuity and route-level precision/recall/F1."""
    trip = trip_with_route
    labels = align_route(trip, max_dist=label_cfg.get("max_dist_m", 50.0), lam=label_cfg.get("progress_weight", 0.1))
    canon = canonical_arc_ids(trip.arcs)
    lab = labels.label_arc
    m = lab >= 0
    has_pred = point_arc >= 0
    correct = m & has_pred & (canon[np.where(has_pred, point_arc, 0)] == canon[np.where(m, lab, 0)])
    true_set = set(canon[trip.route])
    pred_set = set(canon[point_arc[has_pred]])
    tp = len(true_set & pred_set)
    prec = tp / max(len(pred_set), 1)
    rec = tp / max(len(true_set), 1)
    return dict(
        labelled_points=int(m.sum()), correct_points=int(correct.sum()),
        top1_accuracy=float(correct.sum() / max(m.sum(), 1)),
        route_continuity=float(route_continuity(trip.arcs, point_arc[has_pred])),
        route_precision=float(prec), route_recall=float(rec),
        route_f1=float(2 * prec * rec / (prec + rec)) if prec + rec else 0.0,
        unlabelled_points=int((~m).sum()),
    )
