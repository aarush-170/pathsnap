import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pathsnap.evaluation import score_prediction  # noqa: E402
from pathsnap.labels import align_route  # noqa: E402
from test_phase2 import _synthetic_out_and_back  # noqa: E402

CFG = dict(max_dist_m=20, progress_weight=0.1)


def test_perfect_prediction_scores_one():
    trip, _ = _synthetic_out_and_back()
    truth = align_route(trip, max_dist=20, lam=0.1).label_arc
    s = score_prediction(trip, truth, CFG)
    assert s["top1_accuracy"] == 1.0 and s["route_continuity"] == 1.0 and s["route_f1"] == 1.0


def test_wrong_direction_counts_as_wrong_and_missing_as_wrong():
    trip, half = _synthetic_out_and_back()
    truth = align_route(trip, max_dist=20, lam=0.1).label_arc
    pred = truth.copy()
    twin = {0: 3, 1: 2, 2: 1, 3: 0}                  # same street, opposite direction
    pred[:5] = [twin[a] for a in pred[:5]]
    pred[5] = -1                                     # no prediction for one point
    s = score_prediction(trip, pred, CFG)
    assert s["correct_points"] == len(truth) - 6
    assert np.isclose(s["top1_accuracy"], (len(truth) - 6) / len(truth))
