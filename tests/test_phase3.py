import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pathsnap.candidates import generate_candidates, trip_geometry  # noqa: E402
from pathsnap.features import FEATURES, trip_features  # noqa: E402
from pathsnap.metrics import route_continuity, top1_accuracy  # noqa: E402
from test_phase2 import _synthetic_out_and_back  # noqa: E402


def _features():
    trip, half = _synthetic_out_and_back()
    g = trip_geometry(trip)
    c = generate_candidates(trip, radius=20, geom=g)
    return trip, half, c, trip_features(trip, g, c)


def test_feature_table_shape_and_finite():
    _, _, c, F = _features()
    assert list(F.columns) == FEATURES
    assert len(F) == len(c)
    assert np.isfinite(F.to_numpy()).all()


def test_heading_separates_two_way_twins():
    _, half, c, F = _features()
    out = c.point_idx < half                      # outbound: moving east along arcs 0/1
    east = c.arc.isin([0, 1])
    # interior points (heading window fully inside one leg)
    interior = c.point_idx.between(3, half - 4) | c.point_idx.between(half + 3, 2 * half - 4)
    same_dir = (out & east) | (~out & ~east)
    assert (F.heading_cos[interior & same_dir] > 0.9).all()
    assert (F.heading_cos[interior & ~same_dir] < -0.9).all()
    # twins share the distance features
    assert F.groupby([c.point_idx, c.arc.isin([0, 3])]).dist.nunique().max() == 1


def test_features_do_not_need_route():
    trip, _, c, F = _features()
    trip.route = None
    g = trip_geometry(trip)
    F2 = trip_features(trip, g, c)
    pd.testing.assert_frame_equal(F, F2)


def test_top1_and_continuity():
    df = pd.DataFrame({"trip": ["a"] * 4, "point_idx": [0, 0, 1, 1], "rank": [0, 1, 0, 1],
                       "is_true": [True, False, False, True], "s": [0.9, 0.1, 0.8, 0.2]})
    assert top1_accuracy(df, "s") == 0.5
    assert top1_accuracy(df, "s", labelled_points=4) == 0.25
    arcs = np.array([[0, 1], [1, 2], [5, 6]])
    assert route_continuity(arcs, [0, 0, 1, 1]) == 1.0
    assert route_continuity(arcs, [0, 1, 2]) == 0.5
