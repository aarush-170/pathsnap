import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pathsnap.geo import LocalProjector, haversine, point_segment_distance  # noqa: E402
from pathsnap.io import Trip  # noqa: E402
from pathsnap.labels import align_route, canonical_arc_ids  # noqa: E402
from pathsnap.spatial import SegmentIndex  # noqa: E402


def test_point_segment_distance_basic():
    d, t = point_segment_distance([[5, 3], [-4, 0], [12, 0]], [[0, 0]], [[10, 0]])
    np.testing.assert_allclose(d[:, 0], [3, 4, 2])
    np.testing.assert_allclose(t[:, 0], [0.5, 0, 1])


def test_projection_matches_haversine():
    proj = LocalProjector(8.6, 49.87)
    lon, lat = np.array([8.6, 8.7]), np.array([49.87, 49.92])
    xy = proj.to_xy(lon, lat)
    planar = np.linalg.norm(xy[1] - xy[0])
    assert abs(planar - haversine(lon[0], lat[0], lon[1], lat[1])) / planar < 1e-3


def test_segment_index_matches_brute_force():
    rng = np.random.default_rng(0)
    A = rng.uniform(0, 1000, (300, 2))
    B = A + rng.normal(0, 80, (300, 2))
    P = rng.uniform(0, 1000, (200, 2))
    pi, si, d, _ = SegmentIndex(A, B, spacing=10).query(P, 40)
    D, _ = point_segment_distance(P, A, B)
    bp, bs = np.where(D <= 40)
    assert set(zip(pi, si)) == set(zip(bp, bs))


def test_canonical_ids_merge_duplicates_only():
    arcs = np.array([[0, 1], [1, 0], [0, 1], [2, 3]])
    np.testing.assert_array_equal(canonical_arc_ids(arcs), [0, 1, 0, 3])


def _synthetic_out_and_back():
    """A dead-end street driven out and back: nodes 0-1-2, arcs both directions."""
    lon0, lat0 = 8.6, 49.87
    proj = LocalProjector(lon0, lat0)
    nodes_xy = np.array([[0, 0], [100, 0], [200, 0]], float)
    arcs = np.array([[0, 1], [1, 2], [2, 1], [1, 0]])           # 0,1 = outbound; 2,3 = return
    route = np.array([0, 1, 2, 3])
    x_out = np.arange(5, 200, 10.0)
    gps_xy = np.r_[np.c_[x_out, np.full_like(x_out, 3.0)], np.c_[x_out[::-1], np.full_like(x_out, -3.0)]]
    ll = proj.to_lonlat(gps_xy)
    nll = proj.to_lonlat(nodes_xy)
    track = pd.DataFrame({"lon": ll[:, 0], "lat": ll[:, 1], "t": np.arange(len(ll), dtype=float)})
    return Trip("syn", "test", track, nll, arcs, route), len(x_out)


def test_alignment_recovers_direction_on_out_and_back():
    trip, half = _synthetic_out_and_back()
    lab = align_route(trip, max_dist=20, lam=0.1)
    assert (lab.label_arc >= 0).all()
    assert np.all(np.diff(lab.route_pos) >= 0)
    # outbound half must use arcs 0/1, return half arcs 2/3 (same geometry, opposite direction):
    # the |route progress - GPS displacement| term forbids sliding backwards along the outbound arc
    assert set(lab.label_arc[:half]) <= {0, 1}
    assert set(lab.label_arc[half:]) <= {2, 3}
