import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pathsnap.hmm import HMMParams, RoadGraph, decode_trip  # noqa: E402


def _two_parallel_roads():
    """Road A: nodes 0-1-2 along y=0. Road B: nodes 3-4-5 along y=20, NOT connected to A.
    Arcs: 0:(0,1) 1:(1,2) on A, 2:(3,4) 3:(4,5) on B."""
    N = np.array([[0, 0], [100, 0], [200, 0], [0, 20], [100, 20], [200, 20]], float)
    arcs = np.array([[0, 1], [1, 2], [3, 4], [4, 5]])
    return RoadGraph(arcs, N), N


def _cands(xs, arcs_for_x):
    rows = []
    for i, x in enumerate(xs):
        for a in arcs_for_x(x):
            seg_start = {0: 0, 1: 100, 2: 0, 3: 100}[a]
            rows.append((i, a, (x - seg_start) / 100.0))
    return pd.DataFrame(rows, columns=["point_idx", "arc", "t"])


def test_hmm_removes_isolated_jump_to_unconnected_road():
    g, _ = _two_parallel_roads()
    xs = np.arange(5, 200, 10.0)                       # car drives along road A
    P = np.c_[xs, np.full_like(xs, 8.0)]               # GPS sits between the roads, closer to A
    ts = np.arange(len(xs), dtype=float)
    c = _cands(xs, lambda x: [0, 2] if x < 100 else [1, 3])
    prob = np.where(c.arc.isin([0, 1]), 0.8, 0.2)
    flip = c.point_idx == 10                           # one noisy point: classifier prefers road B
    prob = np.where(flip & c.arc.isin([0, 1]), 0.3, np.where(flip, 0.7, prob))
    clf = c.assign(q=prob).sort_values("q").drop_duplicates("point_idx", keep="last").set_index("point_idx").arc
    assert clf.loc[10] == 3                            # classifier-only picks the wrong road
    pred, info = decode_trip(g, P, ts, c, prob, HMMParams(beta=10.0))
    assert pred.loc[10] == 1                           # HMM keeps the car on connected road A
    assert set(pred.values) <= {0, 1}
    assert info["segments"] == 1


def test_gap_breaks_chain_and_unreachable_starts_new_segment():
    g, _ = _two_parallel_roads()
    xs = np.array([10.0, 20.0, 30.0, 40.0])
    P = np.c_[xs, np.zeros(4)]
    ts = np.array([0.0, 1.0, 200.0, 201.0])            # 199 s gap between points 1 and 2
    c = _cands(xs, lambda x: [0])
    pred, info = decode_trip(g, P, ts, c, np.ones(len(c)), HMMParams(gap_break_s=60))
    assert info["gap_breaks"] == 1 and info["segments"] == 2
    assert (pred == 0).all()
    # only road B available after point 1: unreachable from A -> a break, not a crash
    c2 = pd.concat([_cands(xs[:2], lambda x: [0]), _cands(xs[2:], lambda x: [2]).assign(
        point_idx=lambda d: d.point_idx + 2)])
    pred2, info2 = decode_trip(g, P, np.arange(4.0), c2, np.ones(len(c2)), HMMParams())
    assert info2["breaks"] == 1
    assert pred2.tolist() == [0, 0, 2, 2]


def test_path_arcs_fills_between_connected_arcs():
    g, _ = _two_parallel_roads()
    assert g.path_arcs(0, 1) == []
    assert g.path_arcs(0, 2) is None
