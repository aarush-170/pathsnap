"""Per-GPS-point labels derived from the ordered reference route.

The .route file lists the arcs driven, in order, but does not say which GPS point belongs to
which arc. Snapping every point to its nearest route arc is wrong whenever the route passes the
same place twice or runs next to itself (e.g. both carriageways). Instead we solve an
order-preserving alignment with dynamic programming:

  choose for each point i a route position j_i (index into the route) such that
      j_1 <= j_2 <= ... <= j_n            (the vehicle never goes backwards along its route)
  minimising   sum_i dist(p_i, route[j_i])
             + lam * sum_i | (route progress between i-1 and i) - (GPS displacement between i-1 and i) |

The consistency term says "the distance travelled along the route should match the distance
travelled by the GPS". It stops the alignment from jumping ahead to a later revisit of the same
street, and from sliding backwards along an arc (which is how the two directions of a two-way
street are told apart at a U-turn). Points farther than max_dist from every route arc get no
label (-1): the route does not explain them (see results/logs/eda_notes.md).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geo import LocalProjector
from .io import Trip
from .spatial import SegmentIndex

RESTART_PENALTY = 200.0  # metres-equivalent cost of breaking monotonicity (should be rare)


@dataclass
class TripLabels:
    label_arc: np.ndarray      # (n,) arc id per point, -1 if unlabelled
    route_pos: np.ndarray      # (n,) index into trip.route, -1 if unlabelled
    label_dist: np.ndarray     # (n,) distance to the label arc (m), nan if unlabelled
    nearest_arc: np.ndarray    # (n,) naive nearest-route-arc label for comparison, -1 if none
    restarts: int              # times the DP considered breaking monotonicity
    dropped_short_runs: int = 0  # points unlabelled by _drop_short_runs


def canonical_arc_ids(arcs: np.ndarray) -> np.ndarray:
    """Map duplicate arcs (same from,to pair) to the lowest arc id with that pair."""
    key = arcs[:, 0].astype(np.int64) * (int(arcs.max()) + 1) + arcs[:, 1]
    _, first, inv = np.unique(key, return_index=True, return_inverse=True)
    return first[inv]


def align_route(trip: Trip, max_dist: float = 50.0, lam: float = 0.1) -> TripLabels:
    lon, lat = trip.track.lon.values, trip.track.lat.values
    proj = LocalProjector.for_points(lon, lat)
    P = proj.to_xy(lon, lat)
    nxy = proj.to_xy(trip.nodes[:, 0], trip.nodes[:, 1])
    route = trip.route
    A, B = nxy[trip.arcs[route, 0]], nxy[trip.arcs[route, 1]]
    seg_len = np.linalg.norm(B - A, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)[:-1]])  # route distance at the start of each arc

    pi, sj, d, t = SegmentIndex(A, B).query(P, max_dist)
    order = np.lexsort((sj, pi))
    pi, sj, d, t = pi[order], sj[order], d[order], t[order]
    bounds = np.searchsorted(pi, np.arange(len(P) + 1))

    n = len(P)
    nearest = np.full(n, -1)
    # DP over the sparse states of each point
    # states[i] = (route positions j, distances, backpointer into previous labelled point, prev point index)
    states: list[tuple | None] = [None] * n
    prev = None       # (j, pos, cost) of the last labelled point
    prev_i = -1
    restarts = 0
    for i in range(n):
        lo, hi = bounds[i], bounds[i + 1]
        if lo == hi:
            continue
        j, di, ti = sj[lo:hi], d[lo:hi], t[lo:hi]
        nearest[i] = route[j[np.argmin(di)]]
        pos = cum[j] + ti * seg_len[j]
        if prev is None:
            # route progress is measured from the start of the route, so starting deep into it
            # (e.g. a round trip whose start is also near the route's end) is penalised
            cost = di + lam * pos
            back = np.full(len(j), -1)
        else:
            pj, ppos, pcost = prev
            step = float(np.linalg.norm(P[i] - P[prev_i]))
            # (prev states) x (current states) transition costs; only forward-or-same route index allowed
            trans = pcost[:, None] + lam * np.abs((pos[None, :] - ppos[:, None]) - step)
            trans = np.where(pj[:, None] <= j[None, :], trans, np.inf)
            back = np.argmin(trans, axis=0)
            best = trans[back, np.arange(len(j))]
            ok = np.isfinite(best)
            cost = di + np.where(ok, best, 0.0)
            if (~ok).any():
                # every candidate route position lies *before* all previous states: allow a
                # penalised restart from the previous point's best state
                restarts += 1
                g = int(np.argmin(pcost))
                cost[~ok] = di[~ok] + pcost[g] + RESTART_PENALTY
                back[~ok] = g
        states[i] = (j, di, back, prev_i)
        prev = (j, pos, cost)
        prev_i = i

    label_arc = np.full(n, -1)
    route_pos = np.full(n, -1)
    label_dist = np.full(n, np.nan)
    if prev is not None:
        # best final state, then backtrack through the chain of labelled points
        q = int(np.argmin(prev[2]))
        i = prev_i
        while i >= 0 and q >= 0:
            j, di, back, ip = states[i]
            route_pos[i] = j[q]
            label_arc[i] = route[j[q]]
            label_dist[i] = di[q]
            q, i = int(back[q]), ip
    dropped = _drop_short_runs(label_arc, route_pos, label_dist)
    return TripLabels(label_arc, route_pos, label_dist, nearest, restarts, dropped)


def _drop_short_runs(label_arc, route_pos, label_dist, max_back: int = 10, min_run: int = 30) -> int:
    """Unlabel short runs isolated by large backward jumps in route position.

    A large backward jump means part of the GPS trace only matches an unrelated part of the
    route (e.g. the recording starts before the route does, near where the route later ends).
    Such runs are unreliable labels, so they are left unlabelled. Returns points unlabelled.
    """
    idx = np.where(route_pos >= 0)[0]
    if len(idx) < 2:
        return 0
    cut = np.where(np.diff(route_pos[idx]) < -max_back)[0] + 1
    runs = np.split(idx, cut)
    if len(runs) == 1:
        return 0
    dropped = 0
    for r in runs:
        if len(r) < min_run:
            label_arc[r], route_pos[r], label_dist[r] = -1, -1, np.nan
            dropped += len(r)
    return dropped
