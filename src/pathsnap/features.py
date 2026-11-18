"""Feature engineering for (GPS point, candidate arc) pairs.

Every row of the candidate table gets a feature vector. Features only use the GPS track and the
road network (never the reference route), so the exact same code runs on evaluator trips inside
the API. Groups:

  distance   how close the GPS point is to the arc, absolute and relative to the other candidates
  heading    does the arc point the same way the vehicle is moving (separates the two directions
             of a two-way street, which have identical distances)
  context    how well the arc also explains the neighbouring GPS points (i-2 .. i+2)
  motion     speed / sampling gaps (how much to trust heading and context)
  network    local road topology around the arc
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .candidates import TripGeometry
from .geo import angle_diff_deg, bearing_deg
from .io import Trip

HEADING_WINDOW = 2          # GPS heading from points i-k .. i+k
CONTEXT_OFFSETS = (-2, -1, 1, 2)
CONTEXT_EXTEND_M = 30.0     # extend the arc this far at both ends when scoring neighbour points
MIN_HEADING_DISP_M = 3.0    # below this displacement the GPS heading is treated as unknown

FEATURES = [
    # distance
    "dist", "rank", "dist_minus_min", "dist_ratio_min", "dist_z", "n_candidates", "gap_1st_2nd",
    # projection
    "t", "t_at_end", "arc_len",
    # heading
    "heading_cos", "heading_absdiff", "heading_cos_prev", "heading_cos_next", "heading_rank",
    "heading_valid",
    # context
    "ctx_dist_mean", "ctx_dist_max", "ctx_minus_min", "ctx_rank",
    # motion
    "speed", "dt_prev", "dt_next",
    # network
    "from_in_degree", "to_out_degree", "has_reverse_twin",
]


def _seg_dist(P, A, B, extend: float = 0.0):
    """Row-wise distance from P[k] to segment A[k]-B[k], optionally extended at both ends."""
    ab = B - A
    L = np.linalg.norm(ab, axis=1)
    safe = np.where(L > 0, L, 1.0)
    t = ((P - A) * ab).sum(1) / safe ** 2
    lo, hi = -extend / safe, 1 + extend / safe
    t = np.clip(t, np.where(L > 0, lo, 0), np.where(L > 0, hi, 0))
    return np.linalg.norm(P - (A + t[:, None] * ab), axis=1)


def _group_rank(points: np.ndarray, values: np.ndarray, ascending: bool = True) -> np.ndarray:
    s = pd.Series(values).groupby(points).rank(method="min", ascending=ascending)
    return (s.to_numpy() - 1).astype(np.float32)


def trip_features(trip: Trip, geom: TripGeometry, cands: pd.DataFrame) -> pd.DataFrame:
    """Feature table aligned row-for-row with `cands` (columns point_idx, arc, dist, t, rank)."""
    P, N = geom.P, geom.N
    n = len(P)
    ts = trip.track.t.to_numpy()
    pi = cands.point_idx.to_numpy()
    arc = cands.arc.to_numpy()
    u, v = trip.arcs[arc, 0], trip.arcs[arc, 1]
    A, B = N[u], N[v]
    f = pd.DataFrame(index=cands.index)

    # ---- distance
    d = cands.dist.to_numpy().astype(float)
    g = pd.Series(d).groupby(pi)
    dmin = g.transform("min").to_numpy()
    dmean = g.transform("mean").to_numpy()
    dstd = g.transform("std").fillna(0).to_numpy()
    f["dist"] = d
    f["rank"] = cands["rank"].to_numpy()
    f["dist_minus_min"] = d - dmin
    f["dist_ratio_min"] = (d + 1.0) / (dmin + 1.0)
    f["dist_z"] = np.where(dstd > 0, (d - dmean) / np.where(dstd > 0, dstd, 1), 0.0)
    f["n_candidates"] = g.transform("size").to_numpy()
    # gap between the nearest and second-nearest distinct distance at this point (ambiguity)
    # (no second distinct distance -> gap set to the 50 m search radius: unambiguous)
    uniq = pd.DataFrame({"p": pi, "d": np.round(d, 2)}).drop_duplicates().sort_values(["p", "d"])
    r = uniq.groupby("p").cumcount().to_numpy()
    first = uniq[r == 0].set_index("p")["d"]
    second = uniq[r == 1].set_index("p")["d"]
    f["gap_1st_2nd"] = (second - first).reindex(pi).fillna(50.0).to_numpy()

    # ---- projection
    t = cands.t.to_numpy().astype(float)
    f["t"] = t
    f["t_at_end"] = ((t <= 1e-3) | (t >= 1 - 1e-3)).astype(np.int8)
    f["arc_len"] = np.linalg.norm(B - A, axis=1)

    # ---- heading
    k = HEADING_WINDOW
    lo_i, hi_i = np.clip(np.arange(n) - k, 0, n - 1), np.clip(np.arange(n) + k, 0, n - 1)
    disp = P[hi_i] - P[lo_i]
    disp_len = np.linalg.norm(disp, axis=1)
    gps_bear = bearing_deg(disp[:, 0], disp[:, 1])
    step_prev = P - P[np.clip(np.arange(n) - 1, 0, n - 1)]
    step_next = P[np.clip(np.arange(n) + 1, 0, n - 1)] - P
    arc_bear = bearing_deg(B[:, 0] - A[:, 0], B[:, 1] - A[:, 1])
    valid = disp_len[pi] >= MIN_HEADING_DISP_M

    def cos_to(vecs, ok):
        b = bearing_deg(vecs[pi, 0], vecs[pi, 1])
        return np.where(ok, np.cos(np.deg2rad(b - arc_bear)), 0.0)

    f["heading_cos"] = np.where(valid, np.cos(np.deg2rad(gps_bear[pi] - arc_bear)), 0.0)
    f["heading_absdiff"] = np.where(valid, angle_diff_deg(gps_bear[pi], arc_bear), 90.0)
    f["heading_cos_prev"] = cos_to(step_prev, np.linalg.norm(step_prev, axis=1)[pi] >= 1.0)
    f["heading_cos_next"] = cos_to(step_next, np.linalg.norm(step_next, axis=1)[pi] >= 1.0)
    f["heading_rank"] = _group_rank(pi, f["heading_cos"].to_numpy(), ascending=False)
    f["heading_valid"] = valid.astype(np.int8)

    # ---- context: distance from neighbouring GPS points to this (slightly extended) arc
    ctx = np.empty((len(cands), len(CONTEXT_OFFSETS) + 1))
    ctx[:, 0] = d
    for c, o in enumerate(CONTEXT_OFFSETS, start=1):
        q = np.clip(pi + o, 0, n - 1)
        ctx[:, c] = _seg_dist(P[q], A, B, extend=CONTEXT_EXTEND_M)
    f["ctx_dist_mean"] = ctx.mean(1)
    f["ctx_dist_max"] = ctx.max(1)
    cmin = pd.Series(f["ctx_dist_mean"].to_numpy()).groupby(pi).transform("min").to_numpy()
    f["ctx_minus_min"] = f["ctx_dist_mean"].to_numpy() - cmin
    f["ctx_rank"] = _group_rank(pi, np.round(f["ctx_dist_mean"].to_numpy(), 2), ascending=True)

    # ---- motion
    dt_win = np.maximum(ts[hi_i] - ts[lo_i], 1e-6)
    f["speed"] = (disp_len / dt_win)[pi]
    dtp = np.r_[np.nan, np.diff(ts)]
    dtn = np.r_[np.diff(ts), np.nan]
    f["dt_prev"] = np.nan_to_num(dtp, nan=1.0)[pi]
    f["dt_next"] = np.nan_to_num(dtn, nan=1.0)[pi]

    # ---- network
    nn = len(trip.nodes)
    indeg = np.bincount(trip.arcs[:, 1], minlength=nn)
    outdeg = np.bincount(trip.arcs[:, 0], minlength=nn)
    f["from_in_degree"] = indeg[u]
    f["to_out_degree"] = outdeg[v]
    key = trip.arcs[:, 0].astype(np.int64) * (nn + 1) + trip.arcs[:, 1]
    rkey = v.astype(np.int64) * (nn + 1) + u
    f["has_reverse_twin"] = np.isin(rkey, key).astype(np.int8)

    return f[FEATURES].astype(np.float32)
