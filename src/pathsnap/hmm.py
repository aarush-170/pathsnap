"""HMM + Viterbi route reconstruction.

Choosing each GPS point's road independently (classifier-only) can produce a sequence that jumps
between roads that are not connected. The HMM decodes the whole trip at once:

  hidden state     the road a point is on: one of that point's candidate arcs, together with the
                   point's projected position t along the arc (so the state is a place on the network)
  emission         alpha * log P(arc | point), the classifier probability normalised over the point's
                   candidates (alpha weighs the classifier against the network)
  transition       log P(b at i | a at i-1) = -|d_net - d_gps| / beta
                   d_gps = straight-line distance between the two GPS points
                   d_net = distance a vehicle must drive on the directed network from (a, t_a) to (b, t_b):
                           same arc: |t_b - t_a| * len(a)
                           else:     (1 - t_a) * len(a) + shortest_path(head(a) -> tail(b)) + t_b * len(b)
                   Real driving makes d_net ~ d_gps; a jump to an unconnected road needs a long detour
                   (large |d_net - d_gps|) or is unreachable (-inf).
  decoding         Viterbi (max-sum dynamic programming) with back-pointers, in log space.

Special cases
  * time gap > gap_break_s, or no reachable transition at all  -> the chain is cut and a new segment
    starts (decoded independently) instead of forcing an impossible connection
  * points without candidates are skipped; the transition is taken from the last point that had any
  * shortest paths use Dijkstra with a distance cut-off (local search), cached per source node
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .candidates import TripGeometry


@dataclass
class HMMParams:
    beta: float = 10.0          # metres; transition tolerance for |d_net - d_gps|
    alpha: float = 1.0          # weight of the classifier emission
    gap_break_s: float = 60.0   # cut the chain at sampling gaps longer than this
    route_slack_m: float = 200.0   # Dijkstra cut-off = route_factor * d_gps + route_slack_m
    route_factor: float = 3.0
    eps: float = 1e-6


class RoadGraph:
    """Directed road network with arc lengths in metres and a cached, cut-off Dijkstra."""

    def __init__(self, arcs: np.ndarray, N: np.ndarray):
        self.arcs = arcs
        self.n_nodes = len(N)
        self.length = np.linalg.norm(N[arcs[:, 1]] - N[arcs[:, 0]], axis=1)
        order = np.argsort(arcs[:, 0], kind="stable")
        self._out_arcs = order
        self._ptr = np.searchsorted(arcs[order, 0], np.arange(self.n_nodes + 1))
        self._cache: dict[int, tuple[float, dict, dict]] = {}

    def out_arcs(self, node: int) -> np.ndarray:
        return self._out_arcs[self._ptr[node]:self._ptr[node + 1]]

    def dijkstra(self, src: int, cutoff: float):
        """Distances (m) and predecessor arcs from node src, exploring up to `cutoff` metres."""
        hit = self._cache.get(src)
        if hit is not None and hit[0] >= cutoff:
            return hit[1], hit[2]
        dist, pred = {src: 0.0}, {src: -1}
        heap = [(0.0, src)]
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist.get(u, np.inf) or d > cutoff:
                continue
            for a in self.out_arcs(u):
                v = int(self.arcs[a, 1])
                nd = d + self.length[a]
                if nd <= cutoff and nd < dist.get(v, np.inf):
                    dist[v], pred[v] = nd, int(a)
                    heapq.heappush(heap, (nd, v))
        self._cache[src] = (cutoff, dist, pred)
        return dist, pred

    def path_arcs(self, a: int, b: int, cutoff: float = 2000.0) -> list[int] | None:
        """Arcs strictly between arc a and arc b on a shortest path head(a) -> tail(b) ([] if adjacent)."""
        src, dst = int(self.arcs[a, 1]), int(self.arcs[b, 0])
        if src == dst:
            return []
        dist, pred = self.dijkstra(src, cutoff)
        if dst not in dist:
            return None
        out, v = [], dst
        while v != src:
            e = pred[v]
            out.append(e)
            v = int(self.arcs[e, 0])
        return out[::-1]


def decode_trip(graph: RoadGraph, P: np.ndarray, ts: np.ndarray, cands: pd.DataFrame, prob: np.ndarray,
                params: HMMParams) -> tuple[pd.Series, dict]:
    """Viterbi decoding of one trip.

    cands: rows (point_idx, arc, t) for every candidate; prob: classifier P(is_true) per row.
    Returns (predicted arc per point_idx, diagnostics).
    """
    df = pd.DataFrame({"p": cands.point_idx.to_numpy(), "a": cands.arc.to_numpy(),
                       "t": cands.t.to_numpy(dtype=float), "q": prob})
    df = df.sort_values(["p", "a"], kind="stable")
    pts = df.p.unique()
    bounds = np.searchsorted(df.p.to_numpy(), pts)
    bounds = np.r_[bounds, len(df)]
    A, T, Q = df.a.to_numpy(), df.t.to_numpy(), df.q.to_numpy()
    L = graph.length

    layers = []          # per point: (arcs, back-pointer array, segment start flag)
    n_breaks = n_gap_breaks = 0
    prev = None          # (arcs, t, score, point index in trip)
    for k, i in enumerate(pts):
        lo, hi = bounds[k], bounds[k + 1]
        a_cur, t_cur = A[lo:hi], T[lo:hi]
        q = np.maximum(Q[lo:hi], params.eps)
        emit = params.alpha * np.log(q / q.sum())
        start = prev is None
        if not start and ts[i] - ts[prev[3]] > params.gap_break_s:
            start, n_gap_breaks = True, n_gap_breaks + 1
        if not start:
            a_prev, t_prev, s_prev, ip = prev
            d_gps = float(np.linalg.norm(P[i] - P[ip]))
            cutoff = params.route_factor * d_gps + params.route_slack_m
            trans = np.full((len(a_prev), len(a_cur)), -np.inf)
            for r, (a, ta) in enumerate(zip(a_prev, t_prev)):
                dist, _ = graph.dijkstra(int(graph.arcs[a, 1]), cutoff)
                rest = (1.0 - ta) * L[a]
                for c, (b, tb) in enumerate(zip(a_cur, t_cur)):
                    if a == b:
                        d_net = abs(tb - ta) * L[a]
                    else:
                        sp = dist.get(int(graph.arcs[b, 0]))
                        if sp is None:
                            continue
                        d_net = rest + sp + tb * L[b]
                    trans[r, c] = -abs(d_net - d_gps) / params.beta
            tot = s_prev[:, None] + trans
            back = np.argmax(tot, axis=0)
            best = tot[back, np.arange(len(a_cur))]
            if np.isfinite(best).any():
                score = best + emit
                score[~np.isfinite(best)] = -np.inf
            else:                       # nothing reachable: cut the chain here
                start, n_breaks = True, n_breaks + 1
        if start:
            score, back = emit.copy(), np.full(len(a_cur), -1)
        score = score - np.max(score)   # keep numbers bounded
        layers.append((a_cur, back, start, i, score))
        prev = (a_cur, t_cur, score, i)
    # backtrack. A layer that starts a segment has no back-pointer, so the layer before it is the
    # last point of the previous segment: its own best-scoring state starts that segment's backtrack.
    pred, q = {}, None
    for k in range(len(layers) - 1, -1, -1):
        a_cur, back, start, i, score = layers[k]
        if q is None:
            q = int(np.argmax(score))
        pred[i] = int(a_cur[q])
        q = None if start else int(back[q])
    return pd.Series(pred, dtype="int64").sort_index(), dict(segments=1 + n_breaks + n_gap_breaks,
                                                             breaks=n_breaks, gap_breaks=n_gap_breaks)
