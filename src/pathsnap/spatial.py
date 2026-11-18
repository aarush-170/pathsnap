"""Fast radius search from points to line segments.

Segments are densified into sample points every `spacing` metres and indexed with a KD-tree.
Any segment within distance r of a query point has a sample within r + spacing/2, so a
ball query with that enlarged radius gives a complete candidate set; exact point-to-segment
distances are then computed only for those pairs.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


class SegmentIndex:
    def __init__(self, A: np.ndarray, B: np.ndarray, spacing: float = 10.0, seg_ids: np.ndarray | None = None):
        self.A = np.asarray(A, float)
        self.B = np.asarray(B, float)
        self.seg_ids = np.arange(len(self.A)) if seg_ids is None else np.asarray(seg_ids)
        self.spacing = float(spacing)
        L = np.linalg.norm(self.B - self.A, axis=1)
        n = np.maximum(1, np.ceil(L / spacing).astype(int)) + 1        # samples per segment
        owner = np.repeat(np.arange(len(self.A)), n)
        start = np.repeat(np.cumsum(n) - n, n)
        frac = (np.arange(n.sum()) - start) / np.maximum(np.repeat(n, n) - 1, 1)
        self._owner = owner
        samples = self.A[owner] + frac[:, None] * (self.B[owner] - self.A[owner])
        self._tree = cKDTree(samples) if len(samples) else None

    def query(self, P: np.ndarray, r: float):
        """All (point, segment) pairs with distance <= r.

        Returns point_idx, local_seg_idx, dist, t (fraction along segment), each 1-D.
        """
        P = np.asarray(P, float)
        empty = (np.array([], int), np.array([], int), np.array([]), np.array([]))
        if self._tree is None or len(P) == 0:
            return empty
        hits = self._tree.query_ball_point(P, r + self.spacing / 2 + 1e-6, workers=-1)
        lens = np.fromiter((len(h) for h in hits), dtype=np.int64, count=len(hits))
        if lens.sum() == 0:
            return empty
        pi = np.repeat(np.arange(len(P)), lens)
        si = self._owner[np.concatenate([np.asarray(h, dtype=np.int64) for h in hits if len(h)])]
        key = np.unique(pi * len(self.A) + si)
        pi, si = key // len(self.A), key % len(self.A)
        a, b, p = self.A[si], self.B[si], P[pi]
        ab = b - a
        L2 = (ab ** 2).sum(1)
        t = np.where(L2 > 0, ((p - a) * ab).sum(1) / np.where(L2 > 0, L2, 1.0), 0.0).clip(0, 1)
        d = np.linalg.norm(p - (a + t[:, None] * ab), axis=1)
        keep = d <= r
        return pi[keep], si[keep], d[keep], t[keep]
