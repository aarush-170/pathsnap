"""Candidate road segments for each GPS point.

For every GPS point we return the arcs within `radius` metres, keeping the closest ones by
distance rank. Ranks use "min" tie-breaking on distance rounded to 1 cm, so the two directions of
a two-way street (identical geometry, identical distance) always share a rank and are kept or
dropped together; telling them apart is the classifier's job (heading features).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .geo import LocalProjector
from .io import Trip
from .spatial import SegmentIndex


@dataclass
class TripGeometry:
    proj: LocalProjector
    P: np.ndarray        # (n,2) GPS points in metres
    N: np.ndarray        # (num_nodes,2) nodes in metres


def trip_geometry(trip: Trip) -> TripGeometry:
    proj = LocalProjector.for_points(trip.track.lon.values, trip.track.lat.values)
    return TripGeometry(proj, proj.to_xy(trip.track.lon.values, trip.track.lat.values),
                        proj.to_xy(trip.nodes[:, 0], trip.nodes[:, 1]))


def distance_rank(point_idx: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """0-based rank of each candidate's distance within its point ('min' method on 1 cm bins)."""
    df = pd.DataFrame({"p": point_idx, "d": np.round(dist, 2)})
    return (df.groupby("p")["d"].rank(method="min").to_numpy() - 1).astype(np.int32)


def generate_candidates(trip: Trip, radius: float, max_rank: int | None = None,
                        geom: TripGeometry | None = None) -> pd.DataFrame:
    """Columns: point_idx, arc, dist, t, rank. Sorted by point_idx, dist."""
    g = geom or trip_geometry(trip)
    a = trip.arcs
    A, B = g.N[a[:, 0]], g.N[a[:, 1]]
    lo, hi = g.P.min(0) - radius, g.P.max(0) + radius
    near = ((np.minimum(A, B) <= hi) & (np.maximum(A, B) >= lo)).all(1)   # bbox pre-filter
    arc_ids = np.where(near)[0]
    pi, si, d, t = SegmentIndex(A[arc_ids], B[arc_ids]).query(g.P, radius)
    df = pd.DataFrame({"point_idx": pi.astype(np.int32), "arc": arc_ids[si].astype(np.int64),
                       "dist": d.astype(np.float32), "t": t.astype(np.float32)})
    df = df.sort_values(["point_idx", "dist"], kind="stable").reset_index(drop=True)
    df["rank"] = distance_rank(df.point_idx.to_numpy(), df.dist.to_numpy())
    if max_rank is not None:
        df = df[df["rank"] < max_rank].reset_index(drop=True)
    return df
