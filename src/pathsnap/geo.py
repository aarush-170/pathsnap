"""Geometry helpers. All spatial computations are done in metres, never in raw degrees.

We use a local equirectangular projection centred on each trip. Trips span at most a few tens of
kilometres, where this projection's distortion is well below GPS noise (< 0.1 % over 20 km),
and it is far cheaper than a full UTM transform. haversine() is provided for cross-checking.
"""
from __future__ import annotations

import numpy as np

EARTH_R = 6_371_008.8  # mean Earth radius, metres


class LocalProjector:
    """lon/lat (degrees) <-> x/y (metres) around a reference point."""

    def __init__(self, lon0: float, lat0: float):
        self.lon0 = float(lon0)
        self.lat0 = float(lat0)
        self._kx = np.deg2rad(1.0) * EARTH_R * np.cos(np.deg2rad(self.lat0))
        self._ky = np.deg2rad(1.0) * EARTH_R

    @classmethod
    def for_points(cls, lon, lat) -> "LocalProjector":
        return cls(np.nanmean(lon), np.nanmean(lat))

    def to_xy(self, lon, lat) -> np.ndarray:
        lon = np.asarray(lon, dtype=float)
        lat = np.asarray(lat, dtype=float)
        return np.stack([(lon - self.lon0) * self._kx, (lat - self.lat0) * self._ky], axis=-1)

    def to_lonlat(self, xy) -> np.ndarray:
        xy = np.asarray(xy, dtype=float)
        return np.stack([xy[..., 0] / self._kx + self.lon0, xy[..., 1] / self._ky + self.lat0], axis=-1)


def haversine(lon1, lat1, lon2, lat2) -> np.ndarray:
    lon1, lat1, lon2, lat2 = map(np.deg2rad, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_R * np.arcsin(np.sqrt(a))


def point_segment_distance(P, A, B):
    """Distance from points P (n,2) to segments A->B (m,2) -> (dist (n,m), t (n,m)).

    t in [0,1] is the fractional position of the closest point along each segment.
    """
    P = np.asarray(P, float)[:, None, :]
    A = np.asarray(A, float)[None, :, :]
    AB = np.asarray(B, float)[None, :, :] - A
    L2 = (AB ** 2).sum(-1)
    t = np.where(L2 > 0, ((P - A) * AB).sum(-1) / np.where(L2 > 0, L2, 1.0), 0.0)
    t = np.clip(t, 0.0, 1.0)
    Q = A + t[..., None] * AB
    return np.sqrt(((P - Q) ** 2).sum(-1)), t


def min_distance_to_segments(P, A, B, chunk: int = 512) -> np.ndarray:
    """For each point, the distance to the nearest of the given segments (memory-safe)."""
    P = np.asarray(P, float)
    out = np.empty(len(P))
    for i in range(0, len(P), chunk):
        d, _ = point_segment_distance(P[i:i + chunk], A, B)
        out[i:i + chunk] = d.min(1)
    return out


def bearing_deg(dx, dy) -> np.ndarray:
    """Compass bearing (0 = north, clockwise) of a displacement in projected metres."""
    return (np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0


def angle_diff_deg(a, b) -> np.ndarray:
    """Smallest absolute difference between two bearings, in [0, 180]."""
    d = np.abs((np.asarray(a) - np.asarray(b) + 180.0) % 360.0 - 180.0)
    return d
