"""Reading raw PathSnap trip files.

Each trip directory <trip_id>/ contains headerless, whitespace-separated files:
  <id>.track  lon lat timestamp_s      (GPS observations)
  <id>.nodes  lon lat                  (row index = node_id)
  <id>.arcs   from_node to_node        (row index = arc_id, directed)
  <id>.route  arc_id                   (ordered reference route; absent on evaluator side for the model)
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class Trip:
    trip_id: str
    split: str
    track: pd.DataFrame            # columns: lon, lat, t
    nodes: np.ndarray              # (N, 2) lon, lat
    arcs: np.ndarray               # (M, 2) int64 from, to
    route: np.ndarray | None = None  # (R,) int64 arc ids, or None
    meta: dict = field(default_factory=dict)

    @property
    def n_points(self) -> int:
        return len(self.track)


def _read(path: Path, names, dtype=None) -> pd.DataFrame:
    return pd.read_csv(path, sep=r"\s+", header=None, names=names, dtype=dtype, engine="c")


def load_trip(trip_dir: str | os.PathLike, split: str = "", load_route: bool = True) -> Trip:
    trip_dir = Path(trip_dir)
    tid = trip_dir.name
    track = _read(trip_dir / f"{tid}.track", ["lon", "lat", "t"], dtype=float)
    nodes = _read(trip_dir / f"{tid}.nodes", ["lon", "lat"], dtype=float).to_numpy()
    arcs = _read(trip_dir / f"{tid}.arcs", ["u", "v"], dtype=np.int64).to_numpy()
    route = None
    rpath = trip_dir / f"{tid}.route"
    if load_route and rpath.exists():
        route = _read(rpath, ["arc"], dtype=np.int64)["arc"].to_numpy()
    return Trip(tid, split, track, nodes, arcs, route)


def list_trips(split_dir: str | os.PathLike) -> list[Path]:
    split_dir = Path(split_dir)
    if not split_dir.exists():
        return []
    return sorted(p for p in split_dir.iterdir() if p.is_dir())


def iter_trips(split_dir: str | os.PathLike, split: str = "", load_route: bool = True):
    for d in list_trips(split_dir):
        yield load_trip(d, split=split, load_route=load_route)


def file_inventory(split_dir: str | os.PathLike, split: str) -> pd.DataFrame:
    """One row per (trip, file type): existence, size in bytes, line count."""
    rows = []
    for d in list_trips(split_dir):
        for ext in ("track", "nodes", "arcs", "route"):
            p = d / f"{d.name}.{ext}"
            exists = p.exists()
            n_lines = 0
            if exists:
                with open(p, "rb") as fh:
                    n_lines = sum(1 for _ in fh)
            rows.append(dict(split=split, trip=d.name, file=ext, exists=exists,
                             bytes=p.stat().st_size if exists else 0, rows=n_lines))
    return pd.DataFrame(rows)
