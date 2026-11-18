"""Evaluation metrics shared by experiments, the HMM comparison and the API."""
from __future__ import annotations

import numpy as np
import pandas as pd


def top1_predictions(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    """Highest-scoring candidate per (trip, point). Ties broken by distance rank then row order."""
    order = df.sort_values(["trip", "point_idx", score_col, "rank"], ascending=[True, True, False, True],
                           kind="stable")
    return order.drop_duplicates(["trip", "point_idx"], keep="first")


def top1_accuracy(df: pd.DataFrame, score_col: str, labelled_points: int | None = None) -> float:
    """Observation-level Top-1 accuracy over labelled points.

    df must contain only candidates of labelled points (column is_true). The denominator is the
    number of labelled points; points whose true arc is not in the candidate set count as wrong.
    """
    top = top1_predictions(df, score_col)
    denom = labelled_points if labelled_points is not None else len(top)
    return float(top.is_true.sum() / denom) if denom else float("nan")


def candidate_miss_rate(labels: pd.DataFrame) -> float:
    m = labels.label_arc >= 0
    return float((~labels.true_in_candidates[m]).mean())


def route_continuity(arcs: np.ndarray, arc_seq: np.ndarray) -> float:
    """Fraction of changes between consecutive (distinct) predicted arcs that are physically
    connected head-to-tail in the directed road network. 1.0 = a fully drivable sequence."""
    s = np.asarray(arc_seq)
    s = s[s >= 0]
    if len(s) < 2:
        return float("nan")
    change = s[1:] != s[:-1]
    if not change.any():
        return 1.0
    a, b = s[:-1][change], s[1:][change]
    return float(np.mean(arcs[a, 1] == arcs[b, 0]))
