"""Phase 1: exploratory data analysis of the raw PathSnap dataset.

Usage (from repo root):  python -m scripts.eda
Outputs:
  results/logs/eda_file_inventory.csv   file existence / size / row counts
  results/logs/eda_trip_summary.csv     one row per trip: track, network and route statistics
  results/logs/eda_split_summary.csv    describe() per split
  results/logs/eda_anomalies.csv        flagged trips and reasons
  results/figures/eda_*.png             plots
  results/maps/eda_*.html               Folium maps
Only train and student_test partitions are analysed.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection

from pathsnap.geo import LocalProjector, haversine, min_distance_to_segments
from pathsnap.io import file_inventory, iter_trips, load_trip
from pathsnap.network import network_stats, route_breaks
from pathsnap.viz import trip_map

RAW = ROOT / "data" / "raw"
SPLITS = {"train": RAW / "train", "student_test": RAW / "student_test"}
FIG = ROOT / "results" / "figures"
LOG = ROOT / "results" / "logs"
MAPS = ROOT / "results" / "maps"
FAR_M = 50.0  # a GPS point this far from every route arc is considered unexplained by the route


def _run_length(mask: np.ndarray) -> int:
    """Length of the leading run of True values."""
    idx = np.where(~mask)[0]
    return int(idx[0]) if len(idx) else int(len(mask))


def trip_stats(trip, pool: dict) -> dict:
    tr = trip.track
    lon, lat, t = tr.lon.values, tr.lat.values, tr.t.values
    proj = LocalProjector.for_points(lon, lat)
    xy = proj.to_xy(lon, lat)
    dt = np.diff(t)
    step = haversine(lon[:-1], lat[:-1], lon[1:], lat[1:])
    with np.errstate(divide="ignore", invalid="ignore"):
        speed = np.where(dt > 0, step / dt * 3.6, np.nan)  # km/h
    med_lon, med_lat = np.median(lon), np.median(lat)
    far_from_median = haversine(lon, lat, med_lon, med_lat) > 50_000

    s = dict(split=trip.split, trip=trip.trip_id, n_points=len(tr),
             nan_values=int(tr.isna().sum().sum()),
             invalid_coords=int(((np.abs(lat) > 90) | (np.abs(lon) > 180)).sum() + far_from_median.sum()),
             duplicate_timestamps=int((dt == 0).sum()), non_monotonic_timestamps=int((dt < 0).sum()),
             duration_s=float(t[-1] - t[0]), dt_median=float(np.median(dt)),
             frac_dt_eq_1=float(np.mean(dt == 1)), gaps_gt_5s=int((dt > 5).sum()),
             gaps_gt_30s=int((dt > 30).sum()), gaps_gt_60s=int((dt > 60).sum()), max_gap_s=float(dt.max()),
             path_length_km=float(step.sum() / 1000), stationary_frac=float(np.mean(step < 0.5)),
             speed_median_kmh=float(np.nanmedian(speed)), speed_p99_kmh=float(np.nanpercentile(speed, 99)),
             speed_jumps_gt_200kmh=int(np.nansum(speed > 200)),
             extent_x_km=float(np.ptp(xy[:, 0]) / 1000), extent_y_km=float(np.ptp(xy[:, 1]) / 1000),
             lon_min=float(lon.min()), lon_max=float(lon.max()), lat_min=float(lat.min()), lat_max=float(lat.max()))

    s.update(network_stats(trip.nodes, trip.arcs))
    nxy = proj.to_xy(trip.nodes[:, 0], trip.nodes[:, 1])
    arc_len = np.linalg.norm(nxy[trip.arcs[:, 1]] - nxy[trip.arcs[:, 0]], axis=1)
    s.update(arc_len_median_m=float(np.median(arc_len)), zero_length_arcs=int((arc_len < 0.01).sum()))

    pool["dt"].append(dt)
    pool["speed"].append(speed[np.isfinite(speed)])
    pool["arc_len"].append(arc_len)

    r = trip.route
    if r is not None:
        bad = (r < 0) | (r >= len(trip.arcs))
        rv = r[~bad]
        A, B = nxy[trip.arcs[rv, 0]], nxy[trip.arcs[rv, 1]]
        d = min_distance_to_segments(xy, A, B)
        far = d > FAR_M
        s.update(route_arcs=len(r), route_unique_arcs=int(len(np.unique(r))),
                 route_invalid_ids=int(bad.sum()), route_breaks=int(len(route_breaks(trip.arcs, rv))),
                 route_length_km=float(arc_len[rv].sum() / 1000),
                 points_per_route_arc=float(len(tr) / max(len(r), 1)),
                 gps_route_dist_median_m=float(np.median(d)), gps_route_dist_p95_m=float(np.percentile(d, 95)),
                 gps_route_dist_max_m=float(d.max()), frac_points_gt_30m=float(np.mean(d > 30)),
                 frac_points_gt_50m=float(far.mean()),
                 far_points_at_start=_run_length(far), far_points_at_end=_run_length(far[::-1]))
        pool["d_route"].append((trip.split, d))
    return s


def flag_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    rules = {
        "gap > 60 s": df.gaps_gt_60s > 0,
        "> 1% points > 50 m from route": df.frac_points_gt_50m > 0.01,
        "route not continuous": df.route_breaks > 0,
        "speed jump > 200 km/h": df.speed_jumps_gt_200kmh > 0,
        "> 30% stationary samples": df.stationary_frac > 0.30,
        "duplicate / non-monotonic timestamps": (df.duplicate_timestamps + df.non_monotonic_timestamps) > 0,
        "missing / invalid values": (df.nan_values + df.invalid_coords) > 0,
    }
    rows = []
    for _, row in df.iterrows():
        reasons = [k for k, m in rules.items() if m.loc[row.name]]
        if reasons:
            rows.append(dict(split=row.split, trip=row.trip, reasons="; ".join(reasons),
                             max_gap_s=row.max_gap_s, frac_points_gt_50m=round(row.frac_points_gt_50m, 4),
                             far_points_at_start=row.far_points_at_start, far_points_at_end=row.far_points_at_end,
                             route_breaks=row.route_breaks))
    counts = pd.Series({k: int(m.sum()) for k, m in rules.items()}, name="n_trips")
    return pd.DataFrame(rows), counts


# ----------------------------------------------------------------------------- plots
def _save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=150)
    plt.close(fig)


def plot_trip_axes(ax, trip, buffer_m=150.0, show_far=False):
    proj = LocalProjector.for_points(trip.track.lon, trip.track.lat)
    xy = proj.to_xy(trip.track.lon.values, trip.track.lat.values)
    nxy = proj.to_xy(trip.nodes[:, 0], trip.nodes[:, 1])
    lo, hi = xy.min(0) - buffer_m, xy.max(0) + buffer_m
    a = trip.arcs
    keep = ((nxy[a[:, 0]] >= lo) & (nxy[a[:, 0]] <= hi)).all(1)
    ax.add_collection(LineCollection(np.stack([nxy[a[keep, 0]], nxy[a[keep, 1]]], 1) / 1000,
                                     colors="#bbbbbb", linewidths=0.4, label="road network"))
    if trip.route is not None:
        r = trip.route
        ax.add_collection(LineCollection(np.stack([nxy[a[r, 0]], nxy[a[r, 1]]], 1) / 1000,
                                         colors="#2ca02c", linewidths=1.6, alpha=0.9, zorder=4,
                                         label="reference route"))
    ax.scatter(xy[:, 0] / 1000, xy[:, 1] / 1000, s=4, c="#1f77b4", alpha=0.35, lw=0, label="GPS", zorder=3)
    ax.plot(*(xy[0] / 1000), "g^", ms=7, zorder=4)
    ax.plot(*(xy[-1] / 1000), "rs", ms=6, zorder=4)
    ax.set_xlim(lo[0] / 1000, hi[0] / 1000)
    ax.set_ylim(lo[1] / 1000, hi[1] / 1000)
    ax.set_aspect("equal")
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")


def make_plots(df, pool, anomalies_ids):
    sns_colors = {"train": "#4c72b0", "student_test": "#dd8452"}

    # 1. points per trip
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for sp, g in df.groupby("split"):
        ax[0].hist(g.n_points, bins=np.logspace(2, 4.5, 30), alpha=0.7, label=f"{sp} (n={len(g)})",
                   color=sns_colors[sp])
        ax[1].scatter(g.n_points, g.route_arcs, s=14, alpha=0.8, label=sp, color=sns_colors[sp])
    ax[0].set_xscale("log"); ax[0].set_xlabel("GPS points per trip"); ax[0].set_ylabel("trips"); ax[0].legend()
    ax[0].set_title("Trajectory length")
    ax[1].set_xscale("log"); ax[1].set_yscale("log"); ax[1].set_xlabel("GPS points"); ax[1].set_ylabel("route arcs")
    ax[1].set_title("GPS points vs reference-route arcs"); ax[1].legend()
    _save(fig, "eda_points_per_trip.png")

    # 2. sampling intervals
    dt = np.concatenate(pool["dt"])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    vals, cnt = np.unique(np.clip(np.round(dt), 0, 11), return_counts=True)
    ax[0].bar(vals, cnt, color="#4c72b0"); ax[0].set_yscale("log")
    ax[0].set_xticks(vals); ax[0].set_xticklabels([str(int(v)) if v < 11 else ">10" for v in vals])
    ax[0].set_xlabel("time between consecutive samples (s, rounded)"); ax[0].set_ylabel("count (log)")
    ax[0].set_title(f"Sampling interval: {np.mean(dt == 1):.1%} exactly 1 s")
    big = dt[dt > 5]
    ax[1].hist(big, bins=np.logspace(np.log10(5), np.log10(max(big.max(), 10)), 30), color="#c44e52")
    ax[1].set_xscale("log"); ax[1].set_xlabel("gap length (s)"); ax[1].set_ylabel("count")
    ax[1].set_title(f"Gaps > 5 s (n={len(big)}, all splits)")
    _save(fig, "eda_sampling_intervals.png")

    # 3. speed
    sp = np.concatenate(pool["speed"])
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.hist(np.clip(sp, 0, 200), bins=100, color="#55a868")
    ax.set_xlabel("instantaneous speed (km/h, clipped at 200)"); ax.set_ylabel("count")
    ax.set_title(f"Speed between consecutive GPS points (median {np.median(sp):.1f} km/h)")
    _save(fig, "eda_speed_distribution.png")

    # 4. GPS -> route distance
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for split in ("train", "student_test"):
        d = np.concatenate([x for s_, x in pool["d_route"] if s_ == split])
        ax[0].hist(np.clip(d, 0, 60), bins=np.arange(0, 61, 1), alpha=0.6, density=True,
                   label=f"{split} (>60 m: {np.mean(d > 60):.1%})", color=sns_colors[split])
        ds = np.clip(np.sort(d), 0.1, None)
        ax[1].plot(ds, np.arange(1, len(ds) + 1) / len(ds), label=split, color=sns_colors[split])
    for a_ in ax:
        for v in (30, 50):
            a_.axvline(v, ls="--", c="k", lw=0.8)
        a_.set_xlabel("distance from GPS point to nearest reference-route arc (m)")
        a_.legend()
    ax[0].set_ylabel("density"); ax[0].set_title("GPS noise relative to reference route (last bin = 60 m+)")
    ax[1].set_xscale("log"); ax[1].set_xlim(0.1, 2000)
    ax[1].set_ylabel("cumulative fraction"); ax[1].set_title("CDF, log scale (dashed: 30 m, 50 m)")
    _save(fig, "eda_gps_to_route_distance.png")

    # 5. network size and connectivity
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for spl, g in df.groupby("split"):
        ax[0].scatter(g.n_nodes, g.n_arcs, s=14, label=spl, color=sns_colors[spl])
        ax[1].scatter(g.n_nodes, g.n_weak_components, s=14, label=spl, color=sns_colors[spl])
        ax[2].scatter(g.n_nodes, g.largest_scc_frac, s=14, label=spl, color=sns_colors[spl])
    ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_xlabel("nodes"); ax[0].set_ylabel("arcs")
    ax[0].set_title("Road-network size per trip")
    ax[1].set_xscale("log"); ax[1].set_xlabel("nodes"); ax[1].set_ylabel("weakly connected components")
    ax[1].set_title("Disconnected components")
    ax[2].set_xscale("log"); ax[2].set_xlabel("nodes"); ax[2].set_ylabel("fraction of nodes in largest SCC")
    ax[2].set_title("Strong connectivity")
    for a_ in ax:
        a_.legend()
    _save(fig, "eda_network_connectivity.png")

    # 6. arc length
    al = np.concatenate(pool["arc_len"])
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.hist(np.clip(al, 0.1, None), bins=np.logspace(-1, 3.5, 80), color="#8172b2")
    ax.set_xscale("log"); ax.set_xlabel("arc (road segment) length (m)"); ax.set_ylabel("count")
    ax.set_title(f"Road segment lengths (median {np.median(al):.1f} m, n={len(al):,})")
    _save(fig, "eda_arc_length.png")

    # 7. data-quality checks
    checks = {
        "NaN values": df.nan_values.sum(), "invalid coordinates": df.invalid_coords.sum(),
        "duplicate timestamps": df.duplicate_timestamps.sum(),
        "non-monotonic timestamps": df.non_monotonic_timestamps.sum(),
        "gaps > 60 s": df.gaps_gt_60s.sum(), "speed jumps > 200 km/h": df.speed_jumps_gt_200kmh.sum(),
        "route breaks": df.route_breaks.sum(), "invalid route arc ids": df.route_invalid_ids.sum(),
        "isolated nodes": df.isolated_nodes.sum(), "zero-length arcs": df.zero_length_arcs.sum(),
    }
    fig, ax = plt.subplots(figsize=(8, 4.5))
    names, vals = list(checks), [int(v) for v in checks.values()]
    bars = ax.barh(names, vals, color=["#55a868" if v == 0 else "#c44e52" for v in vals])
    ax.bar_label(bars, labels=[f"{v:,}" for v in vals], padding=3)
    ax.set_xscale("symlog"); ax.invert_yaxis(); ax.set_xlabel("count across train + student_test")
    ax.set_title("Missing-value and data-quality checks")
    _save(fig, "eda_data_quality.png")


def plot_trip_grid(trips, name, title):
    n = len(trips)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.6 * rows))
    for ax, trip in zip(np.atleast_1d(axes).ravel(), trips):
        plot_trip_axes(ax, trip)
        ax.set_title(f"{trip.split}/{trip.trip_id}  ({trip.n_points} pts, {len(trip.route)} arcs)", fontsize=9)
    for ax in np.atleast_1d(axes).ravel()[n:]:
        ax.axis("off")
    h, l = np.atleast_1d(axes).ravel()[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(FIG / name, dpi=150)
    plt.close(fig)


def plot_anomaly_distance(trips, name):
    fig, axes = plt.subplots(len(trips), 1, figsize=(10, 2.4 * len(trips)), squeeze=False)
    for ax, trip in zip(axes[:, 0], trips):
        proj = LocalProjector.for_points(trip.track.lon, trip.track.lat)
        xy = proj.to_xy(trip.track.lon.values, trip.track.lat.values)
        nxy = proj.to_xy(trip.nodes[:, 0], trip.nodes[:, 1])
        d = min_distance_to_segments(xy, nxy[trip.arcs[trip.route, 0]], nxy[trip.arcs[trip.route, 1]])
        ax.plot(trip.track.t.values - trip.track.t.iat[0], np.clip(d, 0.1, None), lw=0.7)
        ax.axhline(FAR_M, ls="--", c="r", lw=0.8, label=f"{FAR_M:.0f} m")
        ax.set_yscale("log"); ax.set_ylim(0.1, None); ax.set_ylabel("dist (m)")
        ax.set_title(f"{trip.split}/{trip.trip_id}: GPS distance to reference route over time", fontsize=9)
    axes[-1, 0].set_xlabel("time since start (s)")
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=150)
    plt.close(fig)


def main():
    for p in (FIG, LOG, MAPS):
        p.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    inv = pd.concat([file_inventory(d, s) for s, d in SPLITS.items()], ignore_index=True)
    inv.to_csv(LOG / "eda_file_inventory.csv", index=False)
    inv_sum = inv.groupby(["split", "file"]).agg(trips=("trip", "nunique"), all_exist=("exists", "all"),
                                                 total_MB=("bytes", lambda b: round(b.sum() / 1e6, 2)),
                                                 rows_median=("rows", "median"), rows_max=("rows", "max"))
    print("File inventory:\n", inv_sum.to_string(), "\n")

    pool = {"dt": [], "speed": [], "arc_len": [], "d_route": []}
    rows = []
    for split, d in SPLITS.items():
        for trip in iter_trips(d, split=split):
            rows.append(trip_stats(trip, pool))
            print(f"  {split}/{trip.trip_id}: {trip.n_points} pts, {len(trip.arcs)} arcs", end="\r")
    df = pd.DataFrame(rows)
    df.to_csv(LOG / "eda_trip_summary.csv", index=False)
    summary = df.groupby("split").describe().T
    summary.to_csv(LOG / "eda_split_summary.csv")

    anomalies, counts = flag_anomalies(df)
    anomalies.to_csv(LOG / "eda_anomalies.csv", index=False)
    print("\nAnomaly rule counts (trips):\n", counts.to_string(), "\n")

    key = ["n_points", "duration_s", "frac_dt_eq_1", "max_gap_s", "path_length_km", "speed_median_kmh",
           "n_nodes", "n_arcs", "n_weak_components", "largest_scc_frac", "two_way_frac", "route_arcs",
           "route_breaks", "points_per_route_arc", "gps_route_dist_median_m", "gps_route_dist_p95_m",
           "frac_points_gt_50m"]
    print(df.groupby("split")[key].median().T.to_string(), "\n")

    make_plots(df, pool, set(anomalies.trip))

    # typical trips (closest to median size) and the worst anomalies
    tr = df[df.split == "train"].copy()
    typical = tr.iloc[(tr.n_points - tr.n_points.median()).abs().argsort()[:3]].trip.tolist()
    st = df[df.split == "student_test"]
    typical_st = st.iloc[(st.n_points - st.n_points.median()).abs().argsort()[:3]].trip.tolist()
    grid = [load_trip(RAW / "train" / t, "train") for t in typical] + \
           [load_trip(RAW / "student_test" / t, "student_test") for t in typical_st]
    plot_trip_grid(grid, "eda_trajectories_typical.png", "Typical trips: GPS vs reference route vs road network")

    worst = df.sort_values("frac_points_gt_50m", ascending=False).head(3)
    worst_trips = [load_trip(RAW / r.split / r.trip, r.split) for r in worst.itertuples()]
    plot_trip_grid(worst_trips, "eda_trajectories_anomalies.png",
                   "Trips where GPS strays far from the reference route")
    plot_anomaly_distance(worst_trips, "eda_anomaly_distance_over_time.png")

    # Folium maps: one typical trip and the worst anomaly
    m_trip = grid[0]
    trip_map(m_trip, title=f"EDA: train/{m_trip.trip_id}").save(MAPS / f"eda_train_{m_trip.trip_id}.html")
    w = worst_trips[0]
    trip_map(w, title=f"EDA anomaly: {w.split}/{w.trip_id}").save(MAPS / f"eda_anomaly_{w.split}_{w.trip_id}.html")

    print(f"Typical trips plotted: {typical} / {typical_st}")
    print(f"Worst anomaly trips: {worst.trip.tolist()}")
    print(f"Done in {time.time() - t0:.1f}s. Outputs in results/logs, results/figures, results/maps")


if __name__ == "__main__":
    main()
