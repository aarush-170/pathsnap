"""Folium map builders (shared by EDA, evaluation and the API/UI)."""
from __future__ import annotations

import folium
import numpy as np
from scipy.spatial import cKDTree

from .geo import LocalProjector
from .io import Trip

COLORS = dict(gps="#1f77b4", network="#9e9e9e", route="#2ca02c", pred="#d62728", wrong="#ff7f0e",
              baseline="#9467bd")

# Base maps, first = default. OpenStreetMap's own tile servers reject browser requests without an HTTP
# Referer, which is what happens when a saved .html map is opened from disk (file://) or in browsers that
# strip referrers, so a provider without that restriction is the default; OSM stays selectable.
BASEMAPS = [
    ("Esri World Street Map",
     "https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
     "Tiles &copy; Esri &mdash; Source: Esri, HERE, Garmin, USGS, NGA, EPA, USDA, NPS"),
    ("CARTO Light", "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png",
     "&copy; OpenStreetMap contributors &copy; CARTO"),
    ("OpenStreetMap", "https://tile.openstreetmap.org/{z}/{x}/{y}.png", "&copy; OpenStreetMap contributors"),
]


def base_map() -> folium.Map:
    """A folium.Map with the BASEMAPS as switchable base layers (Esri first, shown by default)."""
    m = folium.Map(tiles=None, control_scale=True)
    for i, (name, url, attr) in enumerate(BASEMAPS):
        folium.TileLayer(tiles=url, attr=attr, name=name, max_zoom=19, show=(i == 0)).add_to(m)
    return m


def nearby_arc_ids(trip: Trip, buffer_m: float = 150.0) -> np.ndarray:
    """Arcs with at least one endpoint within buffer_m of any GPS point."""
    proj = LocalProjector.for_points(trip.track.lon, trip.track.lat)
    gps = proj.to_xy(trip.track.lon.values, trip.track.lat.values)
    nxy = proj.to_xy(trip.nodes[:, 0], trip.nodes[:, 1])
    d, _ = cKDTree(gps).query(nxy, distance_upper_bound=buffer_m)
    near = np.isfinite(d)
    return np.where(near[trip.arcs[:, 0]] | near[trip.arcs[:, 1]])[0]


def _arc_latlon(trip: Trip, arc_ids):
    a = trip.arcs[arc_ids]
    p, q = trip.nodes[a[:, 0]], trip.nodes[a[:, 1]]
    return [[(y1, x1), (y2, x2)] for (x1, y1), (x2, y2) in zip(p, q)]


def _route_polyline(trip: Trip, arc_ids):
    """Chain consecutive arcs into polylines, starting a new one at each break (or -1 marker)."""
    lines, cur = [], []
    for i, a in enumerate(arc_ids):
        if a < 0:
            if cur:
                lines.append(cur)
            cur = []
            continue
        u, v = trip.arcs[a]
        if cur and (arc_ids[i - 1] < 0 or trip.arcs[arc_ids[i - 1], 1] != u):
            lines.append(cur)
            cur = []
        if not cur:
            cur.append((trip.nodes[u, 1], trip.nodes[u, 0]))
        cur.append((trip.nodes[v, 1], trip.nodes[v, 0]))
    if cur:
        lines.append(cur)
    return lines


def _legend(items: dict[str, str]) -> folium.Element:
    rows = "".join(
        f'<div><span style="display:inline-block;width:14px;height:4px;background:{c};margin-right:6px;'
        f'vertical-align:middle"></span>{name}</div>' for name, c in items.items())
    html = (f'<div style="position:fixed;bottom:24px;left:24px;z-index:9999;background:white;padding:8px 10px;'
            f'border:1px solid #999;border-radius:4px;font:12px sans-serif">{rows}</div>')
    return folium.Element(html)


def trip_map(trip: Trip, predicted_route=None, wrong_points=None, buffer_m: float = 150.0,
             max_gps_markers: int = 3000, title: str | None = None, baseline_route=None,
             baseline_label: str = "Classifier-only route", predicted_label: str = "Predicted route",
             stats_html: str | None = None) -> folium.Map:
    lat, lon = trip.track.lat.values, trip.track.lon.values
    m = base_map()
    legend = {}

    net = folium.FeatureGroup(name="Road network (nearby)", show=True)
    for seg in _arc_latlon(trip, nearby_arc_ids(trip, buffer_m)):
        folium.PolyLine(seg, color=COLORS["network"], weight=1, opacity=0.6).add_to(net)
    net.add_to(m)
    legend["Road network"] = COLORS["network"]

    if trip.route is not None:
        fg = folium.FeatureGroup(name="Reference route", show=True)
        for line in _route_polyline(trip, trip.route):
            folium.PolyLine(line, color=COLORS["route"], weight=5, opacity=0.8).add_to(fg)
        fg.add_to(m)
        legend["Reference route"] = COLORS["route"]

    if baseline_route is not None:
        fg = folium.FeatureGroup(name=baseline_label, show=False)
        for line in _route_polyline(trip, np.asarray(baseline_route)):
            folium.PolyLine(line, color=COLORS["baseline"], weight=3, opacity=0.9, dash_array="2 6").add_to(fg)
        fg.add_to(m)
        legend[baseline_label + " (toggle on)"] = COLORS["baseline"]

    if predicted_route is not None:
        fg = folium.FeatureGroup(name=predicted_label, show=True)
        for line in _route_polyline(trip, np.asarray(predicted_route)):
            folium.PolyLine(line, color=COLORS["pred"], weight=3, opacity=0.9, dash_array="6").add_to(fg)
        fg.add_to(m)
        legend[predicted_label] = COLORS["pred"]

    gps = folium.FeatureGroup(name="GPS observations", show=True)
    step = max(1, len(lat) // max_gps_markers)
    for i in range(0, len(lat), step):
        folium.CircleMarker((lat[i], lon[i]), radius=1.5, color=COLORS["gps"], fill=True, fill_opacity=0.45,
                            weight=0, tooltip=f"#{i}  t={trip.track.t.iat[i]:.0f}s").add_to(gps)
    gps.add_to(m)
    legend["GPS observation" + (f" (every {step})" if step > 1 else "")] = COLORS["gps"]

    if wrong_points is not None and len(wrong_points):
        fg = folium.FeatureGroup(name="Mismatched points", show=True)
        for i in wrong_points:
            folium.CircleMarker((lat[i], lon[i]), radius=4, color=COLORS["wrong"], fill=True,
                                tooltip=f"mismatch #{i}").add_to(fg)
        fg.add_to(m)
        legend["Mismatched point"] = COLORS["wrong"]

    folium.Marker((lat[0], lon[0]), tooltip="Start", icon=folium.Icon(color="green", icon="play")).add_to(m)
    folium.Marker((lat[-1], lon[-1]), tooltip="End", icon=folium.Icon(color="red", icon="stop")).add_to(m)
    m.fit_bounds([[lat.min(), lon.min()], [lat.max(), lon.max()]])
    folium.LayerControl(collapsed=False).add_to(m)
    m.get_root().html.add_child(_legend(legend))
    if title or stats_html:
        m.get_root().html.add_child(folium.Element(
            f'<div style="position:fixed;top:10px;left:60px;z-index:9999;background:white;padding:6px 10px;'
            f'border:1px solid #999;border-radius:4px;font:13px sans-serif">'
            f'{f"<b>{title}</b>" if title else ""}{f"<br>{stats_html}" if stats_html else ""}</div>'))
    return m
