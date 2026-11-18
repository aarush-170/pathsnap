"""PathSnap evaluator UI (Laptop 2).

Run:  streamlit run ui/app.py
Set the API address in the sidebar (Laptop 1's IP, e.g. http://192.168.1.23:8000), or with the
PATHSNAP_API environment variable. The UI only talks to the API: it never loads data or models, and it
never sees the reference route; it shows the predicted route and the metrics the API computed.
"""
from __future__ import annotations

import os

import folium
import requests
import streamlit as st
from streamlit_folium import st_folium

st.set_page_config(page_title="PathSnap", page_icon="🗺️", layout="wide")

DEFAULT_API = os.environ.get("PATHSNAP_API", "http://localhost:8000")

# Base maps, first = default. OpenStreetMap's own servers block browsers that send no Referer (some
# browsers and privacy settings strip it), so Esri is the default and OSM stays selectable.
BASEMAPS = [
    ("Esri World Street Map",
     "https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
     "Tiles &copy; Esri &mdash; Source: Esri, HERE, Garmin, USGS, NGA, EPA, USDA, NPS"),
    ("CARTO Light", "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png",
     "&copy; OpenStreetMap contributors &copy; CARTO"),
    ("OpenStreetMap", "https://tile.openstreetmap.org/{z}/{x}/{y}.png", "&copy; OpenStreetMap contributors"),
]

with st.sidebar:
    st.header("Connection")
    api = st.text_input("API address (Laptop 1)", value=st.session_state.get("api", DEFAULT_API)).rstrip("/")
    st.session_state["api"] = api
    try:
        h = requests.get(f"{api}/health", timeout=5).json()
        st.success(f"Connected: {h['trips']} trips")
        st.caption(f"Model: {h['model']}")
        st.caption(f"Candidates: {h['candidates']['radius_m']:.0f} m, top {h['candidates']['max_k']} | "
                   f"HMM β={h['hmm']['beta']}, α={h['hmm']['alpha']}")
        connected = True
    except Exception as e:  # noqa: BLE001
        st.error(f"Cannot reach the API at {api}")
        st.caption(str(e)[:200])
        connected = False
    use_hmm = st.toggle("Use HMM / Viterbi decoding", value=True,
                        help="Off = classifier-only predictions (for comparison)")

st.title("PathSnap: GPS route reconstruction")
st.caption("Pick an evaluation trip. The API on Laptop 1 runs candidate generation, the LightGBM classifier "
           "and HMM/Viterbi decoding, then scores the result against its own reference data.")

if not connected:
    st.info("Enter the API address of Laptop 1 in the sidebar to begin.")
    st.stop()


@st.cache_data(ttl=60)
def get_trips(api_url):
    return requests.get(f"{api_url}/trips", timeout=10).json()["trips"]


trips = get_trips(api)
labels = {t["trip_id"]: f"Trip {t['trip_id']}  ({t['gps_points']:,} GPS points)" for t in trips}
c1, c2 = st.columns([3, 1])
trip_id = c1.selectbox("Evaluation trip", list(labels), format_func=labels.get)
run = c2.button("Run prediction", type="primary", use_container_width=True)

if run:
    with st.spinner(f"Laptop 1 is processing trip {trip_id}…"):
        try:
            r = requests.post(f"{api}/predict", json={"trip_id": trip_id, "use_hmm": use_hmm}, timeout=120)
            r.raise_for_status()
            st.session_state["result"] = r.json()
        except Exception as e:  # noqa: BLE001
            st.error(f"Prediction failed: {e}")

res = st.session_state.get("result")
if res:
    m, c = res["metrics"]["final"], res["metrics"]["classifier_only"]
    st.subheader(f"Trip {res['trip_id']}: {res['method']}")
    k = st.columns(5)
    k[0].metric("Top-1 accuracy", f"{m['top1_accuracy']:.1%}",
                f"{100 * (m['top1_accuracy'] - c['top1_accuracy']):+.1f} pts vs classifier only"
                if res["method"] != "classifier only" else None)
    k[1].metric(f"Correct points (of {m['labelled_points']:,})", f"{m['correct_points']:,}")
    k[2].metric("Route continuity", f"{m['route_continuity']:.1%}")
    k[3].metric("Route F1", f"{m['route_f1']:.3f}")
    k[4].metric("Processing time", f"{res['timings']['total_s']:.2f} s")

    gps = res["gps"]
    lats, lons = [p[0] for p in gps], [p[1] for p in gps]
    fmap = folium.Map(tiles=None, control_scale=True)
    for i, (name, url, attr) in enumerate(BASEMAPS):
        folium.TileLayer(tiles=url, attr=attr, name=name, max_zoom=19, show=(i == 0)).add_to(fmap)
    fg_gps = folium.FeatureGroup(name="GPS observations")
    for la, lo in gps:
        folium.CircleMarker((la, lo), radius=1.5, color="#1f77b4", fill=True, fill_opacity=0.5, weight=0).add_to(fg_gps)
    fg_gps.add_to(fmap)
    fg_route = folium.FeatureGroup(name="Predicted route")
    for line in res["predicted_route"]:
        folium.PolyLine(line, color="#d62728", weight=4, opacity=0.85).add_to(fg_route)
    fg_route.add_to(fmap)
    folium.Marker(gps[0], tooltip="Start", icon=folium.Icon(color="green", icon="play")).add_to(fmap)
    folium.Marker(gps[-1], tooltip="End", icon=folium.Icon(color="red", icon="stop")).add_to(fmap)
    fmap.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]])
    folium.LayerControl(collapsed=False).add_to(fmap)
    fmap.get_root().html.add_child(folium.Element(
        '<div style="position:fixed;bottom:24px;left:24px;z-index:9999;background:white;padding:6px 10px;'
        'border:1px solid #999;border-radius:4px;font:12px sans-serif">'
        '<div><span style="display:inline-block;width:14px;height:4px;background:#d62728;margin-right:6px"></span>'
        'Predicted route</div><div><span style="display:inline-block;width:8px;height:8px;border-radius:4px;'
        'background:#1f77b4;margin-right:10px"></span>GPS observation</div></div>'))
    st_folium(fmap, height=560, use_container_width=True, returned_objects=[])

    with st.expander("Details"):
        a, b = st.columns(2)
        a.markdown("**Classifier only vs final**")
        a.table({"metric": ["Top-1", "Continuity", "Route precision", "Route recall", "Route F1"],
                 "classifier only": [f"{c['top1_accuracy']:.2%}", f"{c['route_continuity']:.2%}",
                                     f"{c['route_precision']:.3f}", f"{c['route_recall']:.3f}", f"{c['route_f1']:.3f}"],
                 "final": [f"{m['top1_accuracy']:.2%}", f"{m['route_continuity']:.2%}", f"{m['route_precision']:.3f}",
                           f"{m['route_recall']:.3f}", f"{m['route_f1']:.3f}"]})
        b.markdown("**Pipeline**")
        b.table({"item": ["GPS points", "candidate rows", "points without candidates", "HMM segments",
                          "route arcs", "route gaps"],
                 "value": [res["gps_points"], res["candidates"], res["points_without_candidates"],
                           res["hmm_segments"], res["route_arcs"], res["route_gaps"]]})
        b.markdown("**Timings (s)**")
        b.table({"stage": list(res["timings"]), "seconds": list(res["timings"].values())})
