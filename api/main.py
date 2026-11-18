"""PathSnap API (Laptop 1): runs the full prediction pipeline and scores it server-side.

Run from the repo root:
    uvicorn api.main:app --host 0.0.0.0 --port 8000

Endpoints
  GET  /health            model + data status
  GET  /trips             the evaluation trips available on this server (ids and sizes only)
  POST /predict           {"trip_id": "00000015"} -> predicted route, metrics, timings

Ground truth stays on this machine: /predict predicts from a route-free copy of the trip, then reads
the reference route only to compute metrics. Responses never contain reference arcs or geometry.
Every request is logged to this console (the evaluator is shown these logs).

Environment variables (optional)
  PATHSNAP_EVAL_DIR   folder of evaluation trips   (default: data/eval)
  PATHSNAP_MODEL_DIR  folder with exported model   (default: models)
"""
from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from pathsnap.evaluation import score_prediction
from pathsnap.io import list_trips, load_trip
from pathsnap.pipeline import load_artifacts, predict_trip

EVAL_DIR = Path(os.environ.get("PATHSNAP_EVAL_DIR", ROOT / "data" / "eval"))
MODEL_DIR = Path(os.environ.get("PATHSNAP_MODEL_DIR", ROOT / "models"))
MAX_GPS_POINTS_RETURNED = 4000   # GPS points are thinned for the map payload; metrics use all points

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("pathsnap.api")

app = FastAPI(title="PathSnap API", version="1.0",
              description="GPS map-matching: candidate generation + LightGBM + HMM/Viterbi.")
ART = load_artifacts(MODEL_DIR)
TRIPS = {d.name: d for d in list_trips(EVAL_DIR)}
log.info("loaded model %s | %d evaluation trips from %s", ART.config["model"], len(TRIPS), EVAL_DIR)


class PredictRequest(BaseModel):
    trip_id: str = Field(..., examples=["00000015"])
    use_hmm: bool = True


@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    client = request.client.host if request.client else "?"
    log.info("%s %s from %s -> %d (%.0f ms)", request.method, request.url.path, client, response.status_code,
             1000 * (time.perf_counter() - t0))
    return response


@app.get("/health")
def health():
    return dict(status="ok", model=ART.config["model"], features=len(ART.config["features"]),
                hmm=ART.config["hmm"], candidates=ART.config["candidates"], trips=len(TRIPS))


@app.get("/trips")
def trips():
    out = []
    for tid, d in sorted(TRIPS.items()):
        with open(d / f"{tid}.track", "rb") as fh:
            n = sum(1 for _ in fh)
        out.append(dict(trip_id=tid, gps_points=n))
    return dict(trips=out)


def _latlon_lines(trip, route):
    """Predicted route arcs -> list of polylines [[lat, lon], ...], split where the route has a gap."""
    lines, cur, prev = [], [], None
    for a in route:
        if a < 0:
            if cur:
                lines.append(cur)
            cur, prev = [], None
            continue
        u, v = trip.arcs[a]
        if prev is None or trip.arcs[prev, 1] != u:
            if cur:
                lines.append(cur)
            cur = [[float(trip.nodes[u, 1]), float(trip.nodes[u, 0])]]
        cur.append([float(trip.nodes[v, 1]), float(trip.nodes[v, 0])])
        prev = a
    if cur:
        lines.append(cur)
    return lines


@app.post("/predict")
def predict(req: PredictRequest):
    if req.trip_id not in TRIPS:
        raise HTTPException(status_code=404, detail=f"unknown trip_id {req.trip_id!r}; see GET /trips")
    d = TRIPS[req.trip_id]
    log.info("predict trip=%s use_hmm=%s: running pipeline", req.trip_id, req.use_hmm)

    trip = load_trip(d, split="eval", load_route=False)          # the model never sees the route
    out = predict_trip(trip, ART, use_hmm=req.use_hmm)

    t0 = time.perf_counter()
    full = load_trip(d, split="eval", load_route=True)          # server-side scoring only
    s_final = score_prediction(full, out["point_arc"], ART.config["labels"])
    s_clf = score_prediction(full, out["classifier_point_arc"], ART.config["labels"])
    score_s = time.perf_counter() - t0

    log.info("predict trip=%s: %d GPS points, %d candidates | Top-1 classifier %.2f%% -> final %.2f%% | "
             "continuity %.2f%% | route F1 %.3f | %.2fs", req.trip_id, out["n_points"], out["n_candidates"],
             100 * s_clf["top1_accuracy"], 100 * s_final["top1_accuracy"], 100 * s_final["route_continuity"],
             s_final["route_f1"], out["timings"]["total_s"])

    step = max(1, out["n_points"] // MAX_GPS_POINTS_RETURNED)
    gps = trip.track[["lat", "lon"]].to_numpy()[::step].round(6).tolist()
    return dict(
        trip_id=req.trip_id, method="classifier + HMM" if req.use_hmm else "classifier only",
        gps_points=out["n_points"], gps_returned_every=step, gps=gps,
        predicted_route=_latlon_lines(trip, out["route"]), route_arcs=len(out["route"]),
        route_gaps=out["route_gaps"], candidates=out["n_candidates"],
        points_without_candidates=out["points_without_candidates"], hmm_segments=out["hmm"]["segments"],
        metrics=dict(final=s_final, classifier_only=s_clf),
        timings={**{k: round(v, 3) for k, v in out["timings"].items()}, "scoring_s": round(score_s, 3)},
        model=ART.config["model"],
    )
