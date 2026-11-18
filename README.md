


# PathSnap — Distributed Map-Matching & Route Reconstruction

UE24CS352A Machine Learning Hackathon (PES University).

PathSnap matches noisy GPS observations to a road network and reconstructs the route a vehicle actually drove:

1. **Candidate generation**: for every GPS point, find nearby road segments (arcs).
2. **Candidate classifier**: an ML model scores how likely each candidate is the true road.
3. **HMM + Viterbi**: combines classifier scores with road-network connectivity to decode a continuous route.
4. **Deployment**: a FastAPI server (Laptop 1) runs the pipeline, and a UI (Laptop 2) lets the evaluator pick a trip and view the result on a map.

See [PLAN.md](PLAN.md) for the full project plan and status.

## Repository layout

```
configs/        config.yaml (all tunable settings)
src/pathsnap/   core library: io, geo, network, candidates, labels, features, hmm, metrics, pipeline
scripts/        runnable steps: build dataset, train models, tune, evaluate HMM, make maps, export artifacts
notebooks/      EDA and analysis notebooks
api/            FastAPI backend (Laptop 1)
ui/             Evaluator UI (Laptop 2)
models/         exported model + preprocessing + HMM config artifacts
results/        experiment logs (CSV), figures, Folium maps
report/         technical report sources and final PDF
data/raw/       raw dataset (not committed; see below)
data/eval/      evaluator trips (API laptop only; not committed)
tests/          unit tests
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

Place the provided dataset like this:

```
data/raw/train/<trip_id>/<trip_id>.{track,nodes,arcs,route}
data/raw/student_test/<trip_id>/<trip_id>.{track,nodes,arcs,route}
```

## Running

_(Commands will be filled in as each stage is implemented.)_

| Stage | Command |
|---|---|
| EDA (stats, figures, Folium maps) | `python -m scripts.eda` |
| Candidate radius/K sweep | `python -m scripts.build_dataset --sweep` |
| Build labels + candidate tables | `python -m scripts.build_dataset` |
| Unit tests | `python -m pytest -q tests` |
| Train & compare models | `python -m scripts.train_models` |
| Bayesian optimisation | `python -m scripts.tune_optuna` |
| HMM evaluation | `python -m scripts.eval_hmm` |
| Folium maps | `python -m scripts.make_maps` |
| Export artifacts | `python -m scripts.export_artifacts` |
| API (Laptop 1) | `uvicorn api.main:app --host 0.0.0.0 --port 8000` |
| UI (Laptop 2) | `streamlit run ui/app.py` |


## Team Members

| Name | SRN | Class |
|---|---|---|
| Aniket Sen | PES1UG24CS062  | 5A |
| Anirudh Ganesh Prabhu | PES1UG24CS063 | 5A |
| Aashlesh P | PES1UG24CS011 | 5A |
| Aarush Muralidhara | PES1UG24CS010 | 5A |
