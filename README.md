# PathSnap — Distributed Map-Matching & Route Reconstruction

UE24CS352A Machine Learning Hackathon (PES University).

PathSnap matches noisy GPS observations to a road network and reconstructs the route a vehicle actually drove:

1. **Candidate generation**: for every GPS point, find nearby road segments (arcs).
2. **Candidate classifier**: an ML model scores how likely each candidate is the true road.
3. **HMM + Viterbi**: combines classifier scores with road-network connectivity to decode a continuous route.
4. **Deployment**: a FastAPI server (Laptop 1) runs the pipeline, and a UI (Laptop 2) lets the evaluator pick a trip and view the result on a map.

## Results (10 student_test trips)

| Stage | Top-1 accuracy |
|---|---|
| Rule baseline: nearest road + heading (no learning) | 85.76% |
| LightGBM classifier (Optuna-tuned) | 89.37% |
| **Tuned LightGBM + HMM/Viterbi (deployed pipeline)** | **94.93%** |

- **Technical report:** [report/PathSnap_Technical_Report.pdf](report/PathSnap_Technical_Report.pdf), rebuilt with `python report/build_report.py`.
- **Notes for each phase:** `results/logs/phase*_notes.md`.
- **Plan and status:** [PLAN.md](PLAN.md).
- **Two-laptop demo:** [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md).

## Repository layout

```
configs/        config.yaml (all tunable settings)
src/pathsnap/   core library: io, geo, spatial, network, labels, candidates, features, models, hmm, metrics, pipeline, evaluation, viz
scripts/        runnable steps: build dataset, train models, tune, evaluate HMM, make maps, export artifacts
api/            FastAPI backend (Laptop 1)
ui/             Evaluator UI (Laptop 2)
models/         exported model + preprocessing + HMM config artifacts
results/        experiment logs (CSV), figures, Folium maps
report/         technical report sources and final PDF
data/raw/       raw dataset (not committed; see below)
data/eval/      evaluator trips (API laptop only; not committed)
tests/          unit tests (python -m pytest -q tests)
docs/           DEPLOYMENT.md (two-laptop setup)
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

Run the stages in this order. Each one reads the previous stage's outputs.

_(Commands will be filled in as each stage is implemented.)_

| Stage | Command |
|---|---|
| EDA (stats, figures, Folium maps) | `python -m scripts.eda` |
| Candidate radius/K sweep | `python -m scripts.build_dataset --sweep` |
| Build labels + candidate tables | `python -m scripts.build_dataset` |
| Build feature tables + analysis | `python -m scripts.build_features` |
| Unit tests | `python -m pytest -q tests` |
| Train & compare models | `python -m scripts.train_models` (one model: `--models lightgbm`; table/plots only: `--report`; smoke test: `--sample 0.03`) |
| Bayesian optimisation | `python -m scripts.tune_optuna` (run used `--trials 20 --timeout 2700`; refit/compare only: `--final-only`) |
| HMM evaluation | `python -m scripts.eval_hmm --preds results/models/lightgbm_tuned/pred_student_test.parquet --grid` (the predictions file is written by `tune_optuna`) |
| Folium maps | `python -m scripts.make_maps` (all student_test trips: `--all`) |
| Technical report PDF | `python report/build_report.py` |
| Export artifacts | `python -m scripts.export_artifacts` (re-check only: `--verify-only`; evaluator run-check, no scoring: `--check-eval`) |
| API (Laptop 1) | `python -m uvicorn api.main:app --host 0.0.0.0 --port 8000` (see docs/DEPLOYMENT.md) |
| UI (Laptop 2) | `streamlit run ui/app.py`, then set the API address in the sidebar |


## Team Members

| Name | SRN | Class |
|---|---|---|
| Aniket Sen | PES1UG24CS062  | 5A |
| Anirudh Ganesh Prabhu | PES1UG24CS063 | 5A |
| Aashlesh P | PES1UG24CS011 | 5A |
| Aarush Muralidhara | PES1UG24CS010 | 5A |
