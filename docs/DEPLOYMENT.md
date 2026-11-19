# Two-laptop deployment: API on Laptop 1, UI on Laptop 2

| Laptop | Runs | Needs |
|---|---|---|
| **Laptop 1 (API)** | `uvicorn api.main:app` | The repo, the Python packages, `models/` (in the repo) and the **evaluator trips in `data/eval/`**. No training data is needed. |
| **Laptop 2 (UI)** | `streamlit run ui/app.py` | The repo and the Python packages. **No data and no model**: the UI only talks to the API. |

Both laptops must be on the **same network**. A phone hotspot is the most reliable choice, because campus or college Wi-Fi often blocks laptops from talking to each other ("client isolation").

## 0. One-time setup (both laptops)
```bash
git clone https://github.com/aarush-170/pathsnap.git
cd pathsnap
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 1. Laptop 1: start the API
1. Put the evaluator trips at `data/eval/<trip_id>/<trip_id>.{track,nodes,arcs,route}`, one folder per trip, 10 folders.
2. Find this laptop's IP address:
   ```bash
   ipconfig
   ```
   Use the **IPv4 Address** of the Wi-Fi adapter, for example `192.168.43.120`.
3. Start the API:
   ```bash
   python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
   ```
   You should see `loaded model LightGBM (Optuna-tuned, Phase 5 trial 8) | 10 evaluation trips`. **Keep this window visible.** It is the API log the evaluator wants to see. Each request prints the trip, Top-1 before and after the HMM, continuity, route F1 and the time taken.
4. The first time, Windows asks whether to allow Python through the firewall. Tick **Private networks** and allow it. If you missed the prompt, run this once in an **Administrator** PowerShell:
   ```bash
   netsh advfirewall firewall add rule name="PathSnap API 8000" dir=in action=allow protocol=TCP localport=8000
   ```
5. Check it works on Laptop 1 itself: open `http://localhost:8000/health` in a browser. The interactive API docs are at `http://localhost:8000/docs`.

## 2. Laptop 2: start the UI
```bash
streamlit run ui/app.py
```
The browser opens at `http://localhost:8501`. In the sidebar, set **API address** to `http://<Laptop-1-IP>:8000`, for example `http://192.168.43.120:8000`. It turns green ("Connected: 10 trips").

Or set the address before starting:
```bash
set "PATHSNAP_API=http://192.168.43.120:8000"
```
```bash
streamlit run ui/app.py
```

## 3. Demo flow (what the evaluator does)
1. The evaluator picks any of the 10 trips in the dropdown and clicks **Run prediction**.
2. The UI sends only the trip id to Laptop 1.
3. Laptop 1 runs the full pipeline (candidates → features → LightGBM → HMM → route) on a route-free copy of the trip. It then scores the result against its own reference route and logs the request.
4. The UI shows Top-1 accuracy, the gain over classifier-only, correct points, route continuity, route F1, the processing time and the predicted route on a map.
5. Point to Laptop 1's console to show the request log.
6. Optional: switch off "Use HMM / Viterbi decoding" in the sidebar to show the classifier-only result for the same trip.

The UI never receives the reference route: the API returns only the predicted route and the metrics.

## 4. If it does not connect
| Symptom | Fix |
|---|---|
| The UI says "Cannot reach the API" | Check the IP (run `ipconfig` again, since it changes when you switch networks). Check the API window is still running. Check the firewall rule. |
| It works on Laptop 1 (`localhost`) but not from Laptop 2 | The network blocks device-to-device traffic. Put both laptops on a **phone hotspot**. |
| Laptop 2's browser can't open `http://<IP>:8000/health` (connection **times out**) even on the same hotspot | Windows treats the hotspot as a **Public** network, and Python is only allowed through the firewall on Private networks. On Laptop 1: *Settings → Network & internet → Wi-Fi → (the hotspot) → Network profile type → **Private***. This fixed our rehearsal. |
| "unknown trip_id" | The `data/eval/` folder names must be the 8-digit trip ids. |
| Slow first request | Normal. The first request loads trips from disk; trips take 0.4–3.5 s each. |

## 5. Rehearse beforehand
Test on the **same network you will use at the evaluation**, and run all 10 trips once end to end (this run-check is what `python -m scripts.export_artifacts --check-eval` does on a single machine).
To test without the evaluator trips, point the API at student_test:
```bash
set "PATHSNAP_EVAL_DIR=data/raw/student_test"
```
```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```
