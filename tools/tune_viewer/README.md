# CSIRO Ray Tune Viewer (local)

A local FastAPI + React/Vite web app to explore Ray Tune experiment results under a `ray_results/` directory (e.g. `/mnt/csiro_nfs/ray_results`).

It focuses on the **file outputs you already produce**:

- `params.json` (sampled hyperparameters)
- `result.json` (JSONL from `ray.air.session.report()`; metrics over epochs)
- `run/logs/train.yaml`, `run/logs/train.log`, checkpoints and other artifacts

## Quick start (dev)

### Backend (FastAPI)

```bash
cd tools/tune_viewer/backend

python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt

export CSIRO_TUNE_VIEWER_RESULTS_ROOT="/path/to/ray_results"
uvicorn tune_viewer.api.main:app --reload --host 0.0.0.0 --port 8008
```

### Frontend (React/Vite)

```bash
cd tools/tune_viewer/frontend
npm install
VITE_TUNE_VIEWER_API_URL="http://localhost:8008" npm run dev
```

Then open the shown local URL (usually `http://localhost:5173`).

## How to use (UI)

1) **Start the backend** (FastAPI, default `:8008`) and **start the frontend** (Vite, default `:5173`) using the commands above.

2) Open the UI:

- Home: `http://localhost:5173/`
- Analysis: `http://localhost:5173/analysis`

3) Point the backend to your results root:

- Ensure `CSIRO_TUNE_VIEWER_RESULTS_ROOT` is set to the directory that contains experiment folders (e.g. `/mnt/csiro_nfs/ray_results`).
- The UI reads everything via the backend; it does not access the filesystem directly.

## What data it reads (per trial)

- `params.json`: sampled hyperparameters (dotted paths)
- `result.json`: JSONL metrics/time series written by Ray (`ray.air.session.report`)
- Optional “details” (if present):
  - `run/logs/train.yaml`: resolved training config for that trial
  - `run/logs/train.log`: training log
  - `run/checkpoints/last.ckpt`: checkpoint path (for locating artifacts)

## Backend endpoints (for debugging)

- `GET /api/health`: basic config/paths sanity check
- (More endpoints will be added as the viewer is implemented.)

## Configuration

- **`CSIRO_TUNE_VIEWER_RESULTS_ROOT`** (backend): root directory that contains experiment folders (defaults to `/mnt/csiro_nfs/ray_results`).
- **`CSIRO_TUNE_VIEWER_POLL_SECONDS`** (backend): default polling hint for the UI (defaults to `8`).
- **`VITE_TUNE_VIEWER_API_URL`** (frontend): FastAPI base URL (defaults to `http://localhost:8008`).

## Notes

- The backend is read-only and implements strict path safety: file reads are restricted to `CSIRO_TUNE_VIEWER_RESULTS_ROOT`.
- The UI supports **live refresh** by polling summary endpoints; detailed timeseries is fetched on-demand per trial.

## Troubleshooting

- **Frontend can’t reach backend**: verify `VITE_TUNE_VIEWER_API_URL` matches where `uvicorn` is listening (and that CORS is allowed).
- **No experiments listed**: verify `CSIRO_TUNE_VIEWER_RESULTS_ROOT` points to the directory that directly contains folders like `tune-mlp-v6/`.
