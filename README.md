# OrbitalGuard (OGB)

> **IBM Bob AI Builders Challenge — August 2025 | Theme: Advance Space Exploration with AI**

OrbitalGuard (OGB) is a JARVIS-inspired decision-support system for space operators. It analyses spacecraft-camera imagery for potential debris and known spacecraft, combines that with orbital data when available, calculates close-approach risk deterministically, and uses an AI copilot to explain the situation — the human operator remains in control of all decisions.

**Core loop:** See → Analyse → Contextualise → Prioritise → Explain

> ⚠️ OGB is decision-support software, not autonomous spacecraft control. It can detect, analyse, track, prioritise, explain, and recommend. It cannot control spacecraft, execute manoeuvres, send commands, or change trajectories. The human operator makes all decisions.

---

## Table of Contents

1. [Architecture](#architecture)
2. [Dataset](#dataset)
3. [Model Training](#model-training)
4. [Evaluation Results](#evaluation-results)
5. [Installation](#installation)
6. [Running the App](#running-the-app)
7. [API Reference](#api-reference)
8. [Risk Engine](#risk-engine)
9. [AI Copilot](#ai-copilot)
10. [Limitations](#limitations)
11. [How IBM Bob Was Used](#how-ibm-bob-was-used)
12. [Deployment](#deployment)
13. [License & Credits](#license--credits)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  OGB — OrbitalGuard                                         │
│                                                             │
│  Frontend  (Next.js + React + TypeScript)                   │
│    ├── Mission Dashboard  (CesiumJS 3D Earth — V2)         │
│    ├── Camera Analysis    (image/video upload + bbox)       │
│    ├── Threat Centre      (CRITICAL / HIGH / MEDIUM / LOW)  │
│    └── AI Copilot         (chat panel)                      │
│                                                             │
│  Backend   (FastAPI)                                        │
│    ├── POST /api/v1/detect   — YOLOv8n inference            │
│    ├── POST /api/v1/copilot  — AI Copilot chat              │
│    └── GET  /api/v1/health   — liveness probe               │
│                                                             │
│  ML        (Ultralytics YOLOv8n)                            │
│    └── Trained on Space Debris v2 dataset (11 classes)      │
│                                                             │
│  Orbital   (python-sgp4 + NumPy + SciPy)  — V2             │
│    └── TLE → SGP4 → TCA / d_min / v_rel / risk score       │
└─────────────────────────────────────────────────────────────┘
```

**Important constraint:** Visual detection and orbital data are kept strictly separate. A 2D image cannot yield true 3D distance, velocity, or collision probability. Visual–orbital correlation is a V3 stretch feature only.

---

## Dataset

| Property | Value |
|---|---|
| **Source** | Space Debris v2 by [woah-noah](https://universe.roboflow.com/woah-noah/space-debris-mugw2/dataset/2) on Roboflow Universe |
| **License** | CC BY 4.0 — credit: woah-noah / Roboflow |
| **Total images** | 2,467 |
| **Train / Valid / Test** | 2,105 / 239 / 123 |
| **Classes** | 11 (see below) |
| **Image size** | 640×640 (stretch resize) |
| **Pre-processing** | Auto-orientation (EXIF stripped), stretch to 640×640 |
| **Augmentation in export** | 50% horizontal flip + 50% vertical flip, 3 versions per source image |

### Class list (index order matches `data.yaml`)

| Index | Class name |
|---|---|
| 0 | `cheops` |
| 1 | `debris` |
| 2 | `double_start` |
| 3 | `earth_observation_sat_1` |
| 4 | `lisa_pathfinder` |
| 5 | `proba_2` |
| 6 | `proba_3_csc` |
| 7 | `proba_3_ocs` |
| 8 | `smart_1` |
| 9 | `soho` |
| 10 | `xmm_newton` |

> Note: class 2 is `double_start` (not `double_star`) — matches the exact spelling in `data.yaml`.

---

## Model Training

- **Model:** Ultralytics YOLOv8n (nano — CPU-optimised for free-tier deployment)
- **Image size:** 640×640
- **Epochs:** 50
- **Batch size:** 16
- **Optimizer:** AdamW
- **Extra flips disabled** — dataset already contains Roboflow-applied flips; adding more would over-augment.

### Train locally

```bash
cd /path/to/ogb
python ml/training/train.py --epochs 50 --batch 16 --device 0
# CPU fallback:
python ml/training/train.py --epochs 50 --batch 16 --device cpu
```

### Train on Kaggle / Google Colab (free T4 GPU)

Open `ml/training/ogb_train_colab.ipynb` in Colab, set runtime to T4 GPU, add your Roboflow API key, and run all cells. Weights are saved to Google Drive at the end.

After training, copy `best.pt` to `ml/weights/ogb_yolov8n.pt`.

---

## Evaluation Results

> Results below will be updated with actual numbers after the training run completes. The target is mAP@50 ≥ 0.85 — the actual achieved value is reported honestly here.

| Metric | Value |
|---|---|
| **mAP@50** | _TBD after training_ |
| **mAP@50-95** | _TBD_ |
| **Precision** | _TBD_ |
| **Recall** | _TBD_ |
| **F1** | _TBD_ |
| **CPU inference latency (mean)** | _TBD ms_ |

---

## Installation

### Backend

```bash
cd backend
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Unix:     source .venv/bin/activate
pip install -r requirements.txt
```

Create `backend/.env`:

```
GEMINI_API_KEY=your_key_here
```

### Frontend

```bash
cd frontend
npm install
```

---

## Running the App

```bash
# Backend (from repo root)
cd backend
uvicorn app.main:app --reload --port 8000

# Frontend (from repo root)
cd frontend
npm run dev
```

API docs available at: http://localhost:8000/docs

---

## API Reference

### `POST /api/v1/detect`

Upload a spacecraft camera image. Returns bounding boxes, class names, and confidence scores.

**Vision output only** — does not include distance, velocity, or collision probability.

```json
{
  "image_width": 640,
  "image_height": 640,
  "detections": [
    {
      "class_id": 1,
      "class_name": "debris",
      "confidence": 0.87,
      "bounding_box": { "x_center": 0.5, "y_center": 0.4, "width": 0.1, "height": 0.1 },
      "x1_px": 288, "y1_px": 224, "x2_px": 352, "y2_px": 288
    }
  ],
  "detection_count": 1,
  "inference_latency_ms": 45.2,
  "summary": "Detected 1 object(s): 1× debris. Note: visual detection only..."
}
```

### `POST /api/v1/copilot`

Send a message to the AI copilot. Optionally attach structured detection, orbital, or risk context.

```json
{
  "message": "What did OGB detect?",
  "detection_context": { "detection_count": 1, "summary": "1× debris detected." }
}
```

### `GET /api/v1/health`

Liveness probe. Returns `{ "status": "ok" }`.

---

## Risk Engine

_(V2 feature — orbital pipeline required)_

Risk score formula (0–1 project risk-priority score, not a formally validated collision probability):

```
Risk = min(1, (R / d_min) · log10(v_rel + 1) · exp(−Δt_epoch / 7))
```

| Symbol | Unit | Description |
|---|---|---|
| `R` | km | Hard-body radius |
| `d_min` | km | Minimum separation distance |
| `v_rel` | km/s | Relative velocity |
| `Δt_epoch` | days | TLE age |

| Score | Category |
|---|---|
| 0.00–0.24 | 🟢 LOW |
| 0.25–0.49 | 🟡 MEDIUM |
| 0.50–0.74 | 🟠 HIGH |
| 0.75–1.00 | 🔴 CRITICAL |

Thresholds are configurable in [`backend/app/core/config.py`](backend/app/core/config.py) — never hard-coded.

---

## AI Copilot

- Powered by Google Gemini (provider-agnostic — swap via `COPILOT_PROVIDER` in config)
- Receives **structured JSON only** from the vision and orbital pipelines — never raw images or TLEs
- Never calculates orbital mechanics or invents missing values
- If a field is null or absent from the context, it says so explicitly
- Explains detections, summarises threats, prioritises information
- Always makes clear that the human operator makes all decisions

---

## Limitations

- OGB is a **decision-support tool**, not an autonomous collision avoidance system.
- Visual detection provides class + confidence + bounding box only. 3D distance, velocity, and collision probability require orbital data.
- The risk score is a project priority indicator, not a formally validated collision probability.
- Visual–orbital correlation (confirming a detected object matches a tracked orbital object) is a V3 stretch feature — never assumed in MVP/V1.5/V2.
- YOLOv8n (nano) is chosen for CPU-optimised free-tier deployment; a larger model would improve accuracy at the cost of inference speed.
- TLE data degrades over time; the risk formula penalises stale epochs via the `exp(−Δt/7)` decay term.

---

## How IBM Bob Was Used

IBM Bob (the AI software engineer in this repo's IDE) contributed the following during this project:

1. **Project scaffolding** — generated the complete repository structure (`backend/`, `frontend/`, `ml/`, `tests/`, `deployment/`, `docs/`) from the spec.
2. **Backend skeleton** — authored the full FastAPI application: `main.py`, all API v1 routers (`detect`, `copilot`, `health`), Pydantic models (`DetectionResponse`, `CopilotRequest/Response`), and service modules (`detection_service.py`, `copilot_service.py`).
3. **Configuration module** — designed `backend/app/core/config.py` with all risk thresholds, class names, and model paths centralised and documented with units.
4. **Training script** — wrote `ml/training/train.py` including the pre-training sanity check, locked hyperparameters, conservative augmentation rationale, and auto-copy of best weights.
5. **Colab notebook** — generated `ml/training/ogb_train_colab.ipynb` for free T4 GPU training with dataset download, sanity check, training, evaluation, CPU latency benchmark, and Drive export cells.
6. **Risk engine tests** — authored comprehensive unit tests in `tests/unit/test_risk_engine.py` covering formula correctness, unit consistency, zero/near-zero separation, stale TLE decay, clamping, and all four risk category boundaries.
7. **API integration tests** — authored `tests/integration/test_api.py` covering health, detect (happy path, wrong type, empty file, service error), and copilot endpoints.
8. **`.gitignore`** — generated with appropriate exclusions for the dataset, model weights, Python artefacts, Node modules, and secrets.
9. **README** — wrote this document.
10. **Safety framing enforcement** — ensured all system prompt text, API documentation, and code comments consistently frame OGB as decision-support only, not autonomous control.

---

## Deployment

### Live URLs

| Service | URL |
|---|---|
| **Frontend** | _(deploy to Vercel — see steps below)_ |
| **Backend** | _(deploy to Render — see steps below)_ |

> ⚠️ **Cold-start warning:** Render's free tier spins down after 15 minutes of inactivity. The first request after an idle period takes **30–60 seconds** (Python startup + YOLOv8n model load). Hit `GET /api/v1/health` before your demo to pre-warm it. Paid Render/Railway tiers or Fly.io machines eliminate this entirely.

---

### Deploy the Backend (Render free tier)

1. Push this repo to GitHub.
2. Go to [render.com](https://render.com) → **New → Web Service** → connect your repo.
3. Render auto-detects `render.yaml` at the repo root and pre-fills all settings.
4. In the Render dashboard **Environment** tab, add:
   - `GEMINI_API_KEY` → your Google AI Studio key
   - `ALLOWED_ORIGINS` → your Vercel frontend URL (add after step below; you can update it)
5. Click **Deploy**.
6. Note the service URL, e.g. `https://ogb-backend.onrender.com`.

**Smoke-test the deployed backend:**
```bash
curl https://ogb-backend.onrender.com/api/v1/health
# → {"status":"ok"}
```

---

### Deploy the Frontend (Vercel free tier)

1. Go to [vercel.com](https://vercel.com) → **New Project** → import your GitHub repo.
2. Set the **Root Directory** to `frontend`.
3. In **Environment Variables**, add:
   - `NEXT_PUBLIC_API_URL` → `https://ogb-backend.onrender.com` (your Render URL)
4. Click **Deploy**.
5. Note your Vercel URL, e.g. `https://ogb.vercel.app`.

**Update backend CORS** — go back to Render, update `ALLOWED_ORIGINS` to your Vercel URL, and redeploy:
```
ALLOWED_ORIGINS=https://ogb.vercel.app,https://ogb-git-main-youruser.vercel.app
```

---

### Local development (unchanged)

```bash
# Backend
cd backend
uvicorn app.main:app --reload --port 8000

# Frontend (separate terminal)
cd frontend
npm run dev
```

No `ALLOWED_ORIGINS` needed locally — the backend defaults to `localhost:3000/3001`.

---

## License & Credits

- **Dataset:** [Space Debris v2](https://universe.roboflow.com/woah-noah/space-debris-mugw2/dataset/2) by woah-noah on Roboflow Universe. License: **CC BY 4.0**. Used with attribution as required.
- **Model framework:** [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) — AGPL-3.0
- **Orbital propagation:** [python-sgp4](https://github.com/brandon-rhodes/python-sgp4) — MIT
- **Submission:** IBM Bob AI Builders Challenge, August 2025
