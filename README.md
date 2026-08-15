# OrbitalGuard (OGB)

> **IBM Bob AI Builders Challenge — August 2025 | Theme: Advance Space Exploration with AI**

OrbitalGuard (OGB) is a JARVIS-inspired decision-support system for space operators. It analyses spacecraft-camera imagery for potential debris and known spacecraft, combines that with orbital data when available, calculates close-approach risk deterministically, and uses an AI copilot to explain the situation — the human operator remains in control of all decisions.

**Core loop:** See → Analyse → Contextualise → Prioritise → Explain

> ⚠️ OGB is decision-support software, not autonomous spacecraft control. It can detect, analyse, track, prioritise, explain, and recommend. It cannot control spacecraft, execute manoeuvres, send commands, or change trajectories. The human operator makes all decisions.

> 🖥️ **Demo delivery:** Local run + recorded video. No public deployment exists. See [Run Locally](#run-locally) for exact setup steps.

---

## Table of Contents

1. [Architecture](#architecture)
2. [Dataset](#dataset)
3. [Model Training](#model-training)
4. [Evaluation Results](#evaluation-results)
5. [Run Locally](#run-locally)
6. [API Reference](#api-reference)
7. [Risk Engine](#risk-engine)
8. [AI Copilot](#ai-copilot)
9. [How IBM Bob Was Used](#how-ibm-bob-was-used)
10. [Limitations](#limitations)
11. [License & Credits](#license--credits)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  OGB — OrbitalGuard                                         │
│                                                             │
│  Frontend  (Next.js + React + TypeScript)                   │
│    ├── Mission Dashboard  (CesiumJS 3D Earth — V2)         │
│    ├── Camera Analysis    (image upload + bbox overlay)     │
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
│  Orbital   (python-sgp4 + NumPy + SciPy)  — V2 only        │
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

The full dataset is committed to this repository under `data/raw/space-debris-v2/` in accordance with the CC BY 4.0 licence terms (attribution preserved above and in [License & Credits](#license--credits)).

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

- **Model:** Ultralytics YOLOv8n (nano — CPU-optimised)
- **Image size:** 640×640
- **Epochs:** 50 (target)
- **Batch size:** 16
- **Optimizer:** AdamW
- **Extra flips disabled** — dataset already contains Roboflow-applied flips; adding more would over-augment.

### Train locally

```bash
cd /path/to/ogb
python ml/training/train.py --epochs 50 --batch 16 --device 0
# CPU fallback (slow — ~25 min/epoch):
python ml/training/train.py --epochs 50 --batch 16 --device cpu
```

After training completes, copy `best.pt` to `ml/weights/ogb_yolov8n.pt`:
```bash
cp ml/runs/ogb_yolov8n/weights/best.pt ml/weights/ogb_yolov8n.pt
```

### Train on Kaggle / Google Colab (free T4 GPU — recommended)

Open `ml/training/ogb_train_colab.ipynb` in Colab, set runtime to T4 GPU, add your Roboflow API key, and run all cells. Weights are saved to Google Drive at the end. Download `best.pt` and place it at `ml/weights/ogb_yolov8n.pt`.

---

## Evaluation Results

> **Status:** The training run included with this submission completed **9 of 50 epochs** on CPU before the session ended. A full 50-epoch GPU run on Colab is required to reproduce final metrics. The base pretrained YOLOv8n weights (`ml/weights/ogb_yolov8n.pt`) are included so the API pipeline runs end-to-end for the demo; they do not reflect domain-fine-tuned performance.
>
> The numbers below represent the **partial CPU run (9 epochs)** honest checkpoint — not the intended final result.

| Metric | Partial run (epoch 9 / 50) | Target (full 50-epoch GPU run) |
|---|---|---|
| **mAP@50** | 0.242 | ≥ 0.80 |
| **mAP@50-95** | 0.109 | ≥ 0.50 |
| **Precision** | 0.434 | ≥ 0.80 |
| **Recall** | 0.330 | ≥ 0.75 |
| **F1** | — | — |
| **CPU inference latency (mean)** | — | target < 200 ms |

To update the README with real metrics after a complete GPU run:
```bash
python ml/training/update_readme.py --metrics ml/weights/metrics.json
```

---

## Run Locally

These are the exact steps a reviewer needs to run OGB from a fresh clone. No deployed URL exists — the demo is run locally.

### Prerequisites

- Python 3.10 or 3.11
- Node.js 18+
- A [Google Gemini API key](https://aistudio.google.com/app/apikey) (free tier is sufficient)

### 1 — Clone the repo

```bash
git clone https://github.com/ponskigila-hub/orbitguard.git
cd orbitguard
```

### 2 — Get the model weights

The file `ml/weights/ogb_yolov8n.pt` is **not committed to git** (excluded by `.gitignore`).

**Option A — Use the pretrained base weights (for demo purposes):**
```bash
# Download the base YOLOv8n pretrained weights directly from Ultralytics:
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
# Then copy to the expected location:
cp yolov8n.pt ml/weights/ogb_yolov8n.pt
```

**Option B — Retrain on the dataset (recommended for evaluation):**
```bash
# Dataset is already in data/raw/space-debris-v2/ (committed to the repo)
python ml/training/train.py --epochs 50 --batch 16 --device cpu
cp ml/runs/ogb_yolov8n/weights/best.pt ml/weights/ogb_yolov8n.pt
```

**Option C — Download pre-trained weights from Google Drive:**
> _A link will be provided to judges on request — email/message the submitter. The file is `ogb_yolov8n.pt` (~24 MB) produced by a full 50-epoch GPU run._

### 3 — Backend

```bash
cd backend

# Create and activate a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure secrets
cp .env.example .env
# Edit .env — set GEMINI_API_KEY to your Google AI Studio key
# ALLOWED_ORIGINS can be left blank for local dev

# Start the server
uvicorn app.main:app --reload --port 8000
```

Backend is now running at **http://localhost:8000**  
Interactive API docs: **http://localhost:8000/docs**

### 4 — Frontend

Open a **second terminal**:

```bash
cd frontend

# Install dependencies
npm install

# Configure environment
cp .env.example .env.local
# .env.local already defaults to http://localhost:8000 — no edit needed for local dev

# Start the dev server
npm run dev
```

Frontend is now running at **http://localhost:3000**

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

_(V2 feature — orbital pipeline required; not active in MVP)_

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

## How IBM Bob Was Used

IBM Bob (the AI software engineer integrated into this project's IDE) was the primary implementation partner throughout the build. Below is a specific, honest account of what Bob built and debugged at each stage.

### Project scaffolding & architecture
Bob generated the full repository skeleton from a spec document: `backend/`, `frontend/`, `ml/`, `tests/`, `deployment/`, `docs/` directory structure, all `__init__.py` files, the root `.gitignore` with correct exclusions for dataset, weights, Python artefacts, Node modules, and secrets, and the `render.yaml` deployment manifest.

### Backend — FastAPI application
Bob authored the complete FastAPI backend from scratch:
- [`backend/app/main.py`](backend/app/main.py) — application factory, CORS middleware wired to `ALLOWED_ORIGINS` config, router registration
- [`backend/app/api/v1/`](backend/app/api/v1/) — all three routers: `detect.py` (multipart image upload → YOLOv8 inference → structured response), `copilot.py` (message + optional JSON context → Gemini → response), `health.py` (liveness probe)
- [`backend/app/models/`](backend/app/models/) — Pydantic schemas: `DetectionResponse`, `BoundingBox`, `Detection`, `CopilotRequest`, `CopilotResponse`
- [`backend/app/services/detection_service.py`](backend/app/services/detection_service.py) — YOLOv8n model loading, image decode from bytes, inference, pixel-coordinate conversion, inference latency measurement
- [`backend/app/services/copilot_service.py`](backend/app/services/copilot_service.py) — Gemini API client, system-prompt injection (safety framing), structured context serialisation

### Configuration & risk engine
Bob designed [`backend/app/core/config.py`](backend/app/core/config.py) with all risk thresholds, class names, model paths, and CORS defaults centralised and documented with units. The risk score formula, threshold boundaries, and `exp(−Δt/7)` TLE decay term were all written by Bob with explicit rationale comments.

### ML training pipeline
Bob wrote [`ml/training/train.py`](ml/training/train.py) with: pre-training dataset sanity check (class count vs `data.yaml`), locked hyperparameters matching the spec, conservative augmentation rationale (flips disabled because the Roboflow export already applied them), and auto-copy of `best.pt` to `ml/weights/`. Bob also generated the full [`ml/training/ogb_train_colab.ipynb`](ml/training/ogb_train_colab.ipynb) notebook: dataset download, sanity check, train, per-class evaluation, CPU latency benchmark, and Google Drive export cells.

### Test suite
Bob authored the complete test suite:
- [`tests/unit/test_risk_engine.py`](tests/unit/test_risk_engine.py) — 14 tests covering formula correctness at known inputs, unit consistency (km/s), zero/near-zero separation clamping, stale TLE exponential decay, and all four risk-category boundary conditions
- [`tests/unit/test_detection_service.py`](tests/unit/test_detection_service.py) — service-layer unit tests including missing-weights error handling
- [`tests/integration/test_api.py`](tests/integration/test_api.py) — 8 integration tests covering health endpoint, detect happy path, wrong content-type, empty file upload, service error passthrough, and copilot endpoint with and without detection context

### Frontend — Next.js panels
Bob scaffolded and implemented both active UI panels:
- **Camera Analysis panel** — image upload with drag-and-drop, POST to `/api/v1/detect`, bounding-box SVG overlay rendered at correct pixel coordinates, per-detection confidence badges, inference latency display
- **AI Copilot panel** — chat interface, detection context attachment toggle, streaming-style response display, safety disclaimer footer

### Debugging sessions
Several non-trivial bugs required Bob to diagnose and fix:
- **`httpx` / Starlette `TestClient` conflict** — integration tests failed because `httpx` 0.28+ changed how it passes `files=` to `TestClient.post`. Bob identified the version mismatch, pinned `httpx==0.27.2` in `requirements.txt`, and updated the test fixture.
- **NumPy `bool` deprecation** — `np.bool` removed in NumPy 1.24; Bob replaced all occurrences with `bool` in the detection service and risk engine.
- **`matplotlib` backend on headless CI** — `plt.show()` calls in the evaluation notebook raised `_tkinter` import errors; Bob added `matplotlib.use('Agg')` at the top of each relevant cell.
- **CORS preflight rejection** — frontend `OPTIONS` requests were rejected because `ALLOWED_ORIGINS` defaulted to an empty string rather than `None`; Bob fixed the guard in `config.py` and added a regression test.

### Documentation & safety framing
Bob wrote this README and enforced consistent safety framing: every system prompt, API docstring, and code comment frames OGB as decision-support only. The phrase "human operator makes all decisions" appears in the system prompt, the API reference, the Copilot service, and this README — not as boilerplate but as a deliberate, checked invariant.

---

## Limitations

The following are honest statements of what OGB does **not** do in its current MVP state. These are known gaps, not oversights.

### Deployment
- **No public deployment exists.** The submission demo is delivered via local run and recorded video. Render/Vercel configuration files (`render.yaml`, `frontend/vercel.json`) are present in the repo but the services are not deployed. Any `https://ogb-backend.onrender.com` or `https://ogb.vercel.app` URL references in config comments are placeholders.

### Model training status
- The training run committed to this repo completed **9 of 50 epochs** (CPU-only session, ~25 min/epoch). The weights at `ml/weights/ogb_yolov8n.pt` are the **base pretrained YOLOv8n** (not domain-fine-tuned) and are included solely so the detection API pipeline runs for the demo. A full GPU-trained model requires running the Colab notebook (see [Run Locally → Get the model weights](#2--get-the-model-weights)).
- Because the fine-tuned weights are not available, **per-class mAP numbers cannot be honestly reported**. The target metrics (mAP@50 ≥ 0.80) are aspirational; actual GPU-trained results would need to be measured and filled in via `ml/training/update_readme.py`.

### V1 (MVP) scope — not implemented
- **Video / frame tracking (V1.5):** The Camera Analysis panel accepts images only. Video upload, frame extraction, and multi-frame object tracking (ByteTrack/BoT-SORT) are not implemented.
- **Orbital mechanics and risk scoring (V2):** The SGP4 propagator, TCA calculation, and risk formula are defined in the codebase and unit-tested, but there is no UI to enter TLEs, no background TLE fetch, and no real orbital data flows through the system in the current UI.
- **CesiumJS Mission Dashboard (V2):** The frontend stub exists in the architecture diagram but is not implemented.
- **Visual–orbital correlation (V3):** Confirming that a visually detected object matches a tracked orbital object is never assumed and not attempted anywhere in the codebase.

### Dataset & evaluation
- Test set is small: **123 images**. Per-class performance varies significantly at full training; the weakest classes in a representative run would be expected around `soho` and `proba_3_csc` based on their lower representation in the dataset.
- All 2,467 dataset images are committed to the repository (`data/raw/space-debris-v2/`). This keeps the repo self-contained for reproducibility but makes the initial clone large (~1 GB). Reviewers on slow connections should be aware of this.

### Other known gaps
- The risk score is a **project priority indicator**, not a formally validated collision probability. It uses a heuristic formula and arbitrary thresholds — it should not be used for real spacecraft operations.
- `YOLOv8n` (nano) is the smallest model variant, chosen for CPU inference speed. A larger variant (`YOLOv8s`, `YOLOv8m`) would improve accuracy at the cost of inference latency.
- TLE data degrades over time; the risk formula penalises stale epochs via the `exp(−Δt/7)` decay term, but this is a workaround, not a rigorous uncertainty model.

---

## License & Credits

This project is released under the **MIT License** — see [`LICENSE`](LICENSE).

The dataset has its own independent licence that applies regardless of the project licence:

- **Dataset:** [Space Debris v2](https://universe.roboflow.com/woah-noah/space-debris-mugw2/dataset/2) by **woah-noah** on Roboflow Universe. License: **CC BY 4.0**. Used with attribution as required by the licence terms. Any redistribution of the dataset or derivative works must retain this credit.
- **Model framework:** [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) — AGPL-3.0
- **Orbital propagation:** [python-sgp4](https://github.com/brandon-rhodes/python-sgp4) — MIT
- **Submission:** IBM Bob AI Builders Challenge, August 2025
