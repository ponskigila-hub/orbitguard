# OrbitalGuard (OGB)

> **IBM Bob AI Builders Challenge — August 2026 | Theme: Advance Space Exploration with AI**

OrbitalGuard (OGB) is a JARVIS-inspired decision-support system for space operators. It analyses spacecraft-camera imagery for potential debris and known spacecraft, combines that with orbital data when available, calculates close-approach risk deterministically, and uses an AI copilot to explain the situation — the human operator remains in control of all decisions.

**Core loop:** See → Analyse → Contextualise → Prioritise → Explain

> ⚠️ OGB is decision-support software, not autonomous spacecraft control. It can detect, analyse, track, prioritise, explain, and recommend. It cannot control spacecraft, execute manoeuvres, send commands, or change trajectories. The human operator makes all decisions.

> 🖥️ **Demo delivery:** Local run + recorded video. No public deployment exists. See [Run Locally](#run-locally) for exact setup steps.

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Solution Description](#solution-description)
3. [Selected Challenge Theme](#selected-challenge-theme)
4. [Architecture](#architecture)
5. [Dataset](#dataset)
6. [Model Training](#model-training)
7. [Evaluation Results](#evaluation-results)
8. [Run Locally](#run-locally)
9. [API Reference](#api-reference)
10. [Orbital Intelligence](#orbital-intelligence)
11. [Risk Engine](#risk-engine)
12. [AI Copilot](#ai-copilot)
13. [Session Features](#session-features)
14. [How IBM Bob Was Used](#how-ibm-bob-was-used)
15. [Limitations](#limitations)
16. [License & Credits](#license--credits)

---

## Problem Statement

Space operators — satellite operators, small satcom companies, and space situational awareness analysts — must monitor both **visual camera feeds** and **orbital catalog data** to identify potential debris and close-approach threats. In practice, these two data sources are processed manually and separately: an operator has to eyeball a camera image for anomalies, then separately cross-reference orbital elements (TLEs) against a catalog, then manually calculate closest-approach distance and relative velocity to judge risk. This is slow, requires specialised orbital-mechanics expertise, and is error-prone precisely when speed matters most — during an active close-approach window.

**How might we unify visual debris detection and orbital risk analysis into a single AI-assisted workflow, so a space operator can assess collision risk quickly and with confidence, without manually cross-referencing data sources or performing orbital calculations by hand?**

---

## Solution Description

OGB (OrbitalGuard) is a decision-support tool that closes this gap end-to-end:

1. **See** — a fine-tuned YOLOv8 model detects and classifies spacecraft/debris objects from camera imagery.
2. **Analyse** — when orbital data (a TLE) is available, real SGP4 propagation and a two-phase close-approach search compute actual position, velocity, time of closest approach, and minimum separation — no estimation, real orbital mechanics.
3. **Contextualise & Prioritise** — a deterministic, auditable risk formula converts those physical values into a priority score and category (LOW → CRITICAL), so operators can triage multiple objects at a glance via the Threat Center.
4. **Explain** — an AI Copilot, grounded in the actual structured output of the vision and orbital pipelines (and able to call the same real calculation functions via tool-calling), explains what was found in plain language — explicitly declining to speculate when data isn't available, rather than guessing.

At every step, **the human operator makes the final call** — OGB detects, analyses, prioritises, and explains; it never issues commands or autonomously acts on a spacecraft.

---

## Selected Challenge Theme

**Theme: Advance Space Exploration with AI** (IBM Bob AI Builders Challenge, August 2026).

OGB was built specifically around this theme's judging criteria:

- **Technical Execution** — a working, tested pipeline combining computer vision (YOLOv8), real orbital mechanics (SGP4, validated against a published reference vector), a deterministic risk engine, and an LLM copilot with function-calling — not a single-model demo.
- **Innovation** — AI is the functional core, not a wrapper: the Copilot doesn't just summarise text, it calls real backend physics/risk functions as tools and reasons over genuinely structured, multi-source (vision + orbital) context.
- **Challenge Fit** — directly addresses a real space-operations pain point (fragmented visual/orbital situational awareness) rather than a generic AI demo retrofitted with a space theme.
- **Feasibility** — built on free-tier-friendly, CPU-viable components (YOLOv8n, python-sgp4, SciPy) with no paid infrastructure required to run or reproduce.
- **Real-World Impact** — session history, exportable reports (JSON/PDF), and a clear decision-support (not autonomous-control) framing reflect how a tool like this would actually need to behave in an operational context.

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

> **Status:** Training completed — **50 epochs on Colab T4 GPU**. The fine-tuned weights at `ml/weights/ogb_yolov8n.pt` are the domain-trained model. Verified by `python ml/inference/smoke_test.py` which returns class `cheops` at confidence 0.782, a label that cannot appear in base pretrained YOLOv8n (trained on COCO's 80 classes).
>
> Metrics come from the completed Colab run. `ml/weights/metrics.json` contains the machine-readable copy. If you re-train, regenerate it and run `python ml/training/update_readme.py` to refresh this table.

| Metric | Result (50-epoch GPU run) |
|---|---|
| **mAP@50** | 0.8156 |
| **mAP@50-95** | 0.4599 |
| **Precision** | 0.7189 |
| **Recall** | 0.7502 |
| **F1** | 0.7342 |
| **CPU inference latency (mean)** | 140.9 ms (median 133.1 ms, p95 196.4 ms) |

CPU latency measured over 20 runs on Colab CPU benchmark (T4-adjacent).

### Per-class mAP@50

| Class | mAP@50 |
|---|---|
| `cheops` | 0.924 |
| `debris` | 0.788 |
| `double_start` | 0.965 |
| `earth_observation_sat_1` | 0.812 |
| `lisa_pathfinder` | 0.995 |
| `proba_2` | 0.755 |
| `proba_3_csc` | 0.666 ⚠️ |
| `proba_3_ocs` | 0.918 |
| `smart_1` | 0.825 |
| `soho` | 0.529 ⚠️ |
| `xmm_newton` | 0.794 |

⚠️ `soho` and `proba_3_csc` are the weakest classes. Both have ~10–12 images in the test split, so their per-class mAP is sensitive to small sample variance — they are not necessarily harder objects, just underrepresented in the test set.

To regenerate metrics after a new training run:
```bash
python ml/training/update_readme.py --metrics ml/weights/metrics.json
# metrics.json is produced by cell-07-export in ml/training/ogb_train_colab.ipynb
# If metrics.json is absent, the script exits with an error — do not report numbers
# from results.csv as evaluation metrics; run the full Colab evaluation cell first.
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

The file `ml/weights/ogb_yolov8n.pt` is **not committed to git** (excluded by `.gitignore`). The weights at this path are the domain fine-tuned model from the completed 50-epoch Colab T4 GPU run.

**Option A — Download the fine-tuned weights (recommended):**
> A link will be provided to judges on request — email/message the submitter. The file is `ogb_yolov8n.pt` (~6 MB, fine-tuned on the Space Debris v2 dataset, 50 epochs GPU). Produces `cheops` class detections, confirming domain fine-tuning.

**Option B — Retrain from scratch on the dataset:**
```bash
# Dataset is already in data/raw/space-debris-v2/ (committed to the repo)
python ml/training/train.py --epochs 50 --batch 16 --device cpu
cp ml/runs/ogb_yolov8n/weights/best.pt ml/weights/ogb_yolov8n.pt
```

**Option C — Use the base pretrained YOLOv8n (pipeline demo only, no domain classes):**
```bash
# WARNING: base YOLOv8n is trained on COCO's 80 classes only.
# It cannot output cheops/debris/etc. labels. Use only to verify the API
# pipeline runs end-to-end; do not treat its detections as evaluation results.
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
cp yolov8n.pt ml/weights/ogb_yolov8n.pt
```

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

### `POST /api/v1/orbital/analyze` _(V2)_

Propagate a TLE and optionally run conjunction analysis.

**Request:**
```json
{
  "tle_line1": "1 88888U ...",
  "tle_line2": "2 88888 ...",
  "timestamp_utc": "2025-08-01T12:00:00Z",
  "target_tle_line1": "1 88889U ...",
  "target_tle_line2": "2 88889 ...",
  "conjunction_window_hours": 24
}
```

**Response:**
```json
{
  "propagated_state": {
    "ok": true,
    "position_km": [2328.97, -5995.22, 1719.97],
    "velocity_km_s": [2.9121, -0.9834, -7.0908],
    "epoch_utc": "1980-10-01T23:41:24.113Z",
    "tle_age_days": 0.0,
    "propagated_at_utc": "1980-10-01T23:41:24.113Z"
  },
  "conjunction": {
    "ok": true,
    "tca_utc": "1980-10-02T00:15:42.000Z",
    "d_min_km": 12.456,
    "v_rel_km_s": 0.832,
    "risk": { "risk_score": 0.000034, "risk_category": "LOW" }
  },
  "analysis_type": "propagation_and_conjunction"
}
```

---

## Orbital Intelligence

### SGP4 Source

Uses **python-sgp4** by Brandon Rhodes (https://github.com/brandon-rhodes/python-sgp4), version ≥ 2.23.

API: `sgp4.api.Satrec` + `jday` (the Cython-accelerated v2.x interface).

### Close-Approach Algorithm

Two-phase approach to find Time of Closest Approach (TCA):

1. **Coarse phase** — sample `|r1(t) − r2(t)|` every 60 seconds across the analysis window to find the sub-interval that brackets the minimum.
2. **Fine phase** — `scipy.optimize.minimize_scalar` (Brent's method, bounded) on the separation function over the bracketed interval ± 120 s guard, to sub-second TCA precision.
3. **Relative velocity** — central finite difference at TCA ± 1 s.
4. **Risk** — delegates directly to `risk_service.calculate_risk_full()`. No math is re-implemented.

### Reference Test Vector

The SGP4 implementation is validated against **object 88888** from:

> Vallado, D. A.; Crawford, P.; Hujsak, R.; Kelso, T. S. (2006).
> _"Revisiting Spacetrack Report #3: Rev 2."_ AIAA 2006-6753.

TLE (from `sgp4` package's `SGP4-VER.TLE` file):
```
1 88888U          80275.98708465  .00073094  13844-3  66816-4 0    87
2 88888  72.8435 115.9689 0086731  52.6988 110.5714 16.05824518  1058
```

Reference state at tsince = 0 (from `sgp4` package `tcppver.out`):
```
r = [2328.96975262, −5995.22051338, 1719.97297192] km
v = [2.912073281, −0.983417956, −7.090816210] km/s
```

OGB computed result agrees to:
- Position: δ < 1 × 10⁻³ km (1 m) for all three components ✓
- Velocity: δ < 1 × 10⁻⁶ km/s (1 mm/s) for all three components ✓

Verified by `tests/unit/test_orbital_service.py` (12 tests, all pass).

### V2 End-to-End Verification

The full V2 pipeline was manually verified end-to-end:

- **SGP4 propagation** — tested against a real ISS TLE; computed altitude (~420 km) and velocity (~7.67 km/s) matched published ISS orbital parameters (413–422 km altitude, 7.67 km/s orbital speed) to within expected precision.
- **Conjunction analysis** — tested with ISS + Tiangong TLEs; returned TCA, d_min ≈ 8190 km, v_rel ≈ 10.3 km/s, and a LOW risk score (~0.000001) — manually cross-checked against the locked risk formula and confirmed mathematically consistent.
- **Copilot `propagate_tle` tool** — confirmed working via live test: given a real TLE in chat, Copilot correctly invoked the tool (visible tool-call indicator) and returned SGP4-derived position/velocity matching manual verification.
- **Full test suite** — 64/64 tests passing as of this verification.

### UI: Orbital Intelligence Page

Navigate to `/orbital` (link in the top navigation bar). Features:
- **TLE Analysis Form** — paste TLE lines, optional timestamp, optional conjunction target
- **Threat Details** — full result: position vector, velocity vector, TLE age, TCA, d_min, v_rel, risk
- **Threat Center** — list of all analyzed objects, color-coded 🔴🟠🟡🟢, sortable by risk/d_min/TLE age

---

## Risk Engine

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
- Function-calling tools (`calculate_risk_score`, `propagate_tle`) let the Copilot report real calculated values in conversation — the LLM never computes orbital mechanics or risk math itself; it always calls the same tested backend functions used elsewhere in the app

---

## Session Features

### Detection History & Session Memory

Each Camera Analysis session keeps an in-memory (not persisted — no database, per the locked architecture) history of prior detections, shown as a thumbnail strip above the camera feed. The AI Copilot receives a compact summary of this history alongside the current detection, so it can answer comparative questions such as "compare this to the previous detection" or "how many debris have I found this session?" — grounded only in what actually happened in the session, never inventing prior detections that don't exist. Suggested-question chips adapt once session history is available.

### Confidence Threshold Filter

A client-side confidence slider on the Camera Analysis panel lets the operator filter which returned detections are displayed, down to the backend's own minimum confidence floor (25%). Filtering happens instantly against the already-returned detection set — no extra backend call per slider movement — with the bounding boxes, detection count, and summary panel updating live.

### Polished Loading & Transition States

- Skeleton loading placeholders (Camera Analysis and Orbital Intelligence pages) matching the eventual layout, replacing a bare spinner/blank state during upload and inference.
- Bounding boxes fade in over a 200 ms CSS opacity transition when a new detection result arrives, rather than appearing instantly.

### Export Detection Report

An "Export detection report" button on the Camera Analysis panel produces:
- **JSON export** — the full structured detection response, plus the session history summary
- **PDF export** — a client-side generated (via [jsPDF](https://github.com/parallax/jsPDF)) one-page report: image thumbnail, detection list with class/confidence, session history summary, and the standard decision-support safety disclaimer

Both exports run entirely client-side — no backend PDF service was added, keeping the "no database, free-first" architecture intact.

---

## How IBM Bob Was Used

IBM Bob (the AI software engineer integrated into this project's IDE) was the primary implementation partner throughout the build. Below is a specific, honest account of what Bob built and debugged at each stage.

### Project scaffolding & architecture
Bob generated the full repository skeleton from a spec document: `backend/`, `frontend/`, `ml/`, `tests/`, `deployment/`, `docs/` directory structure, all `__init__.py` files, the root `.gitignore` with correct exclusions for dataset, weights, Python artefacts, Node modules, and secrets, and the `render.yaml` deployment manifest.

### Backend — FastAPI application
Bob authored the complete FastAPI backend from scratch:
- [`backend/app/main.py`](backend/app/main.py) — application factory, CORS middleware wired to `ALLOWED_ORIGINS` config, router registration
- [`backend/app/api/v1/`](backend/app/api/v1/) — all routers: `detect.py` (multipart image upload → YOLOv8 inference → structured response), `copilot.py` (message + optional JSON context → Gemini → response, now with function-calling tools), `health.py` (liveness probe), `orbital.py` (TLE propagation + conjunction analysis)
- [`backend/app/models/`](backend/app/models/) — Pydantic schemas: `DetectionResponse`, `BoundingBox`, `Detection`, `CopilotRequest`, `CopilotResponse`, `OrbitalAnalyzeRequest`, `PropagatedState`, `ConjunctionResult`, `OrbitalAnalyzeResponse`
- [`backend/app/services/detection_service.py`](backend/app/services/detection_service.py) — YOLOv8n model loading, image decode from bytes, inference, pixel-coordinate conversion, inference latency measurement, enriched Copilot context (frame position, per-class mAP advisory, spatial relationships)
- [`backend/app/services/copilot_service.py`](backend/app/services/copilot_service.py) — Gemini API client, system-prompt injection (safety framing), structured context serialisation, function-calling tool execution
- [`backend/app/services/orbital_service.py`](backend/app/services/orbital_service.py) — SGP4 TLE propagation (`sgp4.api.Satrec` + `jday`), TLE validation
- [`backend/app/services/conjunction_service.py`](backend/app/services/conjunction_service.py) — coarse + fine phase close-approach analysis, TCA/d_min/v_rel calculation, risk engine integration

### Configuration & risk engine
Bob designed [`backend/app/core/config.py`](backend/app/core/config.py) with all risk thresholds, class names, model paths, and CORS defaults centralised and documented with units. The risk score formula, threshold boundaries, and `exp(−Δt/7)` TLE decay term were all written by Bob with explicit rationale comments.

### ML training pipeline
Bob wrote [`ml/training/train.py`](ml/training/train.py) with: pre-training dataset sanity check (class count vs `data.yaml`), locked hyperparameters matching the spec, conservative augmentation rationale (flips disabled because the Roboflow export already applied them), and auto-copy of `best.pt` to `ml/weights/`. Bob also generated the full [`ml/training/ogb_train_colab.ipynb`](ml/training/ogb_train_colab.ipynb) notebook: dataset download, sanity check, train, per-class evaluation, CPU latency benchmark, and Google Drive export cells.

### Test suite
Bob authored the complete test suite (64 tests total):
- [`tests/unit/test_risk_engine.py`](tests/unit/test_risk_engine.py) — 12 tests covering formula correctness at known inputs, unit consistency (km/s), zero/near-zero separation clamping, stale TLE exponential decay, and all four risk-category boundary conditions
- [`tests/unit/test_detection_service.py`](tests/unit/test_detection_service.py) — 6 service-layer unit tests including summary generation and missing-weights error handling
- [`tests/integration/test_api.py`](tests/integration/test_api.py) — 8 integration tests covering health endpoint, detect happy path, wrong content-type, empty file upload, service error passthrough, and copilot endpoint with and without detection context
- [`tests/unit/test_orbital_service.py`](tests/unit/test_orbital_service.py) — 12 tests validating SGP4 propagation against the Vallado 2006 reference vector and TLE input validation
- [`tests/unit/test_conjunction_service.py`](tests/unit/test_conjunction_service.py) — 9 tests covering close-approach edge cases (identical TLEs, stale TLE, no convergence, normal crossing orbits)
- [`tests/unit/test_tool_calling.py`](tests/unit/test_tool_calling.py) — tests confirming the Copilot's function-calling tools (`calculate_risk_score`, `propagate_tle`) invoke the real backend math rather than LLM-generated numbers

### Frontend — Next.js panels
Bob scaffolded and implemented all active UI panels:
- **Camera Analysis panel** — image upload with drag-and-drop, POST to `/api/v1/detect`, bounding-box SVG overlay rendered at correct pixel coordinates with a 200 ms fade-in transition, per-detection confidence badges, inference latency display, confidence threshold slider, session history thumbnail strip, and export report button
- **AI Copilot panel** — chat interface with internal scroll (fixed-height, auto-scroll to newest message), markdown rendering, detection context attachment, session history awareness, suggested-question chips, function-calling tool-call indicators, and safety disclaimer footer
- **Orbital Intelligence panel** (`/orbital`) — TLE analysis form, Threat Center list, Threat Details panel

### Debugging sessions
Several non-trivial bugs required Bob to diagnose and fix:
- **`httpx` / Starlette `TestClient` conflict** — integration tests failed because `httpx` 0.28+ changed how it passes `files=` to `TestClient.post`. Bob resolved this by using `fastapi.testclient.TestClient` directly (FastAPI 0.111+ ships its own bundled client) rather than pinning `httpx`.
- **NumPy 2.x / matplotlib ABI mismatch** — a shared local Python environment had NumPy 2.x installed alongside a matplotlib build compiled against NumPy 1.x, crashing at import time. Resolved by using a dedicated virtual environment for this project rather than a shared course environment.
- **CORS preflight rejection** — frontend `OPTIONS` requests were rejected because `ALLOWED_ORIGINS` defaulted to an empty string rather than `None`; Bob fixed the guard in `config.py` and added a regression test.
- **Copilot markdown rendering & tone consistency** — raw markdown syntax (`**bold**`) was displaying literally in the chat UI; fixed with `react-markdown`. A follow-up pass caught that the more conversational tone update had only been applied to some response categories, not all — fixed by consolidating to a single system-prompt tone instruction applied uniformly.
- **Fabricated evaluation numbers (caught and corrected)** — at one point, evaluation metrics in the README were reported from a stale/incorrect source (an interrupted 9-epoch local CPU run) rather than the completed 50-epoch Colab GPU run. This was caught by independently verifying the weights file via `smoke_test.py` (the `cheops` class output is impossible from base pretrained YOLOv8n, proving the weights were correctly fine-tuned) and cross-checking file sizes, then correcting the README with the verified numbers and adding an explicit `metrics.json` provenance note to prevent recurrence.

### Documentation & safety framing
Bob wrote this README and enforced consistent safety framing: every system prompt, API docstring, and code comment frames OGB as decision-support only. The phrase "human operator makes all decisions" appears in the system prompt, the API reference, the Copilot service, and this README — not as boilerplate but as a deliberate, checked invariant.

---

## Limitations

The following are honest statements of what OGB does **not** do in its current state. These are known gaps, not oversights.

### Deployment
- **No public deployment exists.** The submission demo is delivered via local run and recorded video. Render/Vercel configuration files (`render.yaml`, `frontend/vercel.json`) are present in the repo but the services are not deployed. Any `https://ogb-backend.onrender.com` or `https://ogb.vercel.app` URL references in config comments are placeholders.

### Model training status
- Training is **complete**: 50 epochs on Colab T4 GPU. The weights at `ml/weights/ogb_yolov8n.pt` are the domain fine-tuned model; `ml/inference/smoke_test.py` confirms this — it returns class `cheops` at confidence 0.782, which is impossible from base pretrained YOLOv8n (COCO classes only). See [Evaluation Results](#evaluation-results) for the verified metrics.
- `ml/runs/ogb_yolov8n/results.csv` in this repo records only 9 CPU epochs (an earlier interrupted local run). That CSV is **not** the source of the evaluation metrics; the Colab GPU run produced the final weights and metrics recorded in `ml/weights/metrics.json`.

### V1 (MVP) scope — not implemented
- **Video / frame tracking (V1.5):** The Camera Analysis panel accepts images only. Video upload, frame extraction, and multi-frame object tracking (ByteTrack/BoT-SORT) are not implemented.
- **Visual–orbital correlation (V3):** Confirming that a visually detected object matches a tracked orbital object is never assumed and not attempted anywhere in the codebase.

### V2 (Orbital Intelligence) — implemented
- **SGP4 propagation:** `POST /api/v1/orbital/analyze` is live. TLE → ECI state validated against Vallado 2006 object 88888.
- **Conjunction analysis:** coarse 60 s sampling + scipy Brent fine-search. 9 tests pass.
- **Threat Center UI:** `/orbital` page with form, details panel, and sortable threat list.
- **CesiumJS (V2.5):** Not implemented. Skipped per project scope — a working Threat Center without 3D visuals beats a broken one with them.

### Dataset & evaluation
- Test set is small: **123 images**. Per-class performance varies significantly; the weakest classes (`soho`, `proba_3_csc`) have only ~10–12 images in the test split, so their mAP is sensitive to small sample variance — they are not necessarily harder objects, just underrepresented in the test set.
- All 2,467 dataset images are committed to the repository (`data/raw/space-debris-v2/`). This keeps the repo self-contained for reproducibility but makes the initial clone large (~1 GB). Reviewers on slow connections should be aware of this.

### Other known gaps
- The risk score is a **project priority indicator**, not a formally validated collision probability. It uses a heuristic formula and arbitrary thresholds — it should not be used for real spacecraft operations.
- `YOLOv8n` (nano) is the smallest model variant, chosen for CPU inference speed. A larger variant (`YOLOv8s`, `YOLOv8m`) would improve accuracy at the cost of inference latency.
- TLE data degrades over time; the risk formula penalises stale epochs via the `exp(−Δt/7)` decay term, but this is a workaround, not a rigorous uncertainty model.
- Session history and export features are in-memory / client-side only (no database, per the locked architecture) — history does not persist across a page reload.

---

## License & Credits

This project is released under the **MIT License** — see [`LICENSE`](LICENSE).

The dataset has its own independent licence that applies regardless of the project licence:

- **Dataset:** [Space Debris v2](https://universe.roboflow.com/woah-noah/space-debris-mugw2/dataset/2) by **woah-noah** on Roboflow Universe. License: **CC BY 4.0**. Used with attribution as required by the licence terms. Any redistribution of the dataset or derivative works must retain this credit.
- **Model framework:** [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) — AGPL-3.0
- **Orbital propagation:** [python-sgp4](https://github.com/brandon-rhodes/python-sgp4) — MIT
- **PDF export:** [jsPDF](https://github.com/parallax/jsPDF) — MIT
- **Submission:** IBM Bob AI Builders Challenge, August 2026