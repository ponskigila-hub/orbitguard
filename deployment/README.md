# OGB — deployment

Deployment configurations for free-tier hosting.

## Backend — Railway / Render / Fly.io

```bash
# From repo root
cd backend
uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

## Frontend — Vercel

```bash
cd frontend
vercel --prod
```

## Environment variables required

| Variable | Description |
|---|---|
| `GEMINI_API_KEY` | Google Gemini API key for the AI Copilot |
