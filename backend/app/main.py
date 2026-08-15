"""
OGB — OrbitalGuard
FastAPI application entry point.
"""
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
load_dotenv()

from app.core.config import API_V1_PREFIX, APP_TITLE, APP_VERSION
from app.api.v1 import api_router

app = FastAPI(
    title=APP_TITLE,
    version=APP_VERSION,
    description=(
        "OGB (OrbitalGuard) is a decision-support system for space operators. "
        "It analyses spacecraft-camera imagery for potential debris, "
        "combines that with orbital data when available, calculates "
        "close-approach risk deterministically, and uses an AI copilot to "
        "explain the situation — the human operator makes all decisions."
    ),
)

# CORS — read allowed origins from env; defaults to localhost for local dev.
# In production set ALLOWED_ORIGINS to the deployed Vercel URL, e.g.:
#   ALLOWED_ORIGINS=https://ogb.vercel.app,https://ogb-git-main-username.vercel.app
_raw_origins = os.environ.get("ALLOWED_ORIGINS", "")
ALLOWED_ORIGINS: list[str] = (
    [o.strip() for o in _raw_origins.split(",") if o.strip()]
    if _raw_origins
    else [
        "http://localhost:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3000",
    ]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

app.include_router(api_router, prefix=API_V1_PREFIX)


@app.get("/", include_in_schema=False)
async def root():
    return {"service": "OGB API", "docs": "/docs"}
