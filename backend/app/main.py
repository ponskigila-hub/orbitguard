"""
OGB — OrbitalGuard
FastAPI application entry point.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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

# CORS — restrict in production; open for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix=API_V1_PREFIX)


@app.get("/", include_in_schema=False)
async def root():
    return {"service": "OGB API", "docs": "/docs"}
