from fastapi import APIRouter
from .detect import router as detect_router
from .copilot import router as copilot_router
from .health import router as health_router

api_router = APIRouter()
api_router.include_router(health_router)
api_router.include_router(detect_router)
api_router.include_router(copilot_router)
