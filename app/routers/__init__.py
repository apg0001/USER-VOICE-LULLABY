"""라우터 모듈"""
from .health import router as health_router
from .training import router as training_router
from .inference import router as inference_router
from .models import router as models_router
from .jobs import router as jobs_router

__all__ = [
    "health_router",
    "training_router",
    "inference_router",
    "models_router",
    "jobs_router",
]

