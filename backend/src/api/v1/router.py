"""
API v1 router: aggregates all v1 route modules.
"""

from fastapi import APIRouter

from src.api.v1.routes.chat import conversations_router
from src.api.v1.routes.chat import router as chat_router
from src.api.v1.routes.documents import router as documents_router
from src.domain.schemas import HealthResponse

v1_router = APIRouter(prefix="/api/v1")

v1_router.include_router(chat_router)
v1_router.include_router(conversations_router)
v1_router.include_router(documents_router)


@v1_router.get("/health", response_model=HealthResponse, tags=["health"])
async def health_check() -> HealthResponse:
    """Service liveness check."""
    return HealthResponse()
