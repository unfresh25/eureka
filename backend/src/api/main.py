"""
FastAPI application factory for Eureka Legal Agent.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.exception_handlers import register_exception_handlers
from src.api.v1.router import v1_router
from src.config import get_settings

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """Application factory — creates and configures the FastAPI instance."""
    settings = get_settings()

    app = FastAPI(
        title=settings.api_title,
        version=settings.api_version,
        description="Agente legal colombiano con RAG y generación de documentos.",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    register_exception_handlers(app)

    app.include_router(v1_router)

    @app.on_event("startup")
    async def on_startup() -> None:
        logger.info("Eureka Legal Agent v%s starting up", settings.api_version)

    return app


app = create_app()
