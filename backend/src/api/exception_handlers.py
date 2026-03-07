"""
Global exception handlers for FastAPI.

Maps domain exceptions → consistent JSON error responses.
Unexpected exceptions return a generic 500 to avoid leaking internals.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src.domain.errors import EurekaError

logger = logging.getLogger(__name__)


def register_exception_handlers(app: FastAPI) -> None:
    """Attach all exception handlers to the FastAPI application."""

    @app.exception_handler(EurekaError)
    async def handle_eureka_error(request: Request, exc: EurekaError) -> JSONResponse:
        """Handle all known domain errors with their prescribed HTTP status codes."""
        logger.warning(
            "Domain error [%s] on %s %s: %s",
            exc.__class__.__name__,
            request.method,
            request.url.path,
            exc.message,
        )
        return JSONResponse(
            status_code=exc.http_status,
            content=exc.to_dict(),
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        """Handle unexpected errors without exposing internals."""
        logger.exception("Unexpected error on %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=500,
            content={
                "error": "InternalError",
                "message": "An unexpected error occurred. Please try again.",
                "code": "INTERNAL_ERROR",
                "details": {},
            },
        )
