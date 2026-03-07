"""
Shared pytest fixtures for Eureka test suite.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def sample_legal_question() -> str:
    return "¿Cuándo procede una acción de tutela por vulneración al derecho a la salud?"


@pytest.fixture
def sample_document_request() -> str:
    return (
        "Necesito una tutela porque la EPS me negó el medicamento Rituximab recetado por mi médico."
    )
