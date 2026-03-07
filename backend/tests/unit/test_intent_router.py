"""
Unit tests for IntentRouter.

Uses mocked Ollama client to test classification logic in isolation.
All external I/O is mocked — these tests are fast and require no running services.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agent.intent_router import OllamaIntentRouter
from src.domain.models import IntentType, Message, MessageRole


@pytest.fixture
def mock_llm() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def router(mock_llm: AsyncMock) -> OllamaIntentRouter:
    r = OllamaIntentRouter(model="test-model", ollama_host="http://localhost:11434")
    r._client = mock_llm
    return r


def _make_response(content: str) -> MagicMock:
    msg = MagicMock()
    msg.content = content
    resp = MagicMock()
    resp.message = msg
    return resp


@pytest.mark.asyncio
async def test_classify_consulta(router: OllamaIntentRouter, mock_llm: AsyncMock) -> None:
    """Responds with 'consulta' when user asks a legal question."""
    mock_llm.chat.return_value = _make_response(
        '{"intent": "consulta", "doc_type": null, "confidence": 0.95}'
    )
    result = await router.classify("¿Qué es una tutela?", history=[])
    assert result.intent == IntentType.CONSULTA
    assert result.doc_type is None
    assert result.confidence == pytest.approx(0.95)


@pytest.mark.asyncio
async def test_classify_generar_documento(router: OllamaIntentRouter, mock_llm: AsyncMock) -> None:
    """Responds with 'generar_documento' when user wants a document."""
    mock_llm.chat.return_value = _make_response(
        '{"intent": "generar_documento", "doc_type": "tutela", "confidence": 0.99}'
    )
    result = await router.classify("Necesito una tutela por negación de medicamento", history=[])
    assert result.intent == IntentType.GENERAR_DOCUMENTO
    assert result.doc_type == "tutela"


@pytest.mark.asyncio
async def test_classify_fallback_on_invalid_json(
    router: OllamaIntentRouter, mock_llm: AsyncMock
) -> None:
    """Falls back to 'consulta' when LLM returns unparseable output."""
    mock_llm.chat.return_value = _make_response("No puedo clasificar esto")
    result = await router.classify("algo raro", history=[])
    assert result.intent == IntentType.CONSULTA
    assert result.confidence == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_classify_with_markdown_fences(
    router: OllamaIntentRouter, mock_llm: AsyncMock
) -> None:
    """Handles JSON wrapped in markdown code fences."""
    mock_llm.chat.return_value = _make_response(
        '```json\n{"intent": "clarificar", "doc_type": null, "confidence": 0.7}\n```'
    )
    result = await router.classify("no sé qué necesito", history=[])
    assert result.intent == IntentType.CLARIFICAR


@pytest.mark.asyncio
async def test_classify_includes_history(router: OllamaIntentRouter, mock_llm: AsyncMock) -> None:
    """History messages are included in the Ollama messages list."""
    mock_llm.chat.return_value = _make_response(
        '{"intent": "consulta", "doc_type": null, "confidence": 0.8}'
    )
    history = [
        Message(role=MessageRole.USER, content="Hola"),
        Message(role=MessageRole.ASSISTANT, content="Hola, ¿en qué le puedo ayudar?"),
    ]
    await router.classify("¿Y qué pasa con los derechos de petición?", history=history)

    call_args = mock_llm.chat.call_args
    messages = call_args.kwargs["messages"]
    assert len(messages) == 4
