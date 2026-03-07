"""
Intent Router: classifies user messages into actionable intents.

Uses Qwen2.5-VL via Ollama with a concise JSON prompt. Raises LLMError
on communication failures and returns a safely parsed IntentResult.
"""

from __future__ import annotations

import json
import logging
from typing import Protocol, runtime_checkable

import ollama

from src.domain.errors import LLMError
from src.domain.models import IntentResult, IntentType, Message

logger = logging.getLogger(__name__)

_ROUTER_SYSTEM_PROMPT = """Eres un clasificador de intenciones para un asistente legal colombiano.
Clasifica el mensaje del usuario en exactamente UNO de estos intents:
- "consulta": el usuario hace una pregunta legal o pide información.
- "generar_documento": el usuario quiere crear un documento legal (tutela, demanda, contrato, 
derecho de petición, poder, etc.)
- "clarificar": no hay suficiente información para responder o generar.

Responde ÚNICAMENTE con un JSON válido sin texto adicional:
{"intent": "<intent>", "doc_type": "<tipo_de_documento_o_null>", "confidence": <0.0_a_1.0>}
"""


@runtime_checkable
class IntentClassifier(Protocol):
    """Interface for intent classification."""

    async def classify(self, message: str, history: list[Message]) -> IntentResult: ...


class OllamaIntentRouter:
    """
    Intent classifier backed by Qwen2.5-VL via Ollama.

    SRP: only responsible for classifying user intent.
    """

    def __init__(self, model: str, ollama_host: str) -> None:
        self._model = model
        self._client = ollama.AsyncClient(host=ollama_host)

    async def classify(self, message: str, history: list[Message]) -> IntentResult:
        """Classify the user's message intent."""
        messages = self._build_messages(message, history)
        try:
            response = await self._client.chat(
                model=self._model,
                messages=messages,
                options={"temperature": 0.0},
            )
            raw = response.message.content.strip()
            return self._parse_response(raw)
        except LLMError:
            raise
        except Exception as exc:
            raise LLMError(
                f"Intent classification failed: {exc}",
                model=self._model,
            ) from exc

    @staticmethod
    def _build_messages(message: str, history: list[Message]) -> list[dict]:
        """Build the Ollama messages list including recent history context."""
        msgs: list[dict] = [{"role": "system", "content": _ROUTER_SYSTEM_PROMPT}]
        for turn in history[-4:]:
            msgs.append({"role": turn.role.value, "content": turn.content})
        msgs.append({"role": "user", "content": message})
        return msgs

    @staticmethod
    def _parse_response(raw: str) -> IntentResult:
        """Parse JSON response; fall back to 'consulta' on any parse error."""
        cleaned = (
            raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        )
        try:
            data = json.loads(cleaned)
            intent = IntentType(data.get("intent", "consulta").lower())
            doc_type: str | None = data.get("doc_type") or None
            confidence = float(data.get("confidence", 0.8))
            return IntentResult(intent=intent, doc_type=doc_type, confidence=confidence)
        except (json.JSONDecodeError, ValueError, KeyError) as exc:
            logger.warning("Failed to parse intent response: %s — defaulting to 'consulta'", exc)
            return IntentResult(intent=IntentType.CONSULTA, confidence=0.5)
