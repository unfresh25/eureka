"""
RAG Chain: Retrieve → Rerank → Generate.

Combines hybrid search from Qdrant with BGE-ReRanker-V2-M3 (HuggingFace)
and Qwen2.5-VL for final answer generation.
"""

from __future__ import annotations

import logging
from typing import Any

import ollama
from sentence_transformers import CrossEncoder

from src.agent.embedder import Embedder
from src.config import Settings
from src.db.qdrant import QdrantStore
from src.domain.errors import LLMError, RerankerError
from src.domain.models import ConversationContext, RAGResult, SourceRef

logger = logging.getLogger(__name__)

_RAG_SYSTEM_PROMPT = """Eres un asistente legal colombiano experto y confiable.
Responde ÚNICAMENTE con base en el contexto legal proporcionado a continuación.
Si la respuesta no está en el contexto, dilo explícitamente: "No encontré información suficiente en 
la base de conocimiento para responder esta pregunta." Nunca inventes normas, artículos ni 
jurisprudencia.
Al final de tu respuesta, cita las fuentes usadas con el formato: "Fuentes: [título1], [título2]"
"""


class RAGChain:
    """
    Retrieve-Rerank-Generate pipeline for legal Q&A.

    SRP: responsible only for the RAG flow.
    DIP: depends on Embedder and QdrantStore abstractions.
    """

    def __init__(
        self,
        embedder: Embedder,
        vector_store: QdrantStore,
        reranker: CrossEncoder,  # type: ignore[type-arg]
        llm_client: ollama.AsyncClient,
        settings: Settings,
    ) -> None:
        self._embedder = embedder
        self._store = vector_store
        self._reranker = reranker
        self._llm = llm_client
        self._settings = settings

    async def query(
        self,
        question: str,
        ctx: ConversationContext,
        doc_type_filter: str | None = None,
    ) -> RAGResult:
        """Full RAG pipeline: embed → hybrid search → rerank → generate."""
        dense, sparse_i, sparse_v = await self._embedder.embed(question)

        candidates = await self._store.hybrid_search(
            dense_query=dense,
            sparse_indices=sparse_i,
            sparse_values=sparse_v,
            top_k=self._settings.retrieval_top_k,
            filter_doc_type=doc_type_filter,
        )

        if not candidates:
            return RAGResult(
                answer="No encontré información suficiente en la base de conocimiento para responder esta pregunta.",  # noqa: E501
                sources=[],
            )

        top_chunks = self._rerank(question, candidates)

        sources = [
            SourceRef(
                title=c.get("source_file", "Documento desconocido"),
                excerpt=c.get("text", "")[:200],
                doc_type=c.get("doc_type"),
            )
            for c in top_chunks
        ]

        context_text = self._build_context(top_chunks)
        messages = self._build_messages(question, context_text, ctx)
        answer = await self._generate(messages)

        return RAGResult(answer=answer, sources=sources)

    def _rerank(self, question: str, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Rerank candidates with BGE-ReRanker and return top-N."""
        try:
            pairs = [(question, c.get("text", "")) for c in candidates]
            scores: list[float] = self._reranker.predict(pairs).tolist()
            ranked = sorted(
                zip(candidates, scores, strict=True),
                key=lambda x: x[1],
                reverse=True,
            )
            return [c for c, _ in ranked[: self._settings.rerank_top_n]]
        except Exception as exc:
            raise RerankerError(f"Reranking failed: {exc}") from exc

    @staticmethod
    def _build_context(chunks: list[dict[str, Any]]) -> str:
        parts = []
        for i, chunk in enumerate(chunks, start=1):
            title = chunk.get("source_file", "Documento")
            text = chunk.get("text", "")
            parts.append(f"[{i}] {title}\n{text}")
        return "\n\n---\n\n".join(parts)

    @staticmethod
    def _build_messages(
        question: str,
        context_text: str,
        ctx: ConversationContext,
    ) -> list[dict]:
        msgs: list[dict] = [{"role": "system", "content": _RAG_SYSTEM_PROMPT}]
        memory_text = ctx.memory_as_text()
        if memory_text:
            msgs.append({"role": "system", "content": memory_text})
        for msg in ctx.recent_messages():
            msgs.append({"role": msg.role.value, "content": msg.content})
        user_content = f"Contexto legal recuperado:\n{context_text}\n\nPregunta: {question}"
        msgs.append({"role": "user", "content": user_content})
        return msgs

    async def _generate(self, messages: list[dict]) -> str:
        try:
            response = await self._llm.chat(
                model=self._settings.chat_model,
                messages=messages,
                options={"temperature": 0.2},
            )
            return response.message.content.strip()
        except Exception as exc:
            raise LLMError(
                f"Answer generation failed: {exc}",
                model=self._settings.chat_model,
            ) from exc
