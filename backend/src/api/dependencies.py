"""
FastAPI dependency injection container.

All services are instantiated once (singleton) and injected into route
handlers via Depends(). This keeps route handlers thin and testable.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import ollama
from sentence_transformers import CrossEncoder

from src.agent.doc_generator import (
    DemandaStrategy,
    DerechoPeticionStrategy,
    DocGenerator,
    TutelaStrategy,
)
from src.agent.embedder import BGEEmbedder
from src.agent.intent_router import OllamaIntentRouter
from src.agent.memory_manager import MemoryManager
from src.agent.rag_chain import RAGChain
from src.config import Settings, get_settings
from src.db.qdrant import QdrantStore
from src.db.supabase import SupabaseRepository


@lru_cache
def _get_settings() -> Settings:
    return get_settings()


@lru_cache
def _get_llm_client() -> ollama.AsyncClient:
    s = _get_settings()
    return ollama.AsyncClient(host=s.ollama_base_url)


@lru_cache
def _get_embedder() -> BGEEmbedder:
    s = _get_settings()
    return BGEEmbedder(model=s.embed_model, ollama_host=s.ollama_base_url)


@lru_cache
def _get_reranker() -> CrossEncoder:  # type: ignore[type-arg]
    s = _get_settings()
    return CrossEncoder(s.reranker_model)


@lru_cache
def _get_qdrant_store() -> QdrantStore:
    s = _get_settings()
    return QdrantStore.from_url(s.qdrant_url, s.qdrant_collection)


@lru_cache
def _get_supabase_repo() -> SupabaseRepository:
    s = _get_settings()
    return SupabaseRepository.from_credentials(
        url=s.supabase_url,
        key=s.supabase_key.get_secret_value(),
    )


@lru_cache
def _get_rag_chain() -> RAGChain:
    return RAGChain(
        embedder=_get_embedder(),
        vector_store=_get_qdrant_store(),
        reranker=_get_reranker(),
        llm_client=_get_llm_client(),
        settings=_get_settings(),
    )


@lru_cache
def _get_doc_generator() -> DocGenerator:
    return DocGenerator(
        strategies=[TutelaStrategy(), DemandaStrategy(), DerechoPeticionStrategy()],
        rag_chain=_get_rag_chain(),
        llm_client=_get_llm_client(),
        settings=_get_settings(),
        output_dir=Path("generated_docs"),
    )


@lru_cache
def _get_intent_router() -> OllamaIntentRouter:
    s = _get_settings()
    return OllamaIntentRouter(model=s.chat_model, ollama_host=s.ollama_base_url)


@lru_cache
def _get_memory_manager() -> MemoryManager:
    return MemoryManager(
        repo=_get_supabase_repo(),
        llm_client=_get_llm_client(),
        settings=_get_settings(),
    )


def get_settings_dep() -> Settings:
    return _get_settings()


def get_rag_chain() -> RAGChain:
    return _get_rag_chain()


def get_doc_generator() -> DocGenerator:
    return _get_doc_generator()


def get_intent_router() -> OllamaIntentRouter:
    return _get_intent_router()


def get_memory_manager() -> MemoryManager:
    return _get_memory_manager()


def get_supabase_repo() -> SupabaseRepository:
    return _get_supabase_repo()
