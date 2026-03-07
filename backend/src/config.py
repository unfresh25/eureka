"""
Application settings loaded from environment variables via pydantic-settings.

Usage:
    from src.config import get_settings
    settings = get_settings()

All env vars must be prefixed with EUREKA_ (e.g. EUREKA_OLLAMA_BASE_URL).
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="EUREKA_",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    ollama_base_url: str = "http://localhost:11434"
    chat_model: str = "qwen2.5vl:7b"
    embed_model: str = "bge-m3"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"

    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "legal_kb"
    supabase_url: str
    supabase_key: SecretStr

    retrieval_top_k: int = 20
    rerank_top_n: int = 5

    chunk_size: int = 512
    chunk_overlap: int = 64

    api_title: str = "Eureka Legal Agent"
    api_version: str = "1.0.0"
    cors_origins: list[str] = ["http://localhost:3000", "http://127.0.0.1:5500"]

    history_window: int = 10
    summary_every_n: int = 20


@lru_cache
def get_settings() -> Settings:
    """Singleton accessor for Settings. Cache avoids re-parsing .env on every call."""
    return Settings()
