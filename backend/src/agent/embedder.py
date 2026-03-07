"""
Embedder: generates dense and sparse vectors for hybrid search using BGE-M3.

BGE-M3 via Ollama handles dense embeddings.
Sparse (BM25-style) vectors are computed locally using a simple TF-IDF
approximation (sufficient for the MVP) since Ollama does not expose sparse
embeddings natively.

DIP: callers depend on the `Embedder` Protocol, not this concrete class.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Protocol, runtime_checkable

import ollama

from src.domain.errors import LLMError

logger = logging.getLogger(__name__)


@runtime_checkable
class Embedder(Protocol):
    """Interface for generating dense and sparse vectors from text."""

    async def embed(self, text: str) -> tuple[list[float], list[int], list[float]]:
        """Return (dense_vector, sparse_indices, sparse_values)."""
        ...


class BGEEmbedder:
    """
    Concrete embedder using BGE-M3 (dense) + local BM25 (sparse).

    SRP: only responsible for producing embedding vectors.
    """

    def __init__(self, model: str = "bge-m3", ollama_host: str = "http://localhost:11434") -> None:
        self._model = model
        self._client = ollama.AsyncClient(host=ollama_host)

    async def embed(self, text: str) -> tuple[list[float], list[int], list[float]]:
        """Embed text into dense + sparse vectors."""
        dense = await self._embed_dense(text)
        sparse_indices, sparse_values = self._embed_sparse(text)
        return dense, sparse_indices, sparse_values

    async def _embed_dense(self, text: str) -> list[float]:
        try:
            response = await self._client.embeddings(model=self._model, prompt=text)
            return response.embedding  # type: ignore[attr-defined]
        except Exception as exc:
            raise LLMError(
                f"Dense embedding failed: {exc}",
                model=self._model,
                details={"text_preview": text[:100]},
            ) from exc

    @staticmethod
    def _embed_sparse(text: str) -> tuple[list[int], list[float]]:
        """
        Simple BM25-inspired sparse vector.

        Tokens are hashed to a fixed-size vocabulary (2^16 = 65536 buckets).
        This is a lightweight MVP approximation — can be swapped for SPLADE later.
        """
        vocab_size = 65536
        tokens = re.findall(r"\b\w+\b", text.lower())
        if not tokens:
            return [], []

        counts = Counter(tokens)
        tf: dict[int, float] = {}
        for token, count in counts.items():
            bucket = hash(token) % vocab_size
            # TF = count / total_tokens
            tf[bucket] = tf.get(bucket, 0) + count / len(tokens)

        indices = sorted(tf.keys())
        values = [tf[i] for i in indices]
        return indices, values
