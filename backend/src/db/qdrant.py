"""
Qdrant client wrapper and collection helpers for Eureka Legal Agent.

Provides typed helpers for hybrid search (dense + sparse vectors)
and upsert operations. Raises VectorDBError on failures.
"""

from __future__ import annotations

import logging
from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    Fusion,
    MatchValue,
    NamedSparseVector,
    NamedVector,
    PointStruct,
    Prefetch,
    SparseIndexParams,
    SparseVectorParams,
    VectorParams,
    VectorsConfig,
)

from src.domain.errors import VectorDBError

logger = logging.getLogger(__name__)

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"
DENSE_VECTOR_SIZE = 1024


class QdrantStore:
    """
    Async wrapper around QdrantClient providing hybrid search and upsert.

    SRP: only responsible for vector storage and retrieval operations.
    All error wrapping happens here so callers deal with VectorDBError only.
    """

    def __init__(self, client: AsyncQdrantClient, collection_name: str) -> None:
        self._client = client
        self._collection = collection_name

    @classmethod
    def from_url(cls, url: str, collection_name: str) -> QdrantStore:
        client = AsyncQdrantClient(url=url)
        return cls(client, collection_name)

    async def ensure_collection(self) -> None:
        """Create the collection if it does not already exist."""
        try:
            exists = await self._client.collection_exists(self._collection)
            if not exists:
                await self._client.create_collection(
                    collection_name=self._collection,
                    vectors_config=VectorsConfig(
                        root={
                            DENSE_VECTOR_NAME: VectorParams(
                                size=DENSE_VECTOR_SIZE,
                                distance=Distance.COSINE,
                            )
                        }
                    ),
                    sparse_vectors_config={
                        SPARSE_VECTOR_NAME: SparseVectorParams(
                            index=SparseIndexParams(on_disk=False)
                        )
                    },
                )
                logger.info("Created Qdrant collection '%s'", self._collection)
        except UnexpectedResponse as exc:
            raise VectorDBError(
                f"Failed to ensure collection '{self._collection}'",
                details={"error": str(exc)},
            ) from exc

    async def upsert_chunk(
        self,
        point_id: str,
        dense_vector: list[float],
        sparse_indices: list[int],
        sparse_values: list[float],
        payload: dict[str, Any],
    ) -> None:
        """Insert or update a single chunk with hybrid vectors."""
        try:
            point = PointStruct(
                id=point_id,
                vector={
                    DENSE_VECTOR_NAME: dense_vector,
                    SPARSE_VECTOR_NAME: {
                        "indices": sparse_indices,
                        "values": sparse_values,
                    },
                },
                payload=payload,
            )
            await self._client.upsert(
                collection_name=self._collection,
                points=[point],
                wait=True,
            )
        except UnexpectedResponse as exc:
            raise VectorDBError(
                "Failed to upsert chunk",
                details={"point_id": point_id, "error": str(exc)},
            ) from exc

    async def hybrid_search(
        self,
        dense_query: list[float],
        sparse_indices: list[int],
        sparse_values: list[float],
        top_k: int = 20,
        filter_doc_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Hybrid search using Reciprocal Rank Fusion of dense + sparse results.
        Returns a list of payloads with an added 'score' field.
        """
        query_filter: Filter | None = None
        if filter_doc_type:
            query_filter = Filter(
                must=[FieldCondition(key="doc_type", match=MatchValue(value=filter_doc_type))]
            )

        try:
            results = await self._client.query_points(
                collection_name=self._collection,
                prefetch=[
                    Prefetch(
                        query=NamedVector(name=DENSE_VECTOR_NAME, vector=dense_query),
                        limit=top_k,
                        filter=query_filter,
                    ),
                    Prefetch(
                        query=NamedSparseVector(
                            name=SPARSE_VECTOR_NAME,
                            vector={"indices": sparse_indices, "values": sparse_values},
                        ),
                        limit=top_k,
                        filter=query_filter,
                    ),
                ],
                query=Fusion.RRF,
                limit=top_k,
            )

            return [
                {**point.payload, "score": point.score, "point_id": str(point.id)}
                for point in results.points
                if point.payload
            ]
        except UnexpectedResponse as exc:
            raise VectorDBError(
                "Hybrid search failed",
                details={"error": str(exc)},
            ) from exc
