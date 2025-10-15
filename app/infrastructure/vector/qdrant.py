"""Обёртка над Qdrant для хранения глобальных дескрипторов."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as rest


class QdrantVectorStore:
    """Минимальный клиент Qdrant, использующий cosine-сходство."""

    def __init__(
        self,
        *,
        url: str,
        api_key: str | None,
        collection: str,
        vector_dim: int,
        on_disk: bool,
        shard_number: int,
        replication_factor: int,
    ) -> None:
        self._client = AsyncQdrantClient(url=url, api_key=api_key)
        self._collection = collection
        self._dim = vector_dim
        self._on_disk = on_disk
        self._shards = shard_number
        self._replication = replication_factor

    @property
    def collection(self) -> str:
        return self._collection

    @property
    def dimension(self) -> int:
        return self._dim

    async def ensure_collection(self) -> None:
        collections = await self._client.get_collections()
        names = {item.name for item in collections.collections}
        if self._collection in names:
            return
        await self._client.create_collection(
            collection_name=self._collection,
            vectors_config=rest.VectorParams(
                size=self._dim,
                distance=rest.Distance.COSINE,
                on_disk=self._on_disk,
            ),
            shard_number=self._shards,
            replication_factor=self._replication,
            hnsw_config=rest.HnswConfigDiff(
                m=32,
                ef_construct=256,
                full_scan_threshold=1000,
            ),
            quantization_config=rest.ScalarQuantization(
                scalar=rest.ScalarQuantizationConfig(
                    type=rest.ScalarType.INT8,
                    always_ram=True,
                )
            ),
        )

    async def upsert(
        self,
        *,
        point_id: str,
        vector: np.ndarray,
        payload: dict[str, Any],
    ) -> None:
        normalized = vector.astype(np.float32, copy=False)
        if normalized.ndim != 1:
            raise ValueError("Vector must be 1-D")
        await self._client.upsert(
            collection_name=self._collection,
            points=[
                rest.PointStruct(
                    id=point_id,
                    vector=normalized.tolist(),
                    payload=payload,
                )
            ],
        )

    async def delete(self, point_ids: Sequence[str]) -> None:
        if not point_ids:
            return
        await self._client.delete(
            collection_name=self._collection,
            points_selector=rest.PointIdsList(points=list(point_ids)),
        )

    async def search(
        self, vector: np.ndarray, *, limit: int
    ) -> list[rest.ScoredPoint]:
        normalized = vector.astype(np.float32, copy=False)
        if normalized.ndim != 1:
            raise ValueError("Vector must be 1-D")
        result = await self._client.search(
            collection_name=self._collection,
            vector=normalized.tolist(),
            limit=limit,
            with_payload=True,
        )
        return list(result)

