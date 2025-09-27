"""Инфраструктурный слой для сохранения дескрипторов в S3."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

from ..infrastructure.storage.s3 import S3Storage
from .exceptions import StorageError

T = TypeVar("T")


@dataclass(slots=True)
class StoredFeature(Generic[T]):
    """Результат чтения признаков из хранилища."""

    pack: T
    key: str
    metadata: dict[str, str] | None = None


class FeatureStore(Generic[T]):
    """Управление сохранением и загрузкой дескрипторов."""

    def __init__(
        self,
        storage: S3Storage,
        *,
        prefix: str,
        serializer: Callable[[T], bytes],
        deserializer: Callable[[bytes], T],
    ) -> None:
        self._storage = storage
        self._prefix = prefix.strip("/")
        self._serializer = serializer
        self._deserializer = deserializer

    def build_key(self, stem: str, *, extension: str = "npz") -> str:
        """Построить ключ для хранения."""
        stem_clean = stem.strip("/")
        if self._prefix:
            return f"{self._prefix}/{stem_clean}.{extension}"
        return f"{stem_clean}.{extension}"

    async def save(
        self, stem: str, pack: T, *, metadata: dict[str, str] | None = None
    ) -> str:
        """Сохранить набор дескрипторов и вернуть ключ."""
        try:
            payload = await asyncio.to_thread(self._serializer, pack)
            key = self.build_key(stem)
            await self._storage.upload(
                key=key,
                data=payload,
                content_type="application/octet-stream",
                metadata=metadata or {"type": "features"},
            )
        except Exception as exc:  # pragma: no cover - реальный S3
            raise StorageError("Не удалось сохранить признаки") from exc
        return key

    async def load(self, key: str) -> StoredFeature[T]:
        """Загрузить дескрипторы по ключу."""
        try:
            payload = await self._storage.download(key)
            pack = await asyncio.to_thread(self._deserializer, payload)
        except Exception as exc:  # pragma: no cover - реальный S3
            raise StorageError(f"Не удалось загрузить признаки {key}") from exc
        return StoredFeature(pack=pack, key=key)

    def build_uri(self, key: str) -> str:
        """Получить полный URI до объекта."""
        return self._storage.build_path(key)
