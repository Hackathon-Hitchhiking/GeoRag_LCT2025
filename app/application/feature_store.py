"""Инфраструктурный слой для сохранения дескрипторов на диске."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

from ..infrastructure.storage import FileSystemStorage, S3Storage
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
        storage: FileSystemStorage | S3Storage,
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
            await self._storage.save(
                key,
                payload,
                content_type="application/octet-stream",
                metadata=metadata or {},
            )
        except Exception as exc:  # pragma: no cover - IO errors
            raise StorageError("Не удалось сохранить признаки") from exc
        return key

    async def load(self, key: str) -> StoredFeature[T]:
        """Загрузить дескрипторы по ключу."""
        try:
            payload = await self._storage.load(key)
            pack = await asyncio.to_thread(self._deserializer, payload)
        except Exception as exc:  # pragma: no cover - IO errors
            raise StorageError(f"Не удалось загрузить признаки {key}") from exc
        return StoredFeature(pack=pack, key=key)

    async def load_many(self, keys: list[str]) -> list[StoredFeature[T]]:
        """Загрузить несколько наборов дескрипторов параллельно."""
        if not keys:
            return []
        try:
            batches = await self._storage.load_many(keys)
            results: list[StoredFeature[T]] = []
            for item in batches:
                pack = await asyncio.to_thread(self._deserializer, item.payload)
                results.append(StoredFeature(pack=pack, key=item.key))
        except Exception as exc:  # pragma: no cover - IO errors
            raise StorageError("Не удалось загрузить набор признаков") from exc
        return results

    def build_uri(self, key: str) -> str:
        """Получить полный URI до объекта."""
        return self._storage.build_url(key)
