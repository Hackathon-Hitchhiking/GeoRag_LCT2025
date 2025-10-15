"""LRU-кеш локальных признаков с ограничением по памяти."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Awaitable, Callable
from typing import TypeVar

from cachetools import LRUCache

from .features import LocalFeatureSet

T = TypeVar("T", bound=LocalFeatureSet)


def _feature_size(features: LocalFeatureSet) -> int:
    return int(
        features.keypoints.nbytes
        + features.descriptors.nbytes
        + features.scores.nbytes
    )


class LocalFeatureCache:
    """Асинхронный LRU-кеш локальных дескрипторов."""

    def __init__(self, *, max_bytes: int, max_items: int) -> None:
        self._cache: LRUCache[str, LocalFeatureSet] = LRUCache(
            maxsize=max(max_bytes, 1), getsizeof=_feature_size
        )
        self._locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self._max_items = max_items
        self._order_lock = asyncio.Lock()

    async def get_or_load(
        self, key: str, loader: Callable[[], Awaitable[T]]
    ) -> LocalFeatureSet:
        async with self._order_lock:
            cached = self._cache.get(key)
            if cached is not None:
                return cached
            lock = self._locks[key]

        async with lock:
            async with self._order_lock:
                cached = self._cache.get(key)
                if cached is not None:
                    return cached
            loaded = await loader()
            async with self._order_lock:
                self._cache[key] = loaded
                while len(self._cache) > self._max_items:
                    self._cache.popitem()
                self._locks.pop(key, None)
            return loaded

    async def invalidate(self, key: str) -> None:
        async with self._order_lock:
            self._cache.pop(key, None)
            self._locks.pop(key, None)

    async def clear(self) -> None:
        async with self._order_lock:
            self._cache.clear()
            self._locks.clear()

