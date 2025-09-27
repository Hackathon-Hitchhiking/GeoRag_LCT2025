"""Индекс признаков изображений для быстрого поиска."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from ..infrastructure.database.models import ImageRecord
from ..infrastructure.database.repositories import ImageRepository
from ..infrastructure.database.session import Database
from ..logging import get_logger
from .feature_store import FeatureStore, StoredFeature
from .features import LocalFeatureSet


@dataclass(slots=True)
class CachedImageEntry:
    """Запись кэша с дескрипторами и метаданными."""

    record: ImageRecord
    global_descriptor: np.ndarray
    local_features: LocalFeatureSet
    updated_at: datetime

    @property
    def id(self) -> int:
        return int(self.record.id)


class ImageFeatureIndex:
    """Долгоживущий кэш признаков изображений."""

    def __init__(
        self,
        *,
        database: Database,
        local_store: FeatureStore[LocalFeatureSet],
        refresh_interval: float = 7200.0,
    ) -> None:
        self._database = database
        self._local_store = local_store
        self._refresh_interval = max(refresh_interval, 60.0)
        self._log = get_logger("georag.index")

        self._entries: list[CachedImageEntry] = []
        self._matrix: np.ndarray | None = None

        self._lock = asyncio.Lock()
        self._refresh_lock = asyncio.Lock()
        self._stop_event = asyncio.Event()
        self._refresh_event = asyncio.Event()
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        """Инициализировать индекс и запустить фоновое обновление."""
        await self.refresh(force=True)
        self._task = asyncio.create_task(self._refresh_loop())
        self._log.info(
            "event=index_started entries=%s refresh_interval=%s",
            len(self._entries),
            self._refresh_interval,
        )

    async def stop(self) -> None:
        """Остановить фоновые задачи."""
        self._stop_event.set()
        self._refresh_event.set()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._log.info("event=index_stopped")

    async def request_refresh(self) -> None:
        """Запросить внеплановое обновление (например, после добавления записи)."""
        self._refresh_event.set()

    async def refresh(self, *, force: bool = False) -> None:
        """Обновить кэш записей, подгружая новые и изменённые."""
        if not force and self._refresh_lock.locked():
            self._log.debug("event=index_refresh_skipped reason=busy")
            return

        async with self._refresh_lock:
            async with self._database.session() as session:
                repo = ImageRepository(session)
                records = list(await repo.list_all())

            existing_by_id: dict[int, CachedImageEntry] = {
                entry.id: entry for entry in self._entries
            }
            updated_entries: dict[int, CachedImageEntry] = {}
            load_tasks: list[asyncio.Task[tuple[int, CachedImageEntry]]] = []

            for record in records:
                cached = existing_by_id.get(record.id)
                if (
                    cached is not None
                    and cached.record.updated_at == record.updated_at
                    and cached.record.image_key == record.image_key
                    and cached.record.feature_key == record.feature_key
                ):
                    cached.record = record
                    cached.updated_at = record.updated_at
                    updated_entries[record.id] = cached
                    continue
                load_tasks.append(asyncio.create_task(self._load_entry(record)))

            if load_tasks:
                self._log.info(
                    "event=index_loading_pending count=%s", len(load_tasks)
                )
                for task in asyncio.as_completed(load_tasks):
                    try:
                        record_id, entry = await task
                    except Exception as exc:  # pragma: no cover - пропускаем сбойные записи
                        self._log.exception(
                            "event=index_entry_load_error error=%s", exc
                        )
                        continue
                    updated_entries[record_id] = entry

            async with self._lock:
                self._entries = list(updated_entries.values())
                if self._entries:
                    matrix = np.stack(
                        [entry.global_descriptor for entry in self._entries]
                    ).astype(np.float32, copy=False)
                    self._matrix = matrix
                else:
                    self._matrix = None

            self._log.info(
                "event=index_refresh_completed entries=%s",
                len(self._entries),
            )

    async def _load_entry(self, record: ImageRecord) -> tuple[int, CachedImageEntry]:
        raw_descriptor = record.global_descriptor
        if not raw_descriptor:
            raise ValueError("Запись не содержит глобального дескриптора")

        try:
            stored: StoredFeature[LocalFeatureSet] = await self._local_store.load(
                record.feature_key
            )
        except Exception as exc:
            self._log.exception(
                "event=index_entry_load_failed record_id=%s error=%s",
                record.id,
                exc,
            )
            raise
        descriptor = np.frombuffer(raw_descriptor, dtype=np.float32)
        descriptor = descriptor.astype(np.float32, copy=True)
        entry = CachedImageEntry(
            record=record,
            global_descriptor=descriptor,
            local_features=stored.pack,
            updated_at=record.updated_at,
        )
        self._log.debug(
            "event=index_entry_loaded record_id=%s keypoints=%s",
            record.id,
            stored.pack.keypoints_count,
        )
        return record.id, entry

    async def score_by_global(
        self, query_descriptor: np.ndarray, *, limit: int | None = None
    ) -> list[tuple[CachedImageEntry, float]]:
        """Посчитать глобальное сходство с кэшированными записями."""
        query = query_descriptor.astype(np.float32, copy=False)
        async with self._lock:
            matrix = self._matrix
            entries = tuple(self._entries)

        if matrix is None or matrix.size == 0 or not entries:
            return []

        if query.shape[0] != matrix.shape[1]:
            self._log.warning(
                "event=index_query_mismatch query_dim=%s matrix_dim=%s",
                query.shape[0],
                matrix.shape[1],
            )
            return []

        similarities = matrix @ query
        similarities = np.clip(similarities, -1.0, 1.0)

        if limit is not None and limit > 0 and limit < similarities.shape[0]:
            top_idx = np.argpartition(similarities, -limit)[-limit:]
            top_idx = top_idx[np.argsort(similarities[top_idx])[::-1]]
        else:
            top_idx = np.argsort(similarities)[::-1]

        return [
            (entries[idx], float(similarities[idx]))
            for idx in top_idx
        ]

    async def get_entries(self) -> list[CachedImageEntry]:
        """Вернуть копию списка записей для внешних сценариев."""
        async with self._lock:
            return list(self._entries)

    async def _refresh_loop(self) -> None:
        """Фоновый цикл периодического обновления."""
        try:
            while not self._stop_event.is_set():
                try:
                    await asyncio.wait_for(
                        self._refresh_event.wait(), timeout=self._refresh_interval
                    )
                except asyncio.TimeoutError:
                    pass
                self._refresh_event.clear()
                if self._stop_event.is_set():
                    break
                try:
                    await self.refresh()
                except Exception as exc:  # pragma: no cover - фоновые ошибки
                    self._log.exception("event=index_refresh_failed error=%s", exc)
        finally:
            self._stop_event.set()
