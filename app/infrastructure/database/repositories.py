"""Репозитории доступа к данным на уровне инфраструктуры."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from sqlalchemy import Select, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from .models import ImageRecord


class RepositoryError(RuntimeError):
    """Базовое исключение для ошибок уровня репозитория."""


class ImageRepository:
    """Репозиторий для работы с таблицей изображений."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(self, values: dict[str, Any]) -> ImageRecord:
        """Создать запись об изображении."""
        record = ImageRecord(**values)
        self._session.add(record)
        try:
            await self._session.flush()
        except IntegrityError as exc:  # pragma: no cover - реальный БД конфликт
            raise RepositoryError("Не удалось сохранить запись") from exc
        return record

    async def list_all(self) -> Sequence[ImageRecord]:
        """Вернуть все изображения в порядке добавления."""
        stmt: Select[tuple[ImageRecord]] = select(ImageRecord).order_by(ImageRecord.id)
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def get_by_ids(self, ids: Sequence[int]) -> Sequence[ImageRecord]:
        if not ids:
            return []
        stmt = select(ImageRecord).where(ImageRecord.id.in_(ids))
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def find_by_hash(self, image_hash: str) -> ImageRecord | None:
        """Найти запись по хэшу изображения."""
        stmt = select(ImageRecord).where(ImageRecord.image_hash == image_hash)
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def count(self) -> int:
        """Подсчитать количество изображений в базе."""
        stmt = select(func.count(ImageRecord.id))
        result = await self._session.execute(stmt)
        count_value = result.scalar_one()
        return int(count_value or 0)
