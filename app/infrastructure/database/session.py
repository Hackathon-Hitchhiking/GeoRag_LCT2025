"""Создание и управление подключением к БД."""

from __future__ import annotations

from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from .models import Base


class Database:
    """Инкапсулирует создание подключения к PostgreSQL."""

    def __init__(self, dsn: str, *, echo: bool = False) -> None:
        self.engine = create_async_engine(dsn, echo=echo, future=True)
        self.session_factory = async_sessionmaker(
            self.engine, expire_on_commit=False, class_=AsyncSession
        )

    async def init_models(self) -> None:
        """Создать отсутствующие таблицы."""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    @asynccontextmanager
    async def session(self) -> AsyncSession:
        """Асинхронный контекст для работы с сессией."""
        session = self.session_factory()
        try:
            yield session
        finally:
            await session.close()

    async def aclose(self) -> None:
        await self.engine.dispose()
