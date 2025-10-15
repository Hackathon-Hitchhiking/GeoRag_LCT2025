"""Простое асинхронное файловое хранилище для артефактов."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Final


def _sanitize_key(key: str) -> str:
    key = key.strip().strip("/")
    if not key:
        raise ValueError("storage key must not be empty")
    return key


class FileSystemStorage:
    """Хранение бинарных объектов на локальном или сетевом диске."""

    def __init__(self, root: str | Path) -> None:
        path = Path(root).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        self._root: Final[Path] = path

    @property
    def root(self) -> Path:
        return self._root

    def _resolve(self, key: str, *, create_parents: bool = False) -> Path:
        normalized = _sanitize_key(key)
        path = self._root / normalized
        if create_parents:
            path.parent.mkdir(parents=True, exist_ok=True)
        return path

    async def save(
        self,
        key: str,
        data: bytes,
        *,
        content_type: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> str:
        if not data:
            raise ValueError("storage payload is empty")
        path = self._resolve(key, create_parents=True)

        def _write() -> None:
            path.write_bytes(data)

        await asyncio.to_thread(_write)
        return key

    async def load(self, key: str) -> bytes:
        path = self._resolve(key)

        def _read() -> bytes:
            if not path.exists():
                raise FileNotFoundError(f"storage object {key} is missing")
            return path.read_bytes()

        return await asyncio.to_thread(_read)

    async def load_many(self, keys: list[str]) -> list["S3Object"]:
        from .s3 import S3Object  # локальный импорт для избежания циклов

        async def _load_one(item_key: str) -> S3Object:
            payload = await self.load(item_key)
            return S3Object(key=item_key, payload=payload)

        tasks = [asyncio.create_task(_load_one(key)) for key in keys]
        results = await asyncio.gather(*tasks)
        return list(results)

    async def delete(self, key: str) -> None:
        path = self._resolve(key)

        def _remove() -> None:
            try:
                path.unlink()
            except FileNotFoundError:
                return

        await asyncio.to_thread(_remove)

    def build_url(self, key: str) -> str:
        return str(self._resolve(key))

