"""Асинхронное хранилище поверх S3-совместимого API."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Iterable, Mapping, MutableMapping, Sequence

import boto3
from botocore.client import BaseClient
from botocore.config import Config as BotoConfig


def _sanitize_key(key: str) -> str:
    normalized = key.strip().strip("/")
    if not normalized:
        raise ValueError("storage key must not be empty")
    return normalized


@dataclass(slots=True)
class S3Object:
    """Результат пакетного чтения объекта."""

    key: str
    payload: bytes


class S3Storage:
    """Минимальная обёртка вокруг boto3 с асинхронным API."""

    def __init__(
        self,
        *,
        bucket: str,
        region: str | None = None,
        endpoint_url: str | None = None,
        access_key: str | None = None,
        secret_key: str | None = None,
        session_token: str | None = None,
        public_base_url: str | None = None,
        presign_ttl: int = 3600,
        max_parallel: int = 16,
    ) -> None:
        if not bucket:
            raise ValueError("S3 bucket must be provided")
        self._bucket = bucket
        self._public_base_url = (
            public_base_url.rstrip("/") if public_base_url else None
        )
        self._presign_ttl = presign_ttl
        self._max_parallel = max(1, max_parallel)
        config = BotoConfig(max_pool_connections=max_parallel)
        session = boto3.session.Session()
        self._client: BaseClient = session.client(
            "s3",
            region_name=region,
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            aws_session_token=session_token,
            config=config,
        )

    @property
    def bucket(self) -> str:
        return self._bucket

    async def save(
        self,
        key: str,
        data: bytes,
        *,
        content_type: str | None = None,
        metadata: Mapping[str, str] | None = None,
    ) -> str:
        if not data:
            raise ValueError("storage payload is empty")
        object_key = _sanitize_key(key)

        def _upload() -> None:
            extra: MutableMapping[str, object] = {"Bucket": self._bucket, "Key": object_key, "Body": data}
            if content_type:
                extra["ContentType"] = content_type
            if metadata:
                extra["Metadata"] = dict(metadata)
            self._client.put_object(**extra)

        await asyncio.to_thread(_upload)
        return object_key

    async def load(self, key: str) -> bytes:
        object_key = _sanitize_key(key)

        def _download() -> bytes:
            response = self._client.get_object(Bucket=self._bucket, Key=object_key)
            body = response.get("Body")
            if body is None:
                raise FileNotFoundError(f"Object {object_key} has no body")
            return body.read()

        return await asyncio.to_thread(_download)

    async def load_many(self, keys: Sequence[str]) -> list[S3Object]:
        """Загрузить несколько объектов параллельно."""

        semaphore = asyncio.Semaphore(self._max_parallel)

        async def _fetch(key: str) -> S3Object:
            async with semaphore:
                payload = await self.load(key)
                return S3Object(key=key, payload=payload)

        tasks = [asyncio.create_task(_fetch(_sanitize_key(key))) for key in keys]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        objects: list[S3Object] = []
        for item in results:
            if isinstance(item, Exception):
                raise item
            objects.append(item)
        return objects

    async def delete(self, key: str) -> None:
        object_key = _sanitize_key(key)

        def _remove() -> None:
            self._client.delete_object(Bucket=self._bucket, Key=object_key)

        await asyncio.to_thread(_remove)

    async def delete_many(self, keys: Iterable[str]) -> None:
        payload = {
            "Bucket": self._bucket,
            "Delete": {"Objects": [{"Key": _sanitize_key(key)} for key in keys]},
        }

        def _bulk_delete() -> None:
            self._client.delete_objects(**payload)

        await asyncio.to_thread(_bulk_delete)

    def build_url(self, key: str, *, expires_in: int | None = None) -> str:
        object_key = _sanitize_key(key)
        if self._public_base_url:
            return f"{self._public_base_url}/{object_key}"
        ttl = expires_in or self._presign_ttl
        return self._client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self._bucket, "Key": object_key},
            ExpiresIn=ttl,
        )

