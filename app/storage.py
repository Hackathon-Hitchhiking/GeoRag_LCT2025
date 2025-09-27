"""Abstractions for object storage interactions."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import boto3
from botocore.exceptions import BotoCoreError, ClientError

LOG = logging.getLogger("medical.storage")


class S3Storage:
    """Simple async wrapper over boto3 for storing binary blobs."""

    def __init__(
        self,
        *,
        bucket: str,
        endpoint_url: str | None = None,
        region_name: str | None = None,
        access_key: str | None = None,
        secret_key: str | None = None,
        use_ssl: bool = True,
    ) -> None:
        session = boto3.session.Session()
        self._client = session.client(
            "s3",
            endpoint_url=endpoint_url,
            region_name=region_name,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            use_ssl=use_ssl,
        )
        self._bucket = bucket

    async def upload(
        self,
        *,
        key: str,
        data: bytes,
        content_type: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> str:
        """Upload a binary payload and return its object key."""
        if not data:
            raise ValueError("Cannot upload empty payload to S3")

        extra: dict[str, Any] = {}
        if content_type:
            extra["ContentType"] = content_type
        if metadata:
            extra["Metadata"] = metadata

        def _put() -> None:
            try:
                self._client.put_object(
                    Bucket=self._bucket, Key=key, Body=data, **extra
                )
            except (BotoCoreError, ClientError):  # pragma: no cover - network path
                LOG.exception("event=s3_put_failed key=%s", key)
                raise

        await asyncio.to_thread(_put)
        return key

    async def delete(self, key: str) -> None:
        """Delete an object if it exists."""

        def _delete() -> None:
            try:
                self._client.delete_object(Bucket=self._bucket, Key=key)
            except (BotoCoreError, ClientError):  # pragma: no cover - network path
                LOG.exception("event=s3_delete_failed key=%s", key)
                raise

        await asyncio.to_thread(_delete)

    def build_path(self, key: str) -> str:
        """Return a canonical S3 URI for the given object key."""
        return f"s3://{self._bucket}/{key}"
