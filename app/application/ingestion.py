"""Сервис загрузки изображений и вычисления признаков."""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from ..infrastructure.database.repositories import ImageRepository
from ..infrastructure.database.session import Database
from ..infrastructure.storage.s3 import S3Storage
from ..utils.image import decode_image, detect_extension
from .exceptions import DuplicateImageError, FeatureExtractionError, StorageError
from .feature_store import FeatureStore
from .features import LocalFeatureSet, SuperPointFeatureExtractor
from .geocoder import Geocoder
from .global_descriptors import GlobalDescriptor, NetVLADGlobalExtractor


@dataclass(slots=True)
class IngestionPayload:
    """Данные для добавления изображения."""

    data: bytes
    latitude: float | None = None
    longitude: float | None = None
    metadata: dict[str, Any] | None = None
    source_name: str | None = None


@dataclass(slots=True)
class StoredImage:
    """Результат успешного добавления изображения."""

    record_id: int
    image_key: str
    feature_key: str
    global_descriptor_key: str
    latitude: float | None
    longitude: float | None
    address: str | None
    metadata: dict[str, Any] | None
    descriptor_count: int
    descriptor_dim: int
    keypoint_count: int
    global_descriptor_dim: int
    created_at: datetime
    updated_at: datetime
    local_feature_type: str
    global_descriptor_type: str
    matcher_type: str


class ImageIngestionService:
    """Высокоуровневый сервис загрузки изображений."""

    def __init__(
        self,
        *,
        database: Database,
        storage: S3Storage,
        local_store: FeatureStore[LocalFeatureSet],
        global_store: FeatureStore[GlobalDescriptor],
        local_extractor: SuperPointFeatureExtractor,
        global_extractor: NetVLADGlobalExtractor,
        geocoder: Geocoder,
        image_prefix: str,
        local_feature_type: str,
        global_descriptor_type: str,
        matcher_type: str,
    ) -> None:
        self._database = database
        self._storage = storage
        self._local_store = local_store
        self._global_store = global_store
        self._local_extractor = local_extractor
        self._global_extractor = global_extractor
        self._geocoder = geocoder
        self._image_prefix = image_prefix.strip("/")
        self._local_feature_type = local_feature_type
        self._global_descriptor_type = global_descriptor_type
        self._matcher_type = matcher_type

    async def ingest(self, payload: IngestionPayload) -> StoredImage:
        """Добавить изображение в базу и вернуть описание."""
        digest = hashlib.sha256(payload.data).hexdigest()
        async with self._database.session() as session:
            repo = ImageRepository(session)
            existing = await repo.find_by_hash(digest)
            if existing is not None:
                raise DuplicateImageError("Изображение уже было загружено")
            record = await self._persist(payload, digest, session)
            await session.commit()
        return record

    async def _persist(
        self, payload: IngestionPayload, digest: str, session: AsyncSession
    ) -> StoredImage:
        image = decode_image(payload.data)
        try:
            local_features = await self._local_extractor.aextract(image)
            global_descriptor = await self._global_extractor.aextract(image)
        except FeatureExtractionError:
            raise
        except Exception as exc:  # pragma: no cover - изоляция ошибок OpenCV
            raise FeatureExtractionError("Ошибка извлечения дескрипторов") from exc

        stem = payload.source_name or uuid.uuid4().hex
        image_ext = detect_extension(payload.data)
        image_key = self._build_image_key(stem, image_ext)

        try:
            await self._storage.upload(
                key=image_key,
                data=payload.data,
                content_type=f"image/{image_ext}",
            )
        except Exception as exc:  # pragma: no cover - S3 ошибки
            raise StorageError("Не удалось сохранить изображение") from exc

        feature_key = await self._local_store.save(
            stem, local_features, metadata={"type": "local_features"}
        )
        global_key = await self._global_store.save(
            stem, global_descriptor, metadata={"type": "global_descriptor"}
        )
        address = await self._geocoder.reverse(payload.latitude, payload.longitude)

        values = {
            "image_key": image_key,
            "feature_key": feature_key,
            "global_descriptor_key": global_key,
            "preview_key": None,
            "latitude": payload.latitude,
            "longitude": payload.longitude,
            "address": address,
            "metadata": payload.metadata,
            "image_hash": digest,
            "descriptor_count": local_features.descriptors.shape[0],
            "descriptor_dim": local_features.descriptors.shape[1],
            "keypoint_count": local_features.keypoints_count,
            "global_descriptor_dim": global_descriptor.dimension,
            "global_descriptor": global_descriptor.to_database_blob(),
            "local_feature_type": self._local_feature_type,
            "global_descriptor_type": self._global_descriptor_type,
            "matcher_type": self._matcher_type,
        }
        repo = ImageRepository(session)
        record = await repo.create(values)
        return StoredImage(
            record_id=record.id,
            image_key=image_key,
            feature_key=feature_key,
            global_descriptor_key=global_key,
            latitude=record.latitude,
            longitude=record.longitude,
            address=record.address,
            metadata=record.metadata,
            descriptor_count=record.descriptor_count,
            descriptor_dim=record.descriptor_dim,
            keypoint_count=record.keypoint_count,
            global_descriptor_dim=record.global_descriptor_dim,
            created_at=record.created_at,
            updated_at=record.updated_at,
            local_feature_type=self._local_feature_type,
            global_descriptor_type=self._global_descriptor_type,
            matcher_type=self._matcher_type,
        )

    def _build_image_key(self, stem: str, extension: str) -> str:
        prefix = f"{self._image_prefix}/{stem}" if self._image_prefix else stem
        return f"{prefix}.{extension}"
