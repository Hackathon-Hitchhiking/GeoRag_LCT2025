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
from ..infrastructure.storage import S3Storage
from ..infrastructure.vector.qdrant import QdrantVectorStore
from ..infrastructure.storage import S3Storage
from ..infrastructure.vector.qdrant import QdrantVectorStore
from ..logging import get_logger
from ..utils.image import decode_image, detect_extension
from .exceptions import DuplicateImageError, FeatureExtractionError, StorageError
from .feature_store import FeatureStore
from .features import LocalFeatureSet, SuperPointFeatureExtractor
from .geocoder import Geocoder
from .global_descriptors import NetVLADGlobalExtractor
from .global_descriptors import NetVLADGlobalExtractor


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
    image_path: str
    image_url: str
    feature_path: str
    vector_id: str
    image_path: str
    image_url: str
    feature_path: str
    vector_id: str
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
        local_extractor: SuperPointFeatureExtractor,
        global_extractor: NetVLADGlobalExtractor,
        geocoder: Geocoder,
        vector_store: QdrantVectorStore,
        vector_store: QdrantVectorStore,
        image_prefix: str,
        local_feature_type: str,
        global_descriptor_type: str,
        matcher_type: str,
    ) -> None:
        self._database = database
        self._storage = storage
        self._local_store = local_store
        self._local_extractor = local_extractor
        self._global_extractor = global_extractor
        self._geocoder = geocoder
        self._image_prefix = image_prefix.strip("/")
        self._local_feature_type = local_feature_type
        self._global_descriptor_type = global_descriptor_type
        self._matcher_type = matcher_type
        self._log = get_logger("georag.ingestion")
        self._vector_store = vector_store
        self._vector_store = vector_store

    async def ingest(self, payload: IngestionPayload) -> StoredImage:
        """Добавить изображение в базу и вернуть описание."""
        digest = hashlib.sha256(payload.data).hexdigest()
        self._log.info("event=ingest_started digest=%s size_bytes=%s", digest[:16], len(payload.data))
        async with self._database.session() as session:
            repo = ImageRepository(session)
            existing = await repo.find_by_hash(digest)
            if existing is not None:
                self._log.info(
                    "event=ingest_duplicate digest=%s existing_id=%s",
                    digest[:16],
                    existing.id,
                )
                raise DuplicateImageError("Изображение уже было загружено")
            record = await self._persist(payload, digest, session)
            await session.commit()
        self._log.info("event=ingest_completed record_id=%s digest=%s", record.record_id, digest[:16])
        return record

    async def _persist(
        self, payload: IngestionPayload, digest: str, session: AsyncSession
    ) -> StoredImage:
        image = decode_image(payload.data)
        try:
            local_features = await self._local_extractor.aextract(image)
            global_descriptor = await self._global_extractor.aextract(image)
            self._log.debug(
                "event=features_extracted digest=%s keypoints=%s descriptor_dim=%s",
                digest[:16],
                local_features.keypoints_count,
                local_features.descriptor_dim,
            )
        except FeatureExtractionError:
            self._log.warning("event=feature_extraction_failed digest=%s", digest[:16])
            raise
        except Exception as exc:  # pragma: no cover - изоляция ошибок OpenCV
            self._log.exception("event=feature_extraction_unexpected digest=%s", digest[:16])
            raise FeatureExtractionError("Ошибка извлечения дескрипторов") from exc

        stem = payload.source_name or uuid.uuid4().hex
        unique_suffix = digest[:12]
        artifact_stem = f"{stem}-{unique_suffix}"
        image_ext = detect_extension(payload.data)
        image_key = self._build_image_key(artifact_stem, image_ext)
        image_key = self._build_image_key(artifact_stem, image_ext)

        content_type = f"image/{image_ext}" if image_ext else "application/octet-stream"
        try:
            await self._storage.save(
                image_key,
                payload.data,
                content_type=content_type,
                metadata={"digest": digest},
            )
        except Exception as exc:  # pragma: no cover - IO errors
        except Exception as exc:  # pragma: no cover - IO errors
            self._log.exception(
                "event=image_upload_failed digest=%s image_key=%s",
                digest[:16],
                image_key,
            )
            raise StorageError("Не удалось сохранить изображение") from exc

        feature_key = await self._local_store.save(
            f"{artifact_stem}-local",
            local_features,
            metadata={"type": "local_features", "digest": digest[:16]},
            f"{artifact_stem}-local",
            local_features,
            metadata={"type": "local_features", "digest": digest[:16]},
        )
        image_url = self._storage.build_url(image_key)
        image_url = self._storage.build_url(image_key)
        self._log.debug(
            "event=artifacts_saved digest=%s image_key=%s local_key=%s",
            "event=artifacts_saved digest=%s image_key=%s local_key=%s",
            digest[:16],
            image_key,
            feature_key,
        )
        address = await self._geocoder.reverse(payload.latitude, payload.longitude)

        vector_id = uuid.uuid4().hex
        vector_id = uuid.uuid4().hex
        values = {
            "vector_id": vector_id,
            "image_path": image_key,
            "local_feature_path": feature_key,
            "preview_path": None,
            "vector_id": vector_id,
            "image_path": image_key,
            "local_feature_path": feature_key,
            "preview_path": None,
            "latitude": payload.latitude,
            "longitude": payload.longitude,
            "address": address,
            "metadata_json": payload.metadata,
            "image_hash": digest,
            "descriptor_count": local_features.descriptors.shape[0],
            "descriptor_dim": local_features.descriptors.shape[1],
            "keypoint_count": local_features.keypoints_count,
            "global_descriptor_dim": global_descriptor.dimension,
            "local_feature_type": self._local_feature_type,
            "global_descriptor_type": self._global_descriptor_type,
            "matcher_type": self._matcher_type,
        }
        repo = ImageRepository(session)
        record = await repo.create(values)

        vector_payload = {
            "record_id": int(record.id),
            "image_path": image_key,
            "feature_path": feature_key,
            "latitude": payload.latitude,
            "longitude": payload.longitude,
            "descriptor_dim": local_features.descriptor_dim,
            "keypoints": local_features.keypoints_count,
        }
        await self._vector_store.upsert(
            point_id=vector_id,
            vector=global_descriptor.normalized(),
            payload=vector_payload,
        )

        vector_payload = {
            "record_id": int(record.id),
            "image_path": image_key,
            "feature_path": feature_key,
            "latitude": payload.latitude,
            "longitude": payload.longitude,
            "descriptor_dim": local_features.descriptor_dim,
            "keypoints": local_features.keypoints_count,
        }
        await self._vector_store.upsert(
            point_id=vector_id,
            vector=global_descriptor.normalized(),
            payload=vector_payload,
        )
        return StoredImage(
            record_id=record.id,
            image_path=image_key,
            image_url=image_url,
            feature_path=feature_key,
            vector_id=vector_id,
            image_path=image_key,
            image_url=image_url,
            feature_path=feature_key,
            vector_id=vector_id,
            latitude=record.latitude,
            longitude=record.longitude,
            address=record.address,
            metadata=record.metadata_json,
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
