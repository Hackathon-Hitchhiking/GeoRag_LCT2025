"""Сервис поиска по базе изображений с двухуровневым сопоставлением."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..infrastructure.database.models import ImageRecord
from ..infrastructure.database.repositories import ImageRepository
from ..infrastructure.database.session import Database
from ..infrastructure.storage.s3 import S3Storage
from ..utils.image import decode_image, draw_keypoints, encode_image
from .exceptions import EmptyDatabaseError, FeatureExtractionError, GeocodingError
from .feature_store import FeatureStore, StoredFeature
from .features import LocalFeatureSet, SuperPointFeatureExtractor
from .geocoder import Geocoder
from .global_descriptors import GlobalDescriptor, NetVLADGlobalExtractor
from .matcher import LightGlueMatcher, MatchScore


@dataclass(slots=True)
class SearchPayload:
    """Входные данные для поиска."""

    data: bytes
    plot_dots: bool
    top_k: int


@dataclass(slots=True)
class SearchMatchResult:
    """Совпадение из базы."""

    record: ImageRecord
    image_bytes: bytes
    local_features: LocalFeatureSet
    match_score: MatchScore
    global_similarity: float
    confidence: float


@dataclass(slots=True)
class SearchResult:
    """Результат поиска."""

    query_bytes: bytes
    query_local: LocalFeatureSet
    query_global: GlobalDescriptor
    matches: list[SearchMatchResult]


@dataclass(slots=True)
class CoordinateSearchPayload:
    """Параметры поиска ближайших изображений по координатам."""

    latitude: float
    longitude: float
    plot_dots: bool
    top_k: int


@dataclass(slots=True)
class AddressSearchPayload:
    """Параметры поиска ближайших изображений по адресу."""

    address: str
    plot_dots: bool
    top_k: int


@dataclass(slots=True)
class LocationMatchResult:
    """Описание совпадения при поиске по координатам или адресу."""

    record: ImageRecord
    image_bytes: bytes
    local_features: LocalFeatureSet | None
    distance_meters: float


@dataclass(slots=True)
class LocationSearchResult:
    """Список ближайших изображений."""

    matches: list[LocationMatchResult]


class ImageSearchService:
    """Высокоуровневый сервис поиска схожих изображений."""

    def __init__(
        self,
        *,
        database: Database,
        storage: S3Storage,
        local_store: FeatureStore[LocalFeatureSet],
        local_extractor: SuperPointFeatureExtractor,
        global_extractor: NetVLADGlobalExtractor,
        matcher: LightGlueMatcher,
        retrieval_candidates: int,
        global_weight: float,
        local_weight: float,
        max_results: int,
        geometry_weight: float,
        geocoder: Geocoder,
    ) -> None:
        self._database = database
        self._storage = storage
        self._local_store = local_store
        self._local_extractor = local_extractor
        self._global_extractor = global_extractor
        self._matcher = matcher
        self._retrieval_candidates = max(1, retrieval_candidates)
        self._global_weight = max(0.0, global_weight)
        self._local_weight = max(0.0, local_weight)
        self._geometry_weight = max(0.0, geometry_weight)
        self._geocoder = geocoder
        weight_norm = self._global_weight + self._local_weight + self._geometry_weight
        if weight_norm <= 0:
            self._global_weight = 0.35
            self._local_weight = 0.45
            self._geometry_weight = 0.20
            weight_norm = 1.0
        self._global_weight /= weight_norm
        self._local_weight /= weight_norm
        self._geometry_weight /= weight_norm
        self._max_results = max(1, max_results)

    async def search_by_image(self, payload: SearchPayload) -> SearchResult:
        """Выполнить поиск и вернуть лучшие совпадения."""
        query_image = decode_image(payload.data)
        try:
            query_local = await self._local_extractor.aextract(query_image)
            query_global = await self._global_extractor.aextract(query_image)
        except FeatureExtractionError:
            raise
        except Exception as exc:  # pragma: no cover - изоляция ошибок PyTorch/OpenCV
            raise FeatureExtractionError("Ошибка извлечения признаков запроса") from exc

        query_descriptor = query_global.normalized()

        async with self._database.session() as session:
            repo = ImageRepository(session)
            total = await repo.count()
            if total == 0:
                raise EmptyDatabaseError("В базе отсутствуют изображения")
            records = list(await repo.list_all())

        scored = self._score_by_global(records, query_descriptor)
        if not scored:
            return SearchResult(
                query_bytes=payload.data,
                query_local=query_local,
                query_global=query_global,
                matches=[],
            )

        candidate_limit = min(
            len(scored),
            max(self._retrieval_candidates, payload.top_k),
            self._max_results,
        )
        candidates = scored[:candidate_limit]

        matches: list[SearchMatchResult] = []
        for record, global_score in candidates:
            stored: StoredFeature[LocalFeatureSet] = await self._local_store.load(record.feature_key)
            candidate_features = stored.pack
            match_score = await self._matcher.amatch(query_local, candidate_features)
            confidence = self._aggregate_scores(global_score, match_score)
            if match_score.matches == 0:
                continue
            image_bytes = await self._storage.download(record.image_key)
            matches.append(
                SearchMatchResult(
                    record=record,
                    image_bytes=image_bytes,
                    local_features=candidate_features,
                    match_score=match_score,
                    global_similarity=global_score,
                    confidence=confidence,
                )
            )

        if not matches:
            return SearchResult(
                query_bytes=payload.data,
                query_local=query_local,
                query_global=query_global,
                matches=[],
            )

        matches.sort(key=lambda m: m.confidence, reverse=True)
        top_matches = matches[: payload.top_k]

        return SearchResult(
            query_bytes=payload.data,
            query_local=query_local,
            query_global=query_global,
            matches=top_matches,
        )

    async def search_by_coordinates(
        self, payload: CoordinateSearchPayload
    ) -> LocationSearchResult:
        """Найти ближайшие изображения относительно координат."""

        async with self._database.session() as session:
            repo = ImageRepository(session)
            total = await repo.count()
            if total == 0:
                raise EmptyDatabaseError("В базе отсутствуют изображения")
            records = list(await repo.list_all())

        scored: list[tuple[ImageRecord, float]] = []
        for record in records:
            if record.latitude is None or record.longitude is None:
                continue
            distance = self._haversine(
                payload.latitude, payload.longitude, record.latitude, record.longitude
            )
            scored.append((record, distance))

        if not scored:
            return LocationSearchResult(matches=[])

        scored.sort(key=lambda item: item[1])
        top_records = scored[: payload.top_k]

        matches: list[LocationMatchResult] = []
        for record, distance in top_records:
            image_bytes = await self._storage.download(record.image_key)
            local_features: LocalFeatureSet | None = None
            if payload.plot_dots:
                stored = await self._local_store.load(record.feature_key)
                local_features = stored.pack
            matches.append(
                LocationMatchResult(
                    record=record,
                    image_bytes=image_bytes,
                    local_features=local_features,
                    distance_meters=distance,
                )
            )

        return LocationSearchResult(matches=matches)

    async def search_by_address(self, payload: AddressSearchPayload) -> LocationSearchResult:
        """Найти ближайшие изображения по текстовому адресу."""

        coordinates = await self._geocoder.forward(payload.address)
        if coordinates is None:
            raise GeocodingError("Не удалось определить координаты по адресу")
        latitude, longitude = coordinates
        return await self.search_by_coordinates(
            CoordinateSearchPayload(
                latitude=latitude,
                longitude=longitude,
                plot_dots=payload.plot_dots,
                top_k=payload.top_k,
            )
        )

    def _score_by_global(
        self, records: list[ImageRecord], query_descriptor: np.ndarray
    ) -> list[tuple[ImageRecord, float]]:
        """Отсортировать записи по глобальному сходству."""
        query_descriptor = query_descriptor.astype(np.float32, copy=False)

        vectors: list[np.ndarray] = []
        valid_records: list[ImageRecord] = []
        expected_dim = int(query_descriptor.shape[0])
        for record in records:
            raw_descriptor = record.global_descriptor
            if not raw_descriptor:
                continue
            candidate = np.frombuffer(raw_descriptor, dtype=np.float32)
            if candidate.size != expected_dim or candidate.size == 0:
                continue
            vectors.append(candidate)
            valid_records.append(record)

        if not vectors:
            return []

        matrix = np.stack(vectors).astype(np.float32, copy=False)
        similarities = matrix @ query_descriptor
        similarities = np.clip(similarities, -1.0, 1.0)

        scored = list(zip(valid_records, similarities.tolist()))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored

    def _aggregate_scores(self, global_score: float, match_score: MatchScore) -> float:
        """Комбинировать все уровни оценки из пайплайна HLOC."""

        return (
            self._global_weight * global_score
            + self._local_weight * match_score.local_score
            + self._geometry_weight * match_score.geometric_strength
        )

    async def prepare_visual(self, features: LocalFeatureSet, image_bytes: bytes) -> bytes:
        """Построить визуализацию с ключевыми точками."""
        image = decode_image(image_bytes)
        keypoints = features.to_cv_keypoints()
        visual = draw_keypoints(image, keypoints)
        return encode_image(visual, ext="png")

    def _haversine(
        self, lat1: float, lon1: float, lat2: float, lon2: float
    ) -> float:
        """Посчитать расстояние по поверхности Земли в метрах."""

        radius = 6_371_000.0
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)
        a = (
            math.sin(delta_phi / 2) ** 2
            + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
        )
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return radius * c

    async def search(self, payload: SearchPayload) -> SearchResult:
        """Совместимость с предыдущим интерфейсом."""

        return await self.search_by_image(payload)
