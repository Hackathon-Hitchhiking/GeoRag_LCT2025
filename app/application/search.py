"""Сервис поиска по базе изображений с двухуровневым сопоставлением."""

from __future__ import annotations

import asyncio
import heapq
import math
import uuid
from dataclasses import dataclass

from ..infrastructure.database.models import ImageRecord
from ..infrastructure.database.repositories import ImageRepository
from ..infrastructure.database.session import Database
from ..infrastructure.storage import S3Storage
from ..infrastructure.vector.qdrant import QdrantVectorStore
from ..logging import get_logger
from ..utils.image import decode_image, detect_extension
from .exceptions import EmptyDatabaseError, FeatureExtractionError, GeocodingError
from .feature_store import FeatureStore
from .features import LocalFeatureSet, SuperPointFeatureExtractor
from .geocoder import Geocoder
from .global_descriptors import GlobalDescriptor, NetVLADGlobalExtractor
from .local_cache import LocalFeatureCache
from .matcher import LightGlueMatcher, MatchScore


@dataclass(slots=True)
class SearchPayload:
    """Входные данные для поиска."""

    data: bytes
    plot_dots: bool
    top_k: int
    image_key: str | None = None
    image_url: str | None = None


@dataclass(slots=True)
class SearchMatchResult:
    """Совпадение из базы."""

    record: ImageRecord
    image_url: str
    local_features: LocalFeatureSet
    match_score: MatchScore
    global_similarity: float
    confidence: float


@dataclass(slots=True)
class SearchResult:
    """Результат поиска."""

    query_image_key: str
    query_image_url: str
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
    image_url: str
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
        vector_store: QdrantVectorStore,
        feature_cache: LocalFeatureCache,
        retrieval_candidates: int,
        global_weight: float,
        local_weight: float,
        max_results: int,
        geometry_weight: float,
        geocoder: Geocoder,
        query_prefix: str,
        prefetch_limit: int,
    ) -> None:
        self._database = database
        self._storage = storage
        self._local_store = local_store
        self._local_extractor = local_extractor
        self._global_extractor = global_extractor
        self._matcher = matcher
        self._vector_store = vector_store
        self._feature_cache = feature_cache
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
        self._query_prefix = query_prefix.strip("/")
        self._prefetch_limit = max(1, prefetch_limit)
        self._log = get_logger("georag.search")
        self._log.info(
            "event=search_service_init retrieval=%s global_w=%.3f local_w=%.3f geometry_w=%.3f max_results=%s prefetch=%s",
            self._retrieval_candidates,
            self._global_weight,
            self._local_weight,
            self._geometry_weight,
            self._max_results,
            self._prefetch_limit,
        )

    async def search_by_image(self, payload: SearchPayload) -> SearchResult:
        """Выполнить поиск и вернуть лучшие совпадения."""
        self._log.info(
            "event=search_image_started top_k=%s plot_dots=%s size_bytes=%s",
            payload.top_k,
            payload.plot_dots,
            len(payload.data),
        )
        query_image = decode_image(payload.data)
        try:
            query_local = await self._local_extractor.aextract(query_image)
            query_global = await self._global_extractor.aextract(query_image)
        except FeatureExtractionError:
            raise
        except Exception as exc:  # pragma: no cover - изоляция ошибок PyTorch/OpenCV
            raise FeatureExtractionError("Ошибка извлечения признаков запроса") from exc

        self._log.debug(
            "event=query_features_extracted keypoints=%s descriptor_dim=%s",
            query_local.keypoints_count,
            query_local.descriptor_dim,
        )

        image_key = payload.image_key
        image_url = payload.image_url
        if image_key is None or image_url is None:
            extension = detect_extension(payload.data)
            image_key, image_url = await self.save_query_image(
                payload.data, extension=extension
            )

        query_descriptor = query_global.normalized()
        score_limit = max(self._retrieval_candidates, payload.top_k, self._max_results)
        scored_points = await self._vector_store.search(
            query_descriptor, limit=score_limit
        )

        if not scored_points:
            async with self._database.session() as session:
                repo = ImageRepository(session)
                if await repo.count() == 0:
                    raise EmptyDatabaseError("В базе отсутствуют изображения")
            self._log.info("event=no_similar_images_found")
            return SearchResult(
                query_image_key=image_key,
                query_image_url=image_url,
                query_local=query_local,
                query_global=query_global,
                matches=[],
            )

        record_ids: list[int] = []
        for point in scored_points:
            payload_meta = point.payload or {}
            record_id = payload_meta.get("record_id")
            if record_id is None:
                continue
            record_ids.append(int(record_id))

        async with self._database.session() as session:
            repo = ImageRepository(session)
            records = await repo.get_by_ids(record_ids)

        records_by_id = {record.id: record for record in records}
        ranked_candidates: list[tuple[ImageRecord, float]] = []
        for point in scored_points:
            payload_meta = point.payload or {}
            record_id = payload_meta.get("record_id")
            if record_id is None:
                continue
            record = records_by_id.get(int(record_id))
            if record is None:
                continue
            ranked_candidates.append((record, float(point.score)))

        if not ranked_candidates:
            self._log.info("event=no_candidates_after_metadata")
            return SearchResult(
                query_image_key=image_key,
                query_image_url=image_url,
                query_local=query_local,
                query_global=query_global,
                matches=[],
            )

        await self._prefetch_features(
            [record.local_feature_path for record, _ in ranked_candidates[: self._prefetch_limit]]
        )

        best_heap: list[tuple[float, ImageRecord, MatchScore, float, LocalFeatureSet]] = []
        evaluated = 0
        for record, global_score in ranked_candidates:
            if payload.top_k > 0 and len(best_heap) >= payload.top_k:
                min_conf = best_heap[0][0]
                max_possible = (
                    self._global_weight * global_score
                    + self._local_weight
                    + self._geometry_weight
                )
                if max_possible <= min_conf + 1e-6:
                    self._log.debug(
                        "event=search_pruned_by_upper_bound record_id=%s max_possible=%.4f threshold=%.4f",
                        record.id,
                        max_possible,
                        min_conf,
                    )
                    break

            candidate_features = await self._load_local_features(record.local_feature_path)
            match_score = await self._matcher.amatch(query_local, candidate_features)
            confidence = self._aggregate_scores(global_score, match_score)
            self._log.debug(
                "event=match_scores record_id=%s matches=%s mean_score=%.4f confidence=%.4f",
                record.id,
                match_score.matches,
                match_score.mean_score,
                confidence,
            )
            if match_score.matches == 0:
                continue
            evaluated += 1
            if payload.top_k == 0:
                continue
            entry = (confidence, record, match_score, global_score, candidate_features)
            if len(best_heap) < payload.top_k:
                heapq.heappush(best_heap, entry)
            elif confidence > best_heap[0][0] + 1e-9:
                heapq.heapreplace(best_heap, entry)

        if payload.top_k > 0 and not best_heap:
            self._log.info("event=no_matches_after_local_filter")
            return SearchResult(
                query_image_key=image_key,
                query_image_url=image_url,
                query_local=query_local,
                query_global=query_global,
                matches=[],
            )

        selected = (
            sorted(best_heap, key=lambda item: item[0], reverse=True)
            if payload.top_k > 0
            else []
        )

        matches: list[SearchMatchResult] = []
        for confidence, record, match_score, global_score, features in selected:
            matches.append(
                SearchMatchResult(
                    record=record,
                    image_url=self._storage.build_url(record.image_path),
                    local_features=features,
                    match_score=match_score,
                    global_similarity=global_score,
                    confidence=confidence,
                )
            )

        self._log.info(
            "event=search_image_completed matches_evaluated=%s matches_returned=%s",
            evaluated,
            len(matches),
        )

        return SearchResult(
            query_image_key=image_key,
            query_image_url=image_url,
            query_local=query_local,
            query_global=query_global,
            matches=matches,
        )

    async def search_by_coordinates(
        self, payload: CoordinateSearchPayload
    ) -> LocationSearchResult:
        """Найти ближайшие изображения относительно координат."""

        async with self._database.session() as session:
            repo = ImageRepository(session)
            records = await repo.list_all()

        if not records:
            raise EmptyDatabaseError("В базе отсутствуют изображения")

        scored: list[tuple[ImageRecord, float]] = []
        for record in records:
            if record.latitude is None or record.longitude is None:
                continue
            distance = self._haversine(
                payload.latitude,
                payload.longitude,
                record.latitude,
                record.longitude,
            )
            scored.append((record, distance))

        if not scored:
            return LocationSearchResult(matches=[])

        scored.sort(key=lambda item: item[1])
        top_records = scored[: payload.top_k]

        matches: list[LocationMatchResult] = []
        for record, distance in top_records:
            local_features: LocalFeatureSet | None = None
            if payload.plot_dots:
                local_features = await self._load_local_features(
                    record.local_feature_path
                )
            matches.append(
                LocationMatchResult(
                    record=record,
                    image_url=self._storage.build_url(record.image_path),
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

    async def _load_local_features(self, key: str) -> LocalFeatureSet:
        async def loader() -> LocalFeatureSet:
            stored = await self._local_store.load(key)
            return stored.pack

        return await self._feature_cache.get_or_load(key, loader)

    def _aggregate_scores(self, global_score: float, match_score: MatchScore) -> float:
        """Комбинировать все уровни оценки из пайплайна HLOC."""

        return (
            self._global_weight * global_score
            + self._local_weight * match_score.local_score
            + self._geometry_weight * match_score.geometric_strength
        )

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

    async def save_query_image(
        self, data: bytes, *, extension: str | None = None
    ) -> tuple[str, str]:
        ext = extension or detect_extension(data)
        suffix = f".{ext}" if ext else ""
        stem = uuid.uuid4().hex
        prefix = self._query_prefix
        object_key = f"{prefix}/{stem}{suffix}" if prefix else f"{stem}{suffix}"
        content_type = f"image/{ext}" if ext else "application/octet-stream"
        await self._storage.save(
            object_key,
            data,
            content_type=content_type,
            metadata={"query": "true"},
        )
        return object_key, self._storage.build_url(object_key)

    async def _prefetch_features(self, keys: list[str]) -> None:
        if not keys:
            return
        unique = list(dict.fromkeys(keys))
        tasks = [asyncio.create_task(self._load_local_features(key)) for key in unique]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for key, result in zip(unique, results, strict=False):
            if isinstance(result, Exception):
                self._log.warning(
                    "event=feature_prefetch_failed key=%s error=%s", key, result
                )

    async def search(self, payload: SearchPayload) -> SearchResult:
        """Совместимость с предыдущим интерфейсом."""

        return await self.search_by_image(payload)
