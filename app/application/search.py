"""Сервис поиска по базе изображений с двухуровневым сопоставлением."""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

from ..infrastructure.database.models import ImageRecord
from ..infrastructure.storage.s3 import S3Storage
from ..logging import get_logger
from ..utils.image import decode_image, draw_keypoints, encode_image
from .exceptions import EmptyDatabaseError, FeatureExtractionError, GeocodingError
from .features import LocalFeatureSet, SuperPointFeatureExtractor
from .geocoder import Geocoder
from .global_descriptors import GlobalDescriptor, NetVLADGlobalExtractor
from .index import CachedImageEntry, ImageFeatureIndex
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
        storage: S3Storage,
        local_extractor: SuperPointFeatureExtractor,
        global_extractor: NetVLADGlobalExtractor,
        matcher: LightGlueMatcher,
        index: ImageFeatureIndex,
        retrieval_candidates: int,
        global_weight: float,
        local_weight: float,
        max_results: int,
        geometry_weight: float,
        geocoder: Geocoder,
    ) -> None:
        self._storage = storage
        self._local_extractor = local_extractor
        self._global_extractor = global_extractor
        self._matcher = matcher
        self._index = index
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
        self._log = get_logger("georag.search")
        self._log.info(
            "event=search_service_init retrieval=%s global_w=%.3f local_w=%.3f geometry_w=%.3f max_results=%s",
            self._retrieval_candidates,
            self._global_weight,
            self._local_weight,
            self._geometry_weight,
            self._max_results,
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

        query_descriptor = query_global.normalized()

        score_limit = max(
            self._retrieval_candidates, payload.top_k, self._max_results
        )
        scored = await self._index.score_by_global(
            query_descriptor, limit=score_limit
        )
        if not scored:
            entries_snapshot = await self._index.get_entries()
            if not entries_snapshot:
                raise EmptyDatabaseError("В базе отсутствуют изображения")
            self._log.info("event=no_similar_images_found")
            return SearchResult(
                query_bytes=payload.data,
                query_local=query_local,
                query_global=query_global,
                matches=[],
            )

        candidate_limit = min(len(scored), score_limit)
        candidates = scored[:candidate_limit]

        self._log.debug(
            "event=candidates_selected count=%s global_scores_sample=%s",
            len(candidates),
            [round(score, 4) for _, score in candidates[:5]],
        )

        best_heap: list[tuple[float, CachedImageEntry, MatchScore, float]] = []
        evaluated = 0
        for entry, global_score in candidates:
            record = entry.record
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

            self._log.debug(
                "event=match_candidate record_id=%s image_key=%s global_score=%.4f",
                record.id,
                record.image_key,
                global_score,
            )
            candidate_features = entry.local_features
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
                self._log.debug(
                    "event=match_rejected reason=no_local_matches record_id=%s",
                    record.id,
                )
                continue
            evaluated += 1
            if payload.top_k == 0:
                continue
            if len(best_heap) < payload.top_k:
                heapq.heappush(best_heap, (confidence, entry, match_score, global_score))
            elif confidence > best_heap[0][0] + 1e-9:
                heapq.heapreplace(best_heap, (confidence, entry, match_score, global_score))

        if payload.top_k > 0 and not best_heap:
            self._log.info("event=no_matches_after_local_filter")
            return SearchResult(
                query_bytes=payload.data,
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
        for confidence, entry, match_score, global_score in selected:
            record = entry.record
            image_bytes = await self._storage.download(record.image_key)
            matches.append(
                SearchMatchResult(
                    record=record,
                    image_bytes=image_bytes,
                    local_features=entry.local_features,
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
            query_bytes=payload.data,
            query_local=query_local,
            query_global=query_global,
            matches=matches,
        )

    async def search_by_coordinates(
        self, payload: CoordinateSearchPayload
    ) -> LocationSearchResult:
        """Найти ближайшие изображения относительно координат."""

        entries = await self._index.get_entries()
        if not entries:
            raise EmptyDatabaseError("В базе отсутствуют изображения")

        scored: list[tuple[CachedImageEntry, float]] = []
        for entry in entries:
            record = entry.record
            if record.latitude is None or record.longitude is None:
                continue
            distance = self._haversine(
                payload.latitude, payload.longitude, record.latitude, record.longitude
            )
            scored.append((entry, distance))

        if not scored:
            return LocationSearchResult(matches=[])

        scored.sort(key=lambda item: item[1])
        top_records = scored[: payload.top_k]

        matches: list[LocationMatchResult] = []
        for entry, distance in top_records:
            record = entry.record
            image_bytes = await self._storage.download(record.image_key)
            local_features: LocalFeatureSet | None = None
            if payload.plot_dots:
                local_features = entry.local_features
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
