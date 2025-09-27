"""Маршруты FastAPI для сервиса визуальной геолокации."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, status

from ..application.exceptions import (
    DuplicateImageError,
    EmptyDatabaseError,
    FeatureExtractionError,
    GeocodingError,
    StorageError,
)
from ..application.ingestion import ImageIngestionService, IngestionPayload
from ..application.search import (
    AddressSearchPayload,
    CoordinateSearchPayload,
    ImageSearchService,
    SearchPayload,
)
from ..logging import get_logger
from ..schemas import (
    AddressSearchRequest,
    CoordinatesSearchRequest,
    ImageIngestRequest,
    ImageIngestResponse,
    LocationSearchMatch,
    LocationSearchResponse,
    SearchMatch,
    SearchRequest,
    SearchResponse,
)
from ..utils.image import from_base64, to_base64

LOG = get_logger("georag.api")

router = APIRouter(prefix="/v1", tags=["geolocator"])


def _get_ingestion_service(request: Request) -> ImageIngestionService:
    service = getattr(request.app.state, "ingestion_service", None)
    if service is None:
        raise RuntimeError("Ingestion service не инициализирован")
    return service


def _get_search_service(request: Request) -> ImageSearchService:
    service = getattr(request.app.state, "search_service", None)
    if service is None:
        raise RuntimeError("Search service не инициализирован")
    return service


@router.get("/health", summary="Проверка готовности")
async def healthcheck() -> dict[str, str]:
    """Простейшая проверка статуса сервиса."""
    return {"status": "ok"}


@router.put(
    "/images",
    status_code=status.HTTP_201_CREATED,
    response_model=ImageIngestResponse,
    summary="Добавить изображение в базу",
)
async def add_image(request: Request, payload: ImageIngestRequest) -> ImageIngestResponse:
    """Загрузить новое изображение и сформировать его признаки."""
    try:
        image_bytes = from_base64(payload.image_base64)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    ingestion_service = _get_ingestion_service(request)
    local_store = getattr(request.app.state, "local_store")
    global_store = getattr(request.app.state, "global_store")
    storage = getattr(request.app.state, "storage")
    ingestion_payload = IngestionPayload(
        data=image_bytes,
        latitude=payload.latitude,
        longitude=payload.longitude,
        metadata=payload.metadata,
    )

    try:
        stored = await ingestion_service.ingest(ingestion_payload)
    except DuplicateImageError as exc:
        LOG.info("event=image_duplicate")
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except FeatureExtractionError as exc:
        LOG.warning("event=feature_extraction_failed error=%s", exc)
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except StorageError as exc:
        LOG.error("event=storage_failed error=%s", exc)
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc

    return ImageIngestResponse(
        id=stored.record_id,
        image_uri=storage.build_path(stored.image_key),
        local_feature_uri=local_store.build_uri(stored.feature_key),
        global_descriptor_uri=global_store.build_uri(stored.global_descriptor_key),
        latitude=stored.latitude,
        longitude=stored.longitude,
        address=stored.address,
        metadata=stored.metadata,
        descriptor_count=stored.descriptor_count,
        descriptor_dim=stored.descriptor_dim,
        keypoint_count=stored.keypoint_count,
        global_descriptor_dim=stored.global_descriptor_dim,
        local_feature_type=stored.local_feature_type,
        global_descriptor_type=stored.global_descriptor_type,
        matcher_type=stored.matcher_type,
        created_at=stored.created_at,
        updated_at=stored.updated_at,
    )


@router.post(
    "/search_by_image",
    response_model=SearchResponse,
    summary="Найти схожие изображения",
)
async def search_by_image(request: Request, payload: SearchRequest) -> SearchResponse:
    """Выполнить поиск по изображениям из базы."""
    try:
        query_bytes = from_base64(payload.image_base64)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    search_service = _get_search_service(request)
    search_payload = SearchPayload(
        data=query_bytes,
        plot_dots=payload.plot_dots,
        top_k=payload.top_k,
    )

    try:
        result = await search_service.search_by_image(search_payload)
    except EmptyDatabaseError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except FeatureExtractionError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc

    query_image_bytes = result.query_bytes
    if payload.plot_dots and result.query_local.keypoints_count:
        query_image_bytes = await search_service.prepare_visual(
            result.query_local, query_image_bytes
        )

    storage = getattr(request.app.state, "storage")
    local_store = getattr(request.app.state, "local_store")
    matches: list[SearchMatch] = []
    for match in result.matches:
        image_bytes = match.image_bytes
        if payload.plot_dots:
            image_bytes = await search_service.prepare_visual(match.local_features, image_bytes)
        matches.append(
            SearchMatch(
                image_id=match.record.id,
                confidence=match.confidence,
                global_similarity=match.global_similarity,
                local_matches=match.match_score.matches,
                local_match_ratio=match.match_score.match_ratio,
                local_mean_score=match.match_score.mean_score,
                geometry_inliers=match.match_score.inliers,
                geometry_inlier_ratio=match.match_score.inlier_ratio,
                geometry_score=match.match_score.geometric_strength,
                image_uri=storage.build_path(match.record.image_key),
                feature_uri=local_store.build_uri(match.record.feature_key),
                latitude=match.record.latitude,
                longitude=match.record.longitude,
                address=match.record.address,
                metadata=match.record.metadata,
                image_base64=to_base64(image_bytes),
            )
        )

    return SearchResponse(
        query_image_base64=to_base64(query_image_bytes),
        matches=matches,
    )


@router.post(
    "/search_by_coordinates",
    response_model=LocationSearchResponse,
    summary="Найти ближайшие изображения по координатам",
)
async def search_by_coordinates(
    request: Request, payload: CoordinatesSearchRequest
) -> LocationSearchResponse:
    """Вернуть снимки поблизости от заданных координат."""

    search_service = _get_search_service(request)
    search_payload = CoordinateSearchPayload(
        latitude=payload.latitude,
        longitude=payload.longitude,
        plot_dots=payload.plot_dots,
        top_k=payload.top_k,
    )

    try:
        result = await search_service.search_by_coordinates(search_payload)
    except EmptyDatabaseError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    storage = getattr(request.app.state, "storage")
    local_store = getattr(request.app.state, "local_store")

    matches: list[LocationSearchMatch] = []
    for match in result.matches:
        image_bytes = match.image_bytes
        if payload.plot_dots and match.local_features is not None:
            image_bytes = await search_service.prepare_visual(match.local_features, image_bytes)
        matches.append(
            LocationSearchMatch(
                image_id=match.record.id,
                distance_meters=match.distance_meters,
                image_uri=storage.build_path(match.record.image_key),
                feature_uri=local_store.build_uri(match.record.feature_key),
                latitude=match.record.latitude,
                longitude=match.record.longitude,
                address=match.record.address,
                metadata=match.record.metadata,
                image_base64=to_base64(image_bytes),
            )
        )

    return LocationSearchResponse(matches=matches)


@router.post(
    "/search_by_address",
    response_model=LocationSearchResponse,
    summary="Найти ближайшие изображения по адресу",
)
async def search_by_address(
    request: Request, payload: AddressSearchRequest
) -> LocationSearchResponse:
    """Вернуть снимки поблизости от указанного адреса."""

    search_service = _get_search_service(request)
    search_payload = AddressSearchPayload(
        address=payload.address,
        plot_dots=payload.plot_dots,
        top_k=payload.top_k,
    )

    try:
        result = await search_service.search_by_address(search_payload)
    except GeocodingError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except EmptyDatabaseError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    storage = getattr(request.app.state, "storage")
    local_store = getattr(request.app.state, "local_store")

    matches: list[LocationSearchMatch] = []
    for match in result.matches:
        image_bytes = match.image_bytes
        if payload.plot_dots and match.local_features is not None:
            image_bytes = await search_service.prepare_visual(match.local_features, image_bytes)
        matches.append(
            LocationSearchMatch(
                image_id=match.record.id,
                distance_meters=match.distance_meters,
                image_uri=storage.build_path(match.record.image_key),
                feature_uri=local_store.build_uri(match.record.feature_key),
                latitude=match.record.latitude,
                longitude=match.record.longitude,
                address=match.record.address,
                metadata=match.record.metadata,
                image_base64=to_base64(image_bytes),
            )
        )

    return LocationSearchResponse(matches=matches)
