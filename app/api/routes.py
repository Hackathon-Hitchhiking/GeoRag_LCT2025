"""Маршруты FastAPI для сервиса визуальной геолокации."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException, Request, status

from ..application.depth_anything import DepthAnythingPointCloudGenerator

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
    Point3D,
    SearchMatch,
    SearchRequest,
    SearchResponse,
)
from ..utils.image import decode_image, from_base64

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


def _get_depth_generator(
    request: Request,
) -> DepthAnythingPointCloudGenerator | None:
    return getattr(request.app.state, "depth_generator", None)


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
    request_id = uuid.uuid4().hex
    LOG.info(
        "event=add_image_received request_id=%s has_metadata=%s",
        request_id,
        bool(payload.metadata),
    )
    try:
        image_bytes = from_base64(payload.image_base64)
    except ValueError as exc:
        LOG.warning("event=add_image_bad_input request_id=%s error=%s", request_id, exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    ingestion_service = _get_ingestion_service(request)
    ingestion_payload = IngestionPayload(
        data=image_bytes,
        latitude=payload.latitude,
        longitude=payload.longitude,
        metadata=payload.metadata,
    )

    try:
        stored = await ingestion_service.ingest(ingestion_payload)
    except DuplicateImageError as exc:
        LOG.info("event=image_duplicate request_id=%s", request_id)
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except FeatureExtractionError as exc:
        LOG.warning(
            "event=add_image_feature_failed request_id=%s error=%s",
            request_id,
            exc,
        )
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except StorageError as exc:
        LOG.error(
            "event=add_image_storage_failed request_id=%s error=%s",
            request_id,
            exc,
        )
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc

    LOG.info(
        "event=add_image_completed request_id=%s record_id=%s",
        request_id,
        stored.record_id,
    )
    return ImageIngestResponse(
        id=stored.record_id,
        image_url=stored.image_url,
        vector_id=stored.vector_id,
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
    request_id = uuid.uuid4().hex
    LOG.info(
        "event=search_by_image_received request_id=%s top_k=%s plot_dots=%s",
        request_id,
        payload.top_k,
        payload.plot_dots,
    )
    try:
        query_bytes = from_base64(payload.image_base64)
    except ValueError as exc:
        LOG.warning(
            "event=search_by_image_bad_input request_id=%s error=%s",
            request_id,
            exc,
        )
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
        LOG.info("event=search_by_image_empty_db request_id=%s", request_id)
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except FeatureExtractionError as exc:
        LOG.warning(
            "event=search_by_image_feature_failed request_id=%s error=%s",
            request_id,
            exc,
        )
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc

    point_limit = getattr(request.app.state, "point_cloud_limit", 2048)
    include_points = payload.plot_dots
    query_point_cloud: list[Point3D] = []
    depth_generator = _get_depth_generator(request) if include_points else None
    if include_points:
        if depth_generator is None:
            LOG.warning("event=depth_generator_unavailable request_id=%s", request_id)
        else:
            try:
                query_image = decode_image(query_bytes)
            except ValueError as exc:
                LOG.warning(
                    "event=query_decode_failed request_id=%s error=%s",
                    request_id,
                    exc,
                )
            else:
                try:
                    depth_points = depth_generator.generate_point_cloud(
                        query_image,
                        max_points=point_limit,
                    )
                except Exception as exc:  # pragma: no cover - обработка ошибок модели
                    LOG.exception(
                        "event=depth_generation_failed request_id=%s error=%s",
                        request_id,
                        exc,
                    )
                else:
                    if point_limit > 0:
                        depth_points = depth_points[:point_limit]
                    query_point_cloud = [
                        Point3D(
                            x=pt.x,
                            y=pt.y,
                            z=pt.z,
                        )
                        for pt in depth_points
                    ]

    matches: list[SearchMatch] = []
    for match in result.matches:
        rotation = match.match_score.relative_rotation
        translation = match.match_score.relative_translation
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
                image_url=match.image_url,
                latitude=match.record.latitude,
                longitude=match.record.longitude,
                address=match.record.address,
                metadata=match.record.metadata_json,
                point_cloud=[],
                correspondences=[],
                relative_rotation=(
                    rotation.reshape(-1).astype(float).tolist()
                    if rotation is not None
                    else None
                ),
                relative_translation=(
                    translation.astype(float).tolist() if translation is not None else None
                ),
            )
        )

    response = SearchResponse(
        query_image_url=result.query_image_url,
        query_point_cloud=query_point_cloud,
        matches=matches,
    )
    LOG.info(
        "event=search_by_image_completed request_id=%s matches=%s",
        request_id,
        len(matches),
    )
    return response


@router.post(
    "/search_by_coordinates",
    response_model=LocationSearchResponse,
    summary="Найти ближайшие изображения по координатам",
)
async def search_by_coordinates(
    request: Request, payload: CoordinatesSearchRequest
) -> LocationSearchResponse:
    """Вернуть снимки поблизости от заданных координат."""

    request_id = uuid.uuid4().hex
    LOG.info(
        "event=search_by_coordinates_received request_id=%s lat=%.6f lon=%.6f top_k=%s",
        request_id,
        payload.latitude,
        payload.longitude,
        payload.top_k,
    )
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
        LOG.info("event=search_by_coordinates_empty_db request_id=%s", request_id)
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    matches: list[LocationSearchMatch] = []
    for match in result.matches:
        matches.append(
            LocationSearchMatch(
                image_id=match.record.id,
                distance_meters=match.distance_meters,
                image_url=match.image_url,
                latitude=match.record.latitude,
                longitude=match.record.longitude,
                address=match.record.address,
                metadata=match.record.metadata_json,
                point_cloud=None,
            )
        )

    response = LocationSearchResponse(matches=matches)
    LOG.info(
        "event=search_by_coordinates_completed request_id=%s matches=%s",
        request_id,
        len(matches),
    )
    return response


@router.post(
    "/search_by_address",
    response_model=LocationSearchResponse,
    summary="Найти ближайшие изображения по адресу",
)
async def search_by_address(
    request: Request, payload: AddressSearchRequest
) -> LocationSearchResponse:
    """Вернуть снимки поблизости от указанного адреса."""

    request_id = uuid.uuid4().hex
    LOG.info(
        "event=search_by_address_received request_id=%s address=%s top_k=%s",
        request_id,
        payload.address,
        payload.top_k,
    )
    search_service = _get_search_service(request)
    search_payload = AddressSearchPayload(
        address=payload.address,
        plot_dots=payload.plot_dots,
        top_k=payload.top_k,
    )

    try:
        result = await search_service.search_by_address(search_payload)
    except GeocodingError as exc:
        LOG.warning(
            "event=search_by_address_geocoding_failed request_id=%s error=%s",
            request_id,
            exc,
        )
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except EmptyDatabaseError as exc:
        LOG.info("event=search_by_address_empty_db request_id=%s", request_id)
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    matches: list[LocationSearchMatch] = []
    for match in result.matches:
        matches.append(
            LocationSearchMatch(
                image_id=match.record.id,
                distance_meters=match.distance_meters,
                image_url=match.image_url,
                latitude=match.record.latitude,
                longitude=match.record.longitude,
                address=match.record.address,
                metadata=match.record.metadata_json,
                point_cloud=None,
            )
        )

    response = LocationSearchResponse(matches=matches)
    LOG.info(
        "event=search_by_address_completed request_id=%s matches=%s",
        request_id,
        len(matches),
    )
    return response
