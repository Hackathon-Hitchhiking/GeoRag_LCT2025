"""Точка входа FastAPI приложения."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.gzip import GZipMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .api.routes import router as api_router
from .application.depth_anything import (
    DepthAnythingPointCloudGenerator,
    GroundPlaneFilterConfig,
)
from .application.feature_store import FeatureStore
from .application.features import LocalFeatureSet, SuperPointFeatureExtractor
from .application.geocoder import Geocoder
from .application.global_descriptors import NetVLADGlobalExtractor
from .application.ingestion import ImageIngestionService
from .application.local_cache import LocalFeatureCache
from .application.matcher import LightGlueMatcher
from .application.search import ImageSearchService
from .core.config import get_settings
from .infrastructure.database.session import Database
from .infrastructure.storage import S3Storage
from .infrastructure.vector.qdrant import QdrantVectorStore
from .logging import configure_logging

LOG = logging.getLogger("georag.lifespan")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Инициализировать ресурсы на старте приложения."""
    settings = get_settings()
    database = Database(settings.database_dsn, echo=settings.database_echo)
    await database.init_models()

    storage = S3Storage(
        bucket=settings.s3_bucket,
        region=settings.s3_region,
        endpoint_url=settings.s3_endpoint_url,
        access_key=settings.s3_access_key,
        secret_key=settings.s3_secret_key,
        session_token=settings.s3_session_token,
        public_base_url=settings.s3_public_base_url,
        presign_ttl=settings.s3_presign_ttl,
        max_parallel=settings.s3_max_parallel,
    )

    if settings.compute_device:
        device = torch.device(settings.compute_device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    local_store = FeatureStore[LocalFeatureSet](
        storage,
        prefix=settings.feature_subdir,
        serializer=lambda pack: pack.to_bytes(),
        deserializer=LocalFeatureSet.from_bytes,
    )

    local_extractor = SuperPointFeatureExtractor(
        device=device,
        max_keypoints=settings.feature_max_keypoints,
    )
    global_extractor = NetVLADGlobalExtractor(device=device)
    matcher = LightGlueMatcher(device=device)
    geocoder = Geocoder(user_agent=settings.nominatim_user_agent, timeout=settings.nominatim_timeout)
    feature_cache = LocalFeatureCache(
        max_bytes=settings.feature_cache_size_mb * 1024 * 1024,
        max_items=settings.feature_cache_items,
    )
    vector_store = QdrantVectorStore(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        collection=settings.qdrant_collection,
        vector_dim=4096,
        on_disk=settings.qdrant_on_disk,
        shard_number=settings.qdrant_shard_number,
        replication_factor=settings.qdrant_replication_factor,
    )
    await vector_store.ensure_collection()

    ingestion_service = ImageIngestionService(
        database=database,
        storage=storage,
        local_store=local_store,
        local_extractor=local_extractor,
        global_extractor=global_extractor,
        geocoder=geocoder,
        vector_store=vector_store,
        image_prefix=settings.image_subdir,
        local_feature_type=settings.local_feature_type,
        global_descriptor_type=settings.global_descriptor_type,
        matcher_type=settings.matcher_type,
    )
    search_service = ImageSearchService(
        database=database,
        storage=storage,
        local_store=local_store,
        local_extractor=local_extractor,
        global_extractor=global_extractor,
        matcher=matcher,
        vector_store=vector_store,
        feature_cache=feature_cache,
        retrieval_candidates=settings.retrieval_candidates,
        global_weight=settings.global_score_weight,
        local_weight=settings.local_score_weight,
        max_results=settings.max_search_results,
        geometry_weight=settings.geometry_score_weight,
        geocoder=geocoder,
        query_prefix=settings.query_subdir,
        prefetch_limit=settings.feature_prefetch_limit,
    )

    depth_generator: DepthAnythingPointCloudGenerator | None = None
    try:
        clip_percentiles = (
            settings.depth_clip_lower_percentile,
            settings.depth_clip_upper_percentile,
        ) if settings.depth_clip_enabled else None
        ground_config = GroundPlaneFilterConfig(
            enabled=settings.depth_ground_filter,
            min_normal_y=settings.depth_ground_min_normal,
            distance_threshold=settings.depth_ground_distance,
            min_inlier_ratio=settings.depth_ground_min_ratio,
            max_iterations=settings.depth_ground_iterations,
            relative_distance=settings.depth_ground_relative_distance,
        )
        depth_generator = DepthAnythingPointCloudGenerator(
            repo_id=settings.depth_model_repo,
            filename=settings.depth_model_filename,
            device=device,
            default_sample_step=settings.depth_sample_step,
            clip_percentiles=clip_percentiles,
            ground_plane_filter=ground_config,
        )
    except ImportError as exc:
        LOG.warning("event=depth_generator_disabled reason=%s", exc)
    except Exception as exc:  # pragma: no cover - инициализация может падать на GPU
        LOG.exception("event=depth_generator_failed error=%s", exc)

    app.state.database = database
    app.state.storage = storage
    app.state.local_store = local_store
    app.state.feature_cache = feature_cache
    app.state.vector_store = vector_store
    app.state.ingestion_service = ingestion_service
    app.state.search_service = search_service
    app.state.geocoder = geocoder
    app.state.point_cloud_limit = settings.point_cloud_limit
    app.state.depth_generator = depth_generator
    app.state.depth_generator = depth_generator

    LOG.info('event=lifespan_init message="Сервис визуальной геолокации готов"')

    try:
        yield
    finally:
        try:
            await database.aclose()
        except Exception as exc:  # pragma: no cover - вывод логов
            LOG.exception("event=db_close_failed error=%s", exc)
        LOG.info("event=resource_cleanup status=success")


def create_app() -> FastAPI:
    """Создать и вернуть приложение FastAPI."""
    settings = get_settings()
    configure_logging(service_name="geo_service", level=settings.log_level)
    docs_url = "/docs" if settings.dev_mode else None
    redoc_url = None
    openapi_url = "/openapi.json" if settings.dev_mode else None

    app = FastAPI(
        title="Geo Localization Service",
        version="0.1.0",
        lifespan=lifespan,
        docs_url=docs_url,
        redoc_url=redoc_url,
        openapi_url=openapi_url,
    )
    app.add_middleware(GZipMiddleware, minimum_size=1024)
    if settings.dev_mode:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )
    elif settings.allowed_hosts:
        app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.allowed_hosts)

    app.include_router(router=api_router)
    return app


app = create_app()
