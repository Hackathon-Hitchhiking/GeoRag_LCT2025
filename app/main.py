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
from .application.feature_store import FeatureStore
from .application.features import LocalFeatureSet, SuperPointFeatureExtractor
from .application.geocoder import Geocoder
from .application.global_descriptors import GlobalDescriptor, NetVLADGlobalExtractor
from .application.ingestion import ImageIngestionService
from .application.index import ImageFeatureIndex
from .application.matcher import LightGlueMatcher
from .application.search import ImageSearchService
from .core.config import get_settings
from .infrastructure.database.session import Database
from .infrastructure.storage.s3 import S3Storage
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
        endpoint_url=settings.s3_endpoint_url,
        region_name=settings.s3_region,
        access_key=settings.s3_access_key,
        secret_key=settings.s3_secret_key,
        use_ssl=settings.s3_use_ssl,
    )

    if settings.compute_device:
        device = torch.device(settings.compute_device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    local_store = FeatureStore[LocalFeatureSet](
        storage,
        prefix=settings.s3_features_prefix,
        serializer=lambda pack: pack.to_bytes(),
        deserializer=LocalFeatureSet.from_bytes,
    )
    global_store = FeatureStore[GlobalDescriptor](
        storage,
        prefix=settings.s3_global_prefix,
        serializer=lambda desc: desc.to_bytes(),
        deserializer=GlobalDescriptor.from_bytes,
    )

    index = ImageFeatureIndex(
        database=database,
        local_store=local_store,
        refresh_interval=float(getattr(settings, "index_refresh_interval", 7200)),
    )

    local_extractor = SuperPointFeatureExtractor(
        device=device,
        max_keypoints=settings.feature_max_keypoints,
    )
    global_extractor = NetVLADGlobalExtractor(device=device)
    matcher = LightGlueMatcher(device=device)
    geocoder = Geocoder(user_agent=settings.nominatim_user_agent, timeout=settings.nominatim_timeout)

    await index.start()

    ingestion_service = ImageIngestionService(
        database=database,
        storage=storage,
        local_store=local_store,
        global_store=global_store,
        local_extractor=local_extractor,
        global_extractor=global_extractor,
        geocoder=geocoder,
        image_prefix=settings.s3_images_prefix,
        local_feature_type=settings.local_feature_type,
        global_descriptor_type=settings.global_descriptor_type,
        matcher_type=settings.matcher_type,
        index=index,
    )
    search_service = ImageSearchService(
        storage=storage,
        local_extractor=local_extractor,
        global_extractor=global_extractor,
        matcher=matcher,
        index=index,
        retrieval_candidates=settings.retrieval_candidates,
        global_weight=settings.global_score_weight,
        local_weight=settings.local_score_weight,
        max_results=settings.max_search_results,
        geometry_weight=settings.geometry_score_weight,
        geocoder=geocoder,
    )

    app.state.database = database
    app.state.storage = storage
    app.state.local_store = local_store
    app.state.global_store = global_store
    app.state.index = index
    app.state.ingestion_service = ingestion_service
    app.state.search_service = search_service
    app.state.geocoder = geocoder

    LOG.info('event=lifespan_init message="Сервис визуальной геолокации готов"')

    try:
        yield
    finally:
        try:
            await index.stop()
        except Exception as exc:  # pragma: no cover - вывод логов
            LOG.exception("event=index_stop_failed error=%s", exc)
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
