"""Точка входа ...."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.gzip import GZipMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .api.routes import router as api_router
from .config import get_settings
from .db import Database
from .logging import configure_logging
from .storage import S3Storage

LOG = logging.getLogger("medical.lifespan")


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


    try:
        await ...
    except Exception as exc:
        LOG.exception("event=warmup_failed error=%s", exc)
        await database.aclose()
        raise

    LOG.info('event=lifespan_init message="Сервис мультимодального поиска готов"')

    try:
        yield
    finally:
        rag = getattr(app.state, "rag_service", None)
        if rag is not None:
            try:
                await rag.aclose()
            except Exception as exc:
                LOG.exception("event=rag_close_failed error=%s", exc)
        try:
            await database.aclose()
        except Exception as exc:
            LOG.exception("event=db_close_failed error=%s", exc)
        LOG.info("event=resource_cleanup status=success")


def create_app() -> FastAPI:
    """Создать и вернуть приложение FastAPI."""
    settings = get_settings()
    configure_logging(service_name="Medical_service", level=settings.log_level)
    docs_url = "/docs" if settings.dev_mode else None
    redoc_url = None
    openapi_url = "/openapi.json" if settings.dev_mode else None

    app = FastAPI(
        title="Medical Service",
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
