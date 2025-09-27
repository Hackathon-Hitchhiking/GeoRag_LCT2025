"""Application configuration primitives."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Settings for the multimodal medical RAG service."""

    dev_mode: bool = Field(default=True, alias="DEV_MODE")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    allowed_hosts: list[str] | None = Field(default=None, alias="ALLOWED_HOSTS")


    # Qdrant configuration
    qdrant_host: str = Field(default="localhost", alias="QDRANT_HOST")
    qdrant_port: int = Field(default=6333, alias="QDRANT_PORT")
    qdrant_api_key: str | None = Field(default=None, alias="QDRANT_API_KEY")
    qdrant_collection: str = Field(default="medical_cases", alias="QDRANT_COLLECTION")
    qdrant_distance: Literal["cosine", "dot", "euclid"] = Field(
        default="cosine", alias="QDRANT_DISTANCE"
    )
    qdrant_image_weight: float = Field(default=0.4, alias="QDRANT_IMAGE_WEIGHT")


    # PostgreSQL connection
    database_dsn: str = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost:5432/medical",
        alias="DATABASE_DSN",
    )
    database_echo: bool = Field(default=False, alias="DATABASE_ECHO")

    # Object storage (S3 compatible)
    s3_bucket: str = Field(default="medical-cases", alias="S3_BUCKET")
    s3_endpoint_url: str | None = Field(default=None, alias="S3_ENDPOINT_URL")
    s3_region: str | None = Field(default=None, alias="S3_REGION")
    s3_access_key: str | None = Field(default=None, alias="S3_ACCESS_KEY")
    s3_secret_key: str | None = Field(default=None, alias="S3_SECRET_KEY")
    s3_use_ssl: bool = Field(default=True, alias="S3_USE_SSL")

    model_config = SettingsConfigDict(
        env_file=(".env", "../.env"),
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )


@lru_cache
def get_settings() -> Settings:
    """Load settings from the environment only once."""
    return Settings()
