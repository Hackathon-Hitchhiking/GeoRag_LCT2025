"""Конфигурация приложения."""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Настройки сервиса визуальной геолокации."""

    dev_mode: bool = Field(default=True, alias="DEV_MODE")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    allowed_hosts: list[str] | None = Field(default=None, alias="ALLOWED_HOSTS")

    @field_validator("allowed_hosts", mode="before")
    @classmethod
    def _parse_allowed_hosts(
        cls, value: list[str] | str | None
    ) -> list[str] | None:
        """Разрешить передавать список хостов как строку."""
        if value is None or value == "":
            return None
        if isinstance(value, list):
            # Очищаем элементы, чтобы исключить пустые строки
            hosts = [item.strip() for item in value if item and item.strip()]
            return hosts or None
        if isinstance(value, str):
            hosts = [item.strip() for item in value.split(",") if item.strip()]
            return hosts or None
        return value

    # PostgreSQL connection
    database_dsn: str = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost:5432/geodb",
        alias="DATABASE_DSN",
    )
    database_echo: bool = Field(default=False, alias="DATABASE_ECHO")

    # Object storage (S3 compatible)
    s3_bucket: str = Field(default="geo-images", alias="S3_BUCKET")
    s3_endpoint_url: str | None = Field(default=None, alias="S3_ENDPOINT_URL")
    s3_region: str | None = Field(default=None, alias="S3_REGION")
    s3_access_key: str | None = Field(default=None, alias="S3_ACCESS_KEY")
    s3_secret_key: str | None = Field(default=None, alias="S3_SECRET_KEY")
    s3_use_ssl: bool = Field(default=True, alias="S3_USE_SSL")
    s3_images_prefix: str = Field(default="images", alias="S3_IMAGES_PREFIX")
    s3_features_prefix: str = Field(default="features", alias="S3_FEATURES_PREFIX")
    s3_global_prefix: str = Field(default="global_descriptors", alias="S3_GLOBAL_PREFIX")

    # Feature extraction parameters
    feature_max_keypoints: int = Field(default=4096, alias="FEATURE_MAX_KEYPOINTS")
    local_feature_type: str = Field(
        default="superpoint_max", alias="LOCAL_FEATURE_TYPE"
    )
    global_descriptor_type: str = Field(
        default="netvlad", alias="GLOBAL_DESCRIPTOR_TYPE"
    )
    matcher_type: str = Field(default="lightglue", alias="MATCHER_TYPE")
    retrieval_candidates: int = Field(default=32, alias="RETRIEVAL_CANDIDATES")
    global_score_weight: float = Field(default=0.35, alias="GLOBAL_SCORE_WEIGHT")
    local_score_weight: float = Field(default=0.65, alias="LOCAL_SCORE_WEIGHT")
    geometry_score_weight: float = Field(default=0.25, alias="GEOMETRY_SCORE_WEIGHT")
    compute_device: str | None = Field(default=None, alias="COMPUTE_DEVICE")
    max_search_results: int = Field(default=50, alias="MAX_SEARCH_RESULTS")
    index_refresh_interval: int = Field(
        default=7200, alias="INDEX_REFRESH_INTERVAL"
    )

    # Nominatim integration
    nominatim_user_agent: str | None = Field(
        default=None, alias="NOMINATIM_USER_AGENT"
    )
    nominatim_timeout: int = Field(default=5, alias="NOMINATIM_TIMEOUT")

    model_config = SettingsConfigDict(
        env_file=(".env", "../.env"),
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )


@lru_cache
def get_settings() -> Settings:
    """Загрузить настройки из окружения один раз."""
    return Settings()
