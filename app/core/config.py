"""Конфигурация приложения."""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field, FieldValidationInfo, field_validator
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

    @field_validator("depth_clip_upper_percentile")
    @classmethod
    def _validate_clip_percentiles(
        cls, upper: float, info: FieldValidationInfo
    ) -> float:
        lower = info.data.get("depth_clip_lower_percentile")
        if lower is not None and upper <= lower:
            raise ValueError("DEPTH_CLIP_UPPER должно быть больше DEPTH_CLIP_LOWER")
        return upper

    # PostgreSQL connection
    database_dsn: str = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost:5432/geodb",
        alias="DATABASE_DSN",
    )
    database_echo: bool = Field(default=False, alias="DATABASE_ECHO")

    # Object storage (S3 compatible)
    s3_bucket: str = Field(default="georag", alias="S3_BUCKET")
    s3_region: str | None = Field(default=None, alias="S3_REGION")
    s3_endpoint_url: str | None = Field(default=None, alias="S3_ENDPOINT_URL")
    s3_access_key: str | None = Field(default=None, alias="S3_ACCESS_KEY")
    s3_secret_key: str | None = Field(default=None, alias="S3_SECRET_KEY")
    s3_session_token: str | None = Field(default=None, alias="S3_SESSION_TOKEN")
    s3_public_base_url: str | None = Field(default=None, alias="S3_PUBLIC_BASE_URL")
    s3_presign_ttl: int = Field(default=3600, alias="S3_PRESIGN_TTL", ge=60, le=86400)
    s3_max_parallel: int = Field(default=16, alias="S3_MAX_PARALLEL", ge=1, le=64)
    image_subdir: str = Field(default="images", alias="STORAGE_IMAGE_SUBDIR")
    feature_subdir: str = Field(default="features", alias="STORAGE_FEATURE_SUBDIR")
    preview_subdir: str = Field(default="previews", alias="STORAGE_PREVIEW_SUBDIR")
    query_subdir: str = Field(default="queries", alias="STORAGE_QUERY_SUBDIR")

    # Local feature cache sizing
    feature_cache_size_mb: int = Field(
        default=1536, alias="FEATURE_CACHE_SIZE_MB", ge=128, le=16384
    )
    feature_cache_items: int = Field(
        default=512, alias="FEATURE_CACHE_ITEMS", ge=8, le=16384
    )

    # Qdrant vector database
    qdrant_url: str = Field(default="http://localhost:6333", alias="QDRANT_URL")
    qdrant_api_key: str | None = Field(default=None, alias="QDRANT_API_KEY")
    qdrant_collection: str = Field(
        default="georag_images", alias="QDRANT_COLLECTION"
    )
    qdrant_shard_number: int = Field(
        default=1, alias="QDRANT_SHARDS", ge=1, le=32
    )
    qdrant_replication_factor: int = Field(
        default=1, alias="QDRANT_REPLICATION", ge=1, le=5
    )
    qdrant_on_disk: bool = Field(default=True, alias="QDRANT_ON_DISK")

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
    point_cloud_limit: int = Field(default=2048, alias="POINT_CLOUD_LIMIT", ge=64, le=8192)
    feature_prefetch_limit: int = Field(default=16, alias="FEATURE_PREFETCH_LIMIT", ge=1, le=128)
    depth_model_repo: str = Field(
        default="depth-anything/Depth-Anything-V2-Large", alias="DEPTH_MODEL_REPO"
    )
    depth_model_filename: str = Field(
        default="depth_anything_v2_vitl.pth", alias="DEPTH_MODEL_FILENAME"
    )
    depth_sample_step: int = Field(default=4, alias="DEPTH_SAMPLE_STEP", ge=1, le=64)
    depth_clip_enabled: bool = Field(default=True, alias="DEPTH_CLIP_ENABLED")
    depth_clip_lower_percentile: float = Field(
        default=0.5,
        alias="DEPTH_CLIP_LOWER",
        ge=0.0,
        le=99.9,
    )
    depth_clip_upper_percentile: float = Field(
        default=99.5,
        alias="DEPTH_CLIP_UPPER",
        ge=0.1,
        le=100.0,
    )
    depth_ground_filter: bool = Field(default=True, alias="DEPTH_GROUND_FILTER")
    depth_ground_distance: float = Field(
        default=0.05,
        alias="DEPTH_GROUND_DISTANCE",
        gt=0.0,
    )
    depth_ground_relative_distance: bool = Field(
        default=True,
        alias="DEPTH_GROUND_RELATIVE_DISTANCE",
    )
    depth_ground_min_normal: float = Field(
        default=0.7,
        alias="DEPTH_GROUND_MIN_NORMAL",
        ge=0.0,
        le=1.0,
    )
    depth_ground_min_ratio: float = Field(
        default=0.08,
        alias="DEPTH_GROUND_MIN_RATIO",
        ge=0.0,
        le=1.0,
    )
    depth_ground_iterations: int = Field(
        default=72,
        alias="DEPTH_GROUND_ITERATIONS",
        ge=1,
        le=512,
    )

    # Nominatim integration
    nominatim_user_agent: str | None = Field(
        default=None, alias="NOMINATIM_USER_AGENT"
    )
    nominatim_timeout: int = Field(default=5, alias="NOMINATIM_TIMEOUT")

    # Vision-language model via Ollama
    ollama_base_url: str = Field(
        default="http://localhost:11434", alias="OLLAMA_BASE_URL"
    )
    ollama_model: str = Field(
        default="qwen2.5-vl:7b-instruct", alias="OLLAMA_MODEL"
    )
    ollama_timeout: float = Field(
        default=120.0, alias="OLLAMA_TIMEOUT_SECONDS"
    )

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
