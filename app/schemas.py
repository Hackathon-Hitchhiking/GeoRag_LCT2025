"""Pydantic-схемы запросов и ответов HTTP API."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, PositiveInt, field_validator


class ImageIngestRequest(BaseModel):
    """Запрос на добавление изображения в базу."""

    image_base64: str = Field(..., description="Бинарные данные изображения в base64")
    latitude: float | None = Field(default=None, description="Широта точки съёмки")
    longitude: float | None = Field(default=None, description="Долгота точки съёмки")
    metadata: dict[str, Any] | None = Field(
        default=None, description="Дополнительные пользовательские метаданные"
    )

    @field_validator("image_base64")
    @classmethod
    def validate_non_empty(cls, value: str) -> str:
        if not value:
            raise ValueError("Изображение должно быть передано в base64")
        return value


class ImageIngestResponse(BaseModel):
    """Ответ после успешной загрузки изображения."""

    id: int
    image_url: str
    vector_id: str
    latitude: float | None
    longitude: float | None
    address: str | None
    metadata: dict[str, Any] | None
    descriptor_count: int
    descriptor_dim: int
    keypoint_count: int
    global_descriptor_dim: int
    local_feature_type: str
    global_descriptor_type: str
    matcher_type: str
    created_at: datetime
    updated_at: datetime


class SearchRequest(BaseModel):
    """Запрос на поиск схожих изображений."""

    image_base64: str = Field(..., description="Искомое изображение в base64")
    plot_dots: bool = Field(
        default=False, description="Рисовать ли ключевые точки на результирующих кадрах"
    )
    top_k: PositiveInt = Field(default=5, description="Количество лучших совпадений")

    @field_validator("image_base64")
    @classmethod
    def validate_search_image(cls, value: str) -> str:
        if not value:
            raise ValueError("Для поиска необходимо передать изображение")
        return value

    @field_validator("top_k")
    @classmethod
    def validate_top_k(cls, value: int) -> int:
        if value > 50:
            raise ValueError("top_k не должен превышать 50")
        return value


class Point3D(BaseModel):
    x: float
    y: float
    z: float
    score: float


class MatchCorrespondence(BaseModel):
    query: Point3D
    candidate: Point3D
    score: float


class SearchMatch(BaseModel):
    """Описание одного совпадения из базы."""

    image_id: int
    confidence: float
    global_similarity: float
    local_matches: int
    local_match_ratio: float
    local_mean_score: float
    geometry_inliers: int
    geometry_inlier_ratio: float
    geometry_score: float
    image_url: str
    latitude: float | None
    longitude: float | None
    address: str | None
    metadata: dict[str, Any] | None
    point_cloud: list[Point3D]
    correspondences: list[MatchCorrespondence]
    relative_rotation: list[float] | None = None
    relative_translation: list[float] | None = None


class SearchResponse(BaseModel):
    """Ответ поиска с подготовленными изображениями."""

    query_image_url: str
    query_point_cloud: list[Point3D]
    matches: list[SearchMatch]


class CoordinatesSearchRequest(BaseModel):
    """Запрос на поиск ближайших изображений по координатам."""

    latitude: float = Field(..., ge=-90, le=90, description="Широта точки запроса")
    longitude: float = Field(..., ge=-180, le=180, description="Долгота точки запроса")
    plot_dots: bool = Field(
        default=False, description="Наносить ли ключевые точки на исходные изображения"
    )
    top_k: PositiveInt = Field(default=5, description="Количество ближайших снимков")

    @field_validator("top_k")
    @classmethod
    def validate_top_k(cls, value: int) -> int:
        if value > 50:
            raise ValueError("top_k не должен превышать 50")
        return value


class AddressSearchRequest(BaseModel):
    """Запрос на поиск по адресу."""

    address: str = Field(..., min_length=3, description="Текстовый адрес или объект")
    plot_dots: bool = Field(
        default=False, description="Наносить ли ключевые точки на исходные изображения"
    )
    top_k: PositiveInt = Field(default=5, description="Количество ближайших снимков")

    @field_validator("address")
    @classmethod
    def validate_address(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Адрес не должен быть пустым")
        return value

    @field_validator("top_k")
    @classmethod
    def validate_top_k(cls, value: int) -> int:
        if value > 50:
            raise ValueError("top_k не должен превышать 50")
        return value


class LocationSearchMatch(BaseModel):
    """Результат поиска ближайших изображений по координатам или адресу."""

    image_id: int
    distance_meters: float = Field(..., ge=0)
    image_url: str
    latitude: float | None
    longitude: float | None
    address: str | None
    metadata: dict[str, Any] | None
    point_cloud: list[Point3D] | None = None


class LocationSearchResponse(BaseModel):
    """Ответ на запрос поиска по координатам или адресу."""

    matches: list[LocationSearchMatch]
