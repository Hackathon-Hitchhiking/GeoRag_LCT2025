"""SQLAlchemy модели инфраструктурного слоя."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, Float, Integer, LargeBinary, String, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Базовый класс для всех ORM-моделей."""


class TimestampMixin:
    """Общие временные метки для записей."""

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class ImageRecord(TimestampMixin, Base):
    """Запись о геопривязанном изображении и связанных с ним признаках."""

    __tablename__ = "images"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    image_key: Mapped[str] = mapped_column(String(512), unique=True, nullable=False)
    feature_key: Mapped[str] = mapped_column(String(512), unique=True, nullable=False)
    global_descriptor_key: Mapped[str] = mapped_column(
        String(512), unique=True, nullable=False
    )
    preview_key: Mapped[str | None] = mapped_column(String(512), nullable=True)
    latitude: Mapped[float | None] = mapped_column(Float, nullable=True)
    longitude: Mapped[float | None] = mapped_column(Float, nullable=True)
    address: Mapped[str | None] = mapped_column(String(512), nullable=True)
    metadata: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    image_hash: Mapped[str] = mapped_column(String(128), index=True, nullable=False)
    descriptor_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    descriptor_dim: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    keypoint_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    global_descriptor_dim: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    global_descriptor: Mapped[bytes] = mapped_column(
        LargeBinary, nullable=False, default=b""
    )
    local_feature_type: Mapped[str] = mapped_column(String(64), nullable=False)
    global_descriptor_type: Mapped[str] = mapped_column(String(64), nullable=False)
    matcher_type: Mapped[str] = mapped_column(String(64), nullable=False)

    def __repr__(self) -> str:
        return (
            "ImageRecord(id={id}, image_key={image_key}, lat={lat}, lon={lon})".format(
                id=self.id,
                image_key=self.image_key,
                lat=self.latitude,
                lon=self.longitude,
            )
        )
