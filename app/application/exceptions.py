"""Общие исключения уровня сервисов."""

from __future__ import annotations


class ServiceError(RuntimeError):
    """Базовый класс для ошибок сервисного слоя."""


class DuplicateImageError(ServiceError):
    """Изображение уже присутствует в базе."""


class FeatureExtractionError(ServiceError):
    """Не удалось выделить дескрипторы."""


class EmptyDatabaseError(ServiceError):
    """В базе отсутствуют изображения для поиска."""


class StorageError(ServiceError):
    """Ошибка при взаимодействии с объектным хранилищем."""


class GeocodingError(ServiceError):
    """Ошибка при прямом или обратном геокодировании."""
