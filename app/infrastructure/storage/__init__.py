"""Подсистема работы с объектным хранилищем."""

from .filesystem import FileSystemStorage
from .s3 import S3Storage

__all__ = ["FileSystemStorage", "S3Storage"]
