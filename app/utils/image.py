"""Вспомогательные функции для работы с изображениями."""

from __future__ import annotations

import base64
import binascii
import imghdr
from collections.abc import Sequence
from typing import Any

import cv2
import numpy as np

MIN_WIDTH = 640
MIN_HEIGHT = 420
MAX_WIDTH = 5500
MAX_HEIGHT = 3500


def ensure_supported_size(image: np.ndarray) -> np.ndarray:
    """Подогнать размер изображения под допустимые пределы.

    Диапазон размеров продиктован требованиями продукта (от 640×420 до
    5500×3500). Для слишком больших изображений выполняется мягкое
    downscale с сохранением пропорций. Маленькие изображения аккуратно
    масштабируются вверх, чтобы последующие модели (SuperPoint, NetVLAD)
    получали достаточно информации и вели себя предсказуемо.
    """

    height, width = image.shape[:2]
    scale = 1.0

    if width > MAX_WIDTH or height > MAX_HEIGHT:
        scale = min(MAX_WIDTH / width, MAX_HEIGHT / height)
    elif width < MIN_WIDTH or height < MIN_HEIGHT:
        scale = max(MIN_WIDTH / max(1, width), MIN_HEIGHT / max(1, height))

    if np.isclose(scale, 1.0):
        return image

    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_CUBIC)


def decode_image(data: bytes) -> np.ndarray:
    """Преобразовать сырые байты изображения в матрицу BGR."""
    array = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Не удалось декодировать изображение")
    return image


def encode_image(image: np.ndarray, *, ext: str = "png") -> bytes:
    """Преобразовать матрицу BGR в байты указанного формата."""
    success, buffer = cv2.imencode(f".{ext}", image)
    if not success:
        raise ValueError("Не удалось кодировать изображение")
    return bytes(buffer)


def to_base64(data: bytes) -> str:
    """Представить бинарные данные в base64."""
    return base64.b64encode(data).decode("ascii")


def from_base64(payload: str) -> bytes:
    """Преобразовать base64-строку в байты."""
    try:
        return base64.b64decode(payload, validate=True)
    except (ValueError, binascii.Error) as exc:  # type: ignore[name-defined]
        raise ValueError("Невалидная base64-строка") from exc


def detect_extension(data: bytes, *, fallback: str = "jpg") -> str:
    """Определить расширение изображения по содержимому."""
    detected = imghdr.what(None, h=data)
    if not detected:
        return fallback
    return "jpg" if detected == "jpeg" else detected


def draw_keypoints(image: np.ndarray, keypoints: Sequence[Any]) -> np.ndarray:
    """Отрисовать ключевые точки на копии изображения."""
    output = image.copy()
    return cv2.drawKeypoints(
        output,
        keypoints,
        None,
        color=(0, 255, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )
