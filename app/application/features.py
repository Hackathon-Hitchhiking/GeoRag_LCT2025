"""Работа с локальными признаками SuperPoint."""
from __future__ import annotations

import asyncio
import io
from dataclasses import dataclass

import cv2
import numpy as np
import torch
from lightglue import SuperPoint
from lightglue.utils import batch_to_device, numpy_image_to_torch

from ..utils.image import ensure_supported_size
from .exceptions import FeatureExtractionError


@dataclass(slots=True)
class LocalFeatureSet:
    """Набор локальных признаков, совместимый с LightGlue."""

    keypoints: np.ndarray
    descriptors: np.ndarray
    scores: np.ndarray
    image_size: tuple[int, int]

    def __post_init__(self) -> None:
        if self.keypoints.ndim != 2 or self.keypoints.shape[1] != 2:
            raise ValueError("Ключевые точки должны иметь форму [N, 2]")
        if self.descriptors.ndim != 2:
            raise ValueError("Дескрипторы должны иметь форму [N, D]")
        if self.scores.ndim != 1:
            raise ValueError("Оценки соответствий должны иметь форму [N]")
        if self.keypoints.shape[0] != self.descriptors.shape[0]:
            raise ValueError("Количество дескрипторов и ключевых точек должно совпадать")
        if self.keypoints.shape[0] != self.scores.shape[0]:
            raise ValueError("Количество оценок и ключевых точек должно совпадать")

    @property
    def descriptor_dim(self) -> int:
        """Размерность дескриптора."""
        return int(self.descriptors.shape[1])

    @property
    def keypoints_count(self) -> int:
        """Количество найденных ключевых точек."""
        return int(self.keypoints.shape[0])

    def to_bytes(self) -> bytes:
        """Сериализовать признаки в компактный бинарный формат."""
        buffer = io.BytesIO()
        np.savez_compressed(
            buffer,
            keypoints=self.keypoints.astype(np.float32),
            descriptors=self.descriptors.astype(np.float32),
            scores=self.scores.astype(np.float32),
            height=np.int32(self.image_size[0]),
            width=np.int32(self.image_size[1]),
            version=np.int32(1),
        )
        return buffer.getvalue()

    @classmethod
    def from_bytes(cls, payload: bytes) -> "LocalFeatureSet":
        """Восстановить признаки из бинарного представления."""
        with io.BytesIO(payload) as buffer:
            with np.load(buffer) as data:
                keypoints = data["keypoints"].astype(np.float32)
                descriptors = data["descriptors"].astype(np.float32)
                scores = data["scores"].astype(np.float32)
                height = int(data["height"])  # type: ignore[arg-type]
                width = int(data["width"])  # type: ignore[arg-type]
        return cls(
            keypoints=keypoints,
            descriptors=descriptors,
            scores=scores,
            image_size=(height, width),
        )

    def to_lightglue_inputs(self, device: torch.device) -> dict[str, torch.Tensor]:
        """Подготовить словарь признаков для передачи в LightGlue."""
        keypoints = torch.from_numpy(self.keypoints).to(device=device, dtype=torch.float32)
        descriptors = torch.from_numpy(self.descriptors).to(
            device=device, dtype=torch.float32
        )
        scores = torch.from_numpy(self.scores).to(device=device, dtype=torch.float32)
        image_size = torch.tensor(
            [[float(self.image_size[1]), float(self.image_size[0])]],
            dtype=torch.float32,
            device=device,
        )
        return batch_to_device(
            {
                "keypoints": keypoints.unsqueeze(0),
                "descriptors": descriptors.unsqueeze(0),
                "scores": scores.unsqueeze(0),
                "image_size": image_size,
            },
            device=device,
        )

    def to_cv_keypoints(self) -> list[cv2.KeyPoint]:
        """Преобразовать признаки в формат OpenCV для визуализации."""
        keypoints_cv: list[cv2.KeyPoint] = []
        for (x, y), score in zip(self.keypoints, self.scores):
            size = max(float(score) * 12.0, 1.0)
            keypoints_cv.append(
                cv2.KeyPoint(x=float(x), y=float(y), size=size, response=float(score))
            )
        return keypoints_cv


class SuperPointFeatureExtractor:
    """Извлечение локальных признаков посредством SuperPoint."""

    def __init__(
        self,
        *,
        device: torch.device,
        max_keypoints: int,
        resize: int = 1600,
    ) -> None:
        self._device = device
        conf: dict[str, float | int | None] = {
            "max_num_keypoints": max_keypoints if max_keypoints > 0 else None,
            "detection_threshold": 5e-4,
        }
        self._resize = resize
        self._extractor = SuperPoint(**conf).to(device).eval()

    def extract(self, image_bgr: np.ndarray) -> LocalFeatureSet:
        """Извлечь ключевые точки и дескрипторы из изображения BGR."""
        if image_bgr.ndim != 3:
            raise FeatureExtractionError("Ожидалось цветное изображение BGR")
        image_bgr = ensure_supported_size(image_bgr)
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        tensor = numpy_image_to_torch(gray)
        tensor = tensor.unsqueeze(0).to(self._device)
        try:
            with torch.inference_mode():
                features = self._extractor.extract(
                    tensor, resize=self._resize, side="long", antialias=True
                )
        except Exception as exc:  # pragma: no cover - исключения PyTorch
            raise FeatureExtractionError("Сбой при извлечении SuperPoint признаков") from exc
        keypoints = features["keypoints"][0].detach().cpu().numpy().astype(np.float32)
        descriptors = features["descriptors"][0].detach().cpu().numpy().astype(np.float32)
        scores = features["scores"][0].detach().cpu().numpy().astype(np.float32)
        image_size_tensor = features["image_size"][0].detach().cpu().numpy()
        image_size = (int(image_size_tensor[1]), int(image_size_tensor[0]))
        if keypoints.size == 0:
            raise FeatureExtractionError("Не удалось выделить ключевые точки")
        return LocalFeatureSet(
            keypoints=keypoints,
            descriptors=descriptors,
            scores=scores,
            image_size=image_size,
        )

    async def aextract(self, image_bgr: np.ndarray) -> LocalFeatureSet:
        """Асинхронная обёртка для вычисления локальных признаков."""
        return await asyncio.to_thread(self.extract, image_bgr)
