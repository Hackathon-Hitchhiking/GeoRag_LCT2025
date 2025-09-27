"""Извлечение глобальных дескрипторов на базе NetVLAD."""

from __future__ import annotations

import asyncio
import io
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from scipy.io import loadmat

from ..logging import get_logger
from ..utils.image import ensure_supported_size
from .exceptions import FeatureExtractionError

EPS = 1e-6


@dataclass(slots=True)
class GlobalDescriptor:
    """Обёртка над векторным представлением изображения."""

    vector: np.ndarray

    def __post_init__(self) -> None:
        if self.vector.ndim != 1:
            raise ValueError("Дескриптор должен быть одномерным")

    @property
    def dimension(self) -> int:
        """Размерность вектора дескриптора."""
        return int(self.vector.shape[0])

    def normalized(self) -> np.ndarray:
        """Вернуть L2-нормализованный вектор."""
        norm = float(np.linalg.norm(self.vector))
        if norm <= 0:
            return self.vector.astype(np.float32)
        return (self.vector / norm).astype(np.float32)

    def to_bytes(self) -> bytes:
        """Сериализовать дескриптор для хранения в S3."""
        buffer = io.BytesIO()
        np.savez_compressed(buffer, descriptor=self.vector.astype(np.float32), version=np.int32(1))
        return buffer.getvalue()

    def to_database_blob(self) -> bytes:
        """Получить бинарное представление для сохранения в БД."""
        return self.normalized().astype(np.float32).tobytes()

    @classmethod
    def from_bytes(cls, payload: bytes) -> "GlobalDescriptor":
        """Восстановить дескриптор из сериализованного вида."""
        with io.BytesIO(payload) as buffer:
            with np.load(buffer) as data:
                vector = data["descriptor"].astype(np.float32)
        return cls(vector=vector)


class NetVLADLayer(nn.Module):
    """Обёртка над сверточной свёрткой NetVLAD."""

    def __init__(self, input_dim: int = 512, clusters: int = 64, intranorm: bool = True) -> None:
        super().__init__()
        self.score_proj = nn.Conv1d(input_dim, clusters, kernel_size=1, bias=True)
        centers = nn.parameter.Parameter(torch.empty([input_dim, clusters]))
        nn.init.xavier_uniform_(centers)
        self.register_parameter("centers", centers)
        self.intranorm = intranorm
        self.output_dim = input_dim * clusters

    def forward(self, descriptors: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        batch_size = descriptors.size(0)
        scores = self.score_proj(descriptors)
        scores = F.softmax(scores, dim=1)
        diff = descriptors.unsqueeze(2) - self.centers.unsqueeze(0).unsqueeze(-1)
        desc = (scores.unsqueeze(1) * diff).sum(dim=-1)
        if self.intranorm:
            desc = F.normalize(desc, dim=1)
        desc = desc.view(batch_size, -1)
        desc = F.normalize(desc, dim=1)
        return desc


class NetVLADBackbone(nn.Module):
    """VGG16 + NetVLAD с предобученными весами из HLOC."""

    checkpoint_urls = {
        "VGG16-NetVLAD-Pitts30K": "https://cvg-data.inf.ethz.ch/hloc/netvlad/Pitts30K_struct.mat",
        "VGG16-NetVLAD-TokyoTM": "https://cvg-data.inf.ethz.ch/hloc/netvlad/TokyoTM_struct.mat",
    }

    def __init__(self, model_name: str, *, whiten: bool = True) -> None:
        super().__init__()
        if model_name not in self.checkpoint_urls:
            raise ValueError(f"Неизвестная модель NetVLAD: {model_name}")

        hub_dir = Path(torch.hub.get_dir()) / "netvlad"
        hub_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_file = hub_dir / f"{model_name}.mat"
        if not checkpoint_file.exists():
            url = self.checkpoint_urls[model_name]
            torch.hub.download_url_to_file(url, str(checkpoint_file))

        backbone = list(models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).children())[0]
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.netvlad = NetVLADLayer()
        self.whiten: nn.Module | None = nn.Linear(self.netvlad.output_dim, 4096) if whiten else None

        mat = loadmat(str(checkpoint_file), struct_as_record=False, squeeze_me=True)
        for layer, mat_layer in zip(self.backbone.children(), mat["net"].layers):
            if isinstance(layer, nn.Conv2d):
                weights = torch.tensor(mat_layer.weights[0]).float().permute([3, 2, 0, 1])
                bias = torch.tensor(mat_layer.weights[1]).float()
                layer.weight = nn.Parameter(weights)
                layer.bias = nn.Parameter(bias)

        score_w = mat["net"].layers[30].weights[0]
        centers = -mat["net"].layers[30].weights[1]
        score_w = torch.tensor(score_w).float().permute([1, 0]).unsqueeze(-1)
        centers = torch.tensor(centers).float()
        self.netvlad.score_proj.weight = nn.Parameter(score_w)
        self.netvlad.centers = nn.Parameter(centers)

        if self.whiten is not None:
            w = mat["net"].layers[33].weights[0]
            b = mat["net"].layers[33].weights[1]
            w = torch.tensor(w).float().squeeze().permute([1, 0])
            b = torch.tensor(b.squeeze()).float()
            self.whiten.weight = nn.Parameter(w)
            self.whiten.bias = nn.Parameter(b)

        preprocess_mean = mat["net"].meta.normalization.averageImage[0, 0]
        preprocess_std = np.array([1, 1, 1], dtype=np.float32)
        self.register_buffer("preprocess_mean", torch.tensor(preprocess_mean).float())
        self.register_buffer("preprocess_std", torch.tensor(preprocess_std).float())

    def forward(self, image: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if image.shape[1] != 3:
            raise FeatureExtractionError("NetVLAD ожидает трехканальное изображение")
        image = torch.clamp(image * 255.0, 0.0, 255.0)
        image = image - self.preprocess_mean.view(1, -1, 1, 1)
        image = image / self.preprocess_std.view(1, -1, 1, 1)
        descriptors = self.backbone(image)
        batch, channels, _, _ = descriptors.size()
        descriptors = descriptors.view(batch, channels, -1)
        descriptors = F.normalize(descriptors, dim=1)
        desc = self.netvlad(descriptors)
        if self.whiten is not None:
            desc = self.whiten(desc)
            desc = F.normalize(desc, dim=1)
        return desc


class NetVLADGlobalExtractor:
    """Высокоуровневый экстрактор глобальных дескрипторов."""

    def __init__(self, *, device: torch.device, model_name: str = "VGG16-NetVLAD-Pitts30K") -> None:
        self._device = device
        self._model_name = model_name
        self._network = NetVLADBackbone(model_name).to(device).eval()
        self._log = get_logger("georag.global")
        self._log.info(
            "event=netvlad_init model=%s device=%s", self._model_name, device
        )

    def extract(self, image_bgr: np.ndarray) -> GlobalDescriptor:
        """Получить глобальный дескриптор из изображения."""
        if image_bgr.ndim != 3:
            raise FeatureExtractionError("Ожидалось цветное изображение BGR")
        image_bgr = ensure_supported_size(image_bgr)
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb.astype(np.float32) / 255.0)
        tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(self._device)
        try:
            with torch.inference_mode():
                descriptor = self._network(tensor)
        except Exception as exc:  # pragma: no cover - исключения PyTorch
            self._log.exception("event=netvlad_forward_failed model=%s", self._model_name)
            raise FeatureExtractionError("Ошибка при расчёте глобального дескриптора") from exc
        vector = descriptor.detach().cpu().numpy().reshape(-1).astype(np.float32)
        self._log.debug(
            "event=netvlad_descriptor_ready dimension=%s norm=%.6f",
            vector.size,
            float(np.linalg.norm(vector)),
        )
        return GlobalDescriptor(vector=vector)

    async def aextract(self, image_bgr: np.ndarray) -> GlobalDescriptor:
        """Асинхронная версия получения дескриптора."""
        return await asyncio.to_thread(self.extract, image_bgr)
