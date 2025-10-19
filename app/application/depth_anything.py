from __future__ import annotations

# Обёртка над Depth Anything V2 для построения облака точек.

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from PIL import Image

from depth_anything_v2.dpt import DepthAnythingV2
from huggingface_hub import hf_hub_download


@dataclass(slots=True)
class DepthPoint:
    """Описание точки облака с координатами в трёхмерном пространстве."""

    x: float
    y: float
    z: float


class DepthAnythingPointCloudGenerator:
    """Построитель облака точек на основе Depth Anything V2.

    Экземпляр отвечает за загрузку весов модели, получение карты глубины из
    изображения и перевод её в трёхмерное облако точек в приближённой
    камере-пинхол. Реализация ориентирована на использование оригинальной
    модели из репозитория Depth Anything V2 и кэширует веса через
    ``huggingface_hub``.
    """

    def __init__(
        self,
        *,
        encoder: str = "vitl",
        features: int = 256,
        out_channels: Sequence[int] = (256, 512, 1024, 1024),
        repo_id: str = "depth-anything/Depth-Anything-V2-Large",
        filename: str = "depth_anything_v2_vitl.pth",
        device: str | torch.device | None = None,
        default_sample_step: int = 4,
    ) -> None:
        if DepthAnythingV2 is None or hf_hub_download is None:
            raise ImportError(
                "DepthAnythingPointCloudGenerator требует библиотеку"
                " huggingface-hub и модуль depth_anything_v2 из"
                " https://github.com/DepthAnything/Depth-Anything-V2"
            )

        if isinstance(device, torch.device):
            self._device = device
        elif isinstance(device, str):
            self._device = torch.device(device)
        else:
            if torch.cuda.is_available():
                self._device = torch.device("cuda")
            elif torch.backends.mps.is_available():  # pragma: no cover - macOS
                self._device = torch.device("mps")
            else:
                self._device = torch.device("cpu")

        weight_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="model")

        self._model = DepthAnythingV2(
            encoder=encoder,
            features=features,
            out_channels=list(out_channels),
        )
        state_dict = torch.load(weight_path, map_location=self._device)
        self._model.load_state_dict(state_dict)
        self._model.to(self._device)
        self._model.eval()

        self._default_sample_step = max(1, int(default_sample_step))

    @staticmethod
    def _to_bgr(image: np.ndarray | Image.Image) -> np.ndarray:
        if isinstance(image, Image.Image):
            if image.mode not in {"RGB", "RGBA"}:
                image = image.convert("RGB")
            array = np.array(image, dtype=np.uint8)
            if array.ndim != 3 or array.shape[2] < 3:
                raise ValueError("Ожидается цветное изображение")
            return array[:, :, :3][:, :, ::-1]
        if isinstance(image, np.ndarray):
            if image.ndim != 3 or image.shape[2] < 3:
                raise ValueError("Ожидается массив формы (H, W, 3) или (H, W, 4)")
            if image.dtype != np.uint8:
                array = np.clip(image, 0, 255).astype(np.uint8)
            else:
                array = image
            if array.shape[2] == 3:
                return array
            return array[:, :, :3]
        raise TypeError("Поддерживаются только PIL.Image.Image и numpy.ndarray")

    def infer_depth(self, image: np.ndarray | Image.Image) -> np.ndarray:
        """Получить карту глубины для входного изображения."""

        bgr = self._to_bgr(image)
        with torch.no_grad():
            depth = self._model.infer_image(bgr)
        depth_array = np.asarray(depth, dtype=np.float32)
        if depth_array.ndim != 2:
            raise ValueError("Карта глубины должна быть двумерной")
        return depth_array

    @staticmethod
    def _camera_intrinsics(
        height: int,
        width: int,
        intrinsics: Optional[Dict[str, float]],
    ) -> tuple[float, float, float, float]:
        if not intrinsics:
            focal = float(max(height, width))
            return focal, focal, float(width) / 2.0, float(height) / 2.0

        def _pick(*keys: str, default: float) -> float:
            for key in keys:
                if key in intrinsics and intrinsics[key] is not None:
                    return float(intrinsics[key])
            return default

        fx = _pick("fx", "f_x", "f", default=float(max(height, width)))
        fy = _pick("fy", "f_y", "f", default=float(max(height, width)))
        cx = _pick("cx", "c_x", default=float(width) / 2.0)
        cy = _pick("cy", "c_y", default=float(height) / 2.0)
        return fx, fy, cx, cy

    def generate_point_cloud(
        self,
        image: np.ndarray | Image.Image,
        *,
        intrinsics: Optional[Dict[str, float]] = None,
        sample_step: Optional[int] = None,
        max_points: Optional[int] = None,
    ) -> List[DepthPoint]:
        """Построить облако точек для изображения."""

        bgr = self._to_bgr(image)
        depth_map = self.infer_depth(bgr)
        height, width = depth_map.shape

        fx, fy, cx, cy = self._camera_intrinsics(height, width, intrinsics)

        step = self._default_sample_step if sample_step is None else max(1, int(sample_step))
        if max_points and max_points > 0:
            approx_points = (math.ceil(height / step) * math.ceil(width / step))
            if approx_points > max_points:
                scale = math.sqrt(approx_points / max_points)
                step = max(step, int(math.ceil(step * scale)))

        points: List[DepthPoint] = []
        for v in range(0, height, step):
            row_depth = depth_map[v]
            for u in range(0, width, step):
                depth_value = float(row_depth[u])
                if not math.isfinite(depth_value) or depth_value <= 0:
                    continue
                x = (u - cx) * depth_value / fx
                y = (v - cy) * depth_value / fy
                points.append(DepthPoint(x=x, y=y, z=depth_value))
        return points


__all__ = ["DepthPoint", "DepthAnythingPointCloudGenerator"]
