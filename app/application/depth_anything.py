from __future__ import annotations

# Обёртка над Depth Anything V2 для построения облака точек.

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from PIL import Image

try:  # pragma: no cover - опциональные зависимости проверяются во время запуска
    from depth_anything_v2.dpt import DepthAnythingV2
except ModuleNotFoundError:  # pragma: no cover - в тестовой среде модель может отсутствовать
    DepthAnythingV2 = None  # type: ignore[assignment]

try:  # pragma: no cover - аналогично загрузчику весов
    from huggingface_hub import hf_hub_download
except ModuleNotFoundError:  # pragma: no cover
    hf_hub_download = None  # type: ignore[assignment]


@dataclass(slots=True)
class DepthPoint:
    """Описание точки облака с координатами в трёхмерном пространстве."""

    x: float
    y: float
    z: float


@dataclass(slots=True)
class GroundPlaneFilterConfig:
    """Параметры фильтрации плоскости земли в облаке точек.

    Атрибут ``distance_threshold`` трактуется как относительное значение от
    медианной глубины, если ``relative_distance`` равно ``True``.
    """

    enabled: bool = True
    min_normal_y: float = 0.7
    distance_threshold: float = 0.05
    min_inlier_ratio: float = 0.08
    max_iterations: int = 72
    relative_distance: bool = True
    random_seed: int = 1729


class DepthAnythingPointCloudGenerator:
    """Построитель облака точек на основе Depth Anything V2.

    Экземпляр отвечает за загрузку весов модели, получение карты глубины из
    изображения и перевод её в трёхмерное облако точек в приближённой
    камере-пинхол. Реализация ориентирована на использование оригинальной
    модели из репозитория Depth Anything V2 и кэширует веса через
    ``huggingface_hub``. Генератор дополнительно предоставляет фильтрацию
    глубины по перцентилям и удаление доминирующей плоскости земли, что
    помогает уменьшить количество артефактных точек на горизонте.
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
        clip_percentiles: tuple[float, float] | None = (0.5, 99.5),
        ground_plane_filter: GroundPlaneFilterConfig | None = None,
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
        if clip_percentiles is not None:
            lower, upper = clip_percentiles
            lower = float(lower)
            upper = float(upper)
            if not 0.0 <= lower < upper <= 100.0:
                raise ValueError("clip_percentiles должны быть в диапазоне [0, 100] и lower < upper")
            self._clip_percentiles: tuple[float, float] | None = (lower, upper)
        else:
            self._clip_percentiles = None
        self._ground_config = ground_plane_filter or GroundPlaneFilterConfig()

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

        xs: list[float] = []
        ys: list[float] = []
        zs: list[float] = []
        for v in range(0, height, step):
            row_depth = depth_map[v]
            for u in range(0, width, step):
                depth_value = float(row_depth[u])
                if not math.isfinite(depth_value) or depth_value <= 0:
                    continue
                x = (u - cx) * depth_value / fx
                y = (v - cy) * depth_value / fy
                xs.append(x)
                ys.append(y)
                zs.append(depth_value)

        if not xs:
            return []

        coords = np.column_stack((np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32), np.array(zs, dtype=np.float32)))
        mask = np.ones(coords.shape[0], dtype=bool)

        if self._clip_percentiles is not None:
            lower, upper = self._clip_percentiles
            z_values = coords[:, 2]
            lower_bound = float(np.percentile(z_values, lower))
            upper_bound = float(np.percentile(z_values, upper))
            if not math.isclose(lower_bound, upper_bound):
                mask &= (z_values >= lower_bound) & (z_values <= upper_bound)

        if self._ground_config.enabled:
            mask = self._remove_ground_plane(coords, mask)

        filtered = coords[mask]
        return [DepthPoint(x=float(x), y=float(y), z=float(z)) for x, y, z in filtered]

    def _remove_ground_plane(self, coords: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Удалить доминирующую плоскость земли из облака точек."""

        indices = np.flatnonzero(mask)
        if indices.size < 3:
            return mask

        subset = coords[indices]
        rng = np.random.default_rng(self._ground_config.random_seed)
        best_inliers: np.ndarray | None = None
        best_normal: np.ndarray | None = None
        best_count = 0

        z_median = float(np.median(subset[:, 2])) if subset.size else 1.0
        distance_threshold = float(self._ground_config.distance_threshold)
        if self._ground_config.relative_distance:
            distance_threshold *= max(1.0, z_median)

        for _ in range(max(1, int(self._ground_config.max_iterations))):
            sample_idx = rng.choice(subset.shape[0], size=3, replace=False)
            p0, p1, p2 = subset[sample_idx]
            v1 = p1 - p0
            v2 = p2 - p0
            normal = np.cross(v1, v2)
            norm = np.linalg.norm(normal)
            if norm <= 1e-6:
                continue
            normal = normal / norm
            if abs(float(normal[1])) < self._ground_config.min_normal_y:
                continue
            d = -float(np.dot(normal, p0))
            distances = np.abs(subset @ normal + d)
            inliers = distances <= distance_threshold
            count = int(np.count_nonzero(inliers))
            if count > best_count:
                best_count = count
                best_inliers = inliers
                best_normal = normal

        if best_inliers is None or best_normal is None:
            return mask

        if best_count < max(3, int(self._ground_config.min_inlier_ratio * subset.shape[0])):
            return mask

        if best_normal[1] < 0:
            best_normal = -best_normal
        if best_normal[1] < self._ground_config.min_normal_y:
            return mask

        mask_indices = indices[best_inliers]
        mask[mask_indices] = False
        return mask


__all__ = [
    "DepthPoint",
    "DepthAnythingPointCloudGenerator",
    "GroundPlaneFilterConfig",
]
