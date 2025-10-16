"""Построение 3D-облака ключевых точек SuperPoint для произвольного изображения."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - регистрирует 3D-проекцию в Matplotlib

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.application.features import SuperPointFeatureExtractor
from app.core.config import get_settings
from app.utils.image import decode_image


def resolve_device(explicit: str | None) -> torch.device:
    """Подобрать устройство для инференса."""

    if explicit:
        return torch.device(explicit)
    settings = get_settings()
    if settings.compute_device:
        return torch.device(settings.compute_device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_extractor(device: torch.device, max_keypoints: int | None) -> SuperPointFeatureExtractor:
    """Создать извлекатель признаков SuperPoint."""

    settings = get_settings()
    keypoint_limit = max_keypoints if max_keypoints is not None else settings.feature_max_keypoints
    return SuperPointFeatureExtractor(
        device=device,
        max_keypoints=keypoint_limit,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Построить интерактивное 3D-облако ключевых точек для изображения.",
    )
    parser.add_argument("image", type=Path, help="Путь до изображения (jpg/png/bmp/...)")
    parser.add_argument(
        "--point-limit",
        type=int,
        default=2048,
        help="Максимальное количество точек в облаке (по score).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Явно указать устройство (например, cpu, cuda, cuda:0). По умолчанию берётся из настроек.",
    )
    parser.add_argument(
        "--max-keypoints",
        type=int,
        default=None,
        help="Ограничение SuperPoint по числу ключевых точек. По умолчанию из конфигурации приложения.",
    )
    parser.add_argument(
        "--mode",
        choices=("image", "rays"),
        default="image",
        help="Формат визуализации: 'image' — ключевые точки поверх изображения, 'rays' — нормализованные 3D-лучи.",
    )
    return parser.parse_args()


def plot_point_cloud(points: np.ndarray, scores: np.ndarray, title: str) -> None:
    """Отрисовать облако точек с цветовой картой по score."""

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        c=scores,
        cmap="viridis",
        s=10,
        alpha=0.9,
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)
    fig.colorbar(scatter, ax=ax, shrink=0.75, label="Confidence score")
    plt.tight_layout()
    plt.show()


def plot_on_image(image_bgr: np.ndarray, keypoints: np.ndarray, scores: np.ndarray, title: str) -> None:
    """Отобразить ключевые точки поверх исходного изображения."""

    image_rgb = image_bgr[:, :, ::-1]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(image_rgb)
    scatter = ax.scatter(
        keypoints[:, 0],
        keypoints[:, 1],
        c=scores,
        cmap="viridis",
        s=20,
        alpha=0.9,
    )
    ax.set_xlim(0, image_rgb.shape[1])
    ax.set_ylim(image_rgb.shape[0], 0)
    ax.set_title(title)
    ax.axis("off")
    fig.colorbar(scatter, ax=ax, shrink=0.75, label="Confidence score")
    plt.tight_layout()
    plt.show()


def main() -> None:
    args = parse_args()
    image_path: Path = args.image.expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Изображение {image_path} не найдено")

    device = resolve_device(args.device)
    extractor = build_extractor(device=device, max_keypoints=args.max_keypoints)

    image_bytes = image_path.read_bytes()
    image = decode_image(image_bytes)
    features = extractor.extract(image)

    if args.mode == "rays":
        _, rays, scores = features.to_point_cloud(args.point_limit)
        if rays.size == 0:
            raise RuntimeError("SuperPoint не нашёл ключевых точек на изображении")
        plot_point_cloud(rays, scores, title=f"Point cloud for {image_path.name}")
        return

    keypoints = features.keypoints
    scores = features.scores
    if keypoints.size == 0:
        raise RuntimeError("SuperPoint не нашёл ключевых точек на изображении")
    if args.point_limit and args.point_limit > 0 and keypoints.shape[0] > args.point_limit:
        idx = np.argpartition(-scores, args.point_limit - 1)[: args.point_limit]
        order = np.argsort(-scores[idx])
        idx = idx[order]
        keypoints = keypoints[idx]
        scores = scores[idx]

    plot_on_image(image, keypoints, scores, title=f"Keypoints for {image_path.name}")


if __name__ == "__main__":
    main()
