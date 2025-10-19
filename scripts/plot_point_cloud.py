"""Интерактивная визуализация облака точек Depth Anything V2."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from app.application.depth_anything import (
    DepthAnythingPointCloudGenerator,
    GroundPlaneFilterConfig,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Построить облако точек Depth Anything V2 и визуализировать его в 3D"
        )
    )
    parser.add_argument("image", type=Path, help="Путь к входному изображению")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Устройство для инференса (cuda, mps или cpu)",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=50000,
        help="Максимальное количество точек в облаке",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=None,
        help="Явный шаг дискретизации пикселей (по умолчанию вычисляется автоматически)",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Путь для сохранения изображения вместо открытия интерактивного окна",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=1.5,
        help="Размер точек на графике",
    )
    parser.add_argument(
        "--clip-percentiles",
        type=float,
        nargs=2,
        metavar=("LOWER", "UPPER"),
        default=None,
        help="Диапазон перцентилей глубины для обрезки (0–100). По умолчанию используется значение из генератора",
    )
    parser.add_argument(
        "--no-depth-clip",
        action="store_true",
        help="Отключить фильтрацию по перцентилям глубины",
    )
    parser.add_argument(
        "--disable-ground-filter",
        action="store_true",
        help="Не удалять плоскость земли из облака точек",
    )
    parser.add_argument(
        "--ground-distance",
        type=float,
        default=0.05,
        help="Допустимое расстояние до плоскости земли (в долях медианы глубины)",
    )
    parser.add_argument(
        "--ground-min-normal",
        type=float,
        default=0.7,
        help="Минимальная компонента нормали по оси Y для распознавания земли",
    )
    parser.add_argument(
        "--ground-min-ratio",
        type=float,
        default=0.08,
        help="Минимальная доля точек, принадлежащих плоскости земли",
    )
    parser.add_argument(
        "--ground-iterations",
        type=int,
        default=72,
        help="Количество итераций RANSAC для поиска земли",
    )
    return parser


def _scatter_points(
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    point_size: float,
    title: str,
) -> plt.Figure:
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    if zs.size == 0:
        color_values = zs
    else:
        depth_min = float(np.min(zs))
        depth_max = float(np.max(zs))
        if np.isclose(depth_min, depth_max):
            color_values = np.zeros_like(zs)
        else:
            color_values = (zs - depth_min) / (depth_max - depth_min)

    scatter = ax.scatter(
        xs,
        ys,
        zs,
        c=color_values,
        cmap="viridis",
        s=point_size,
        alpha=0.9,
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)
    ax.view_init(elev=30, azim=45)
    ax.grid(False)
    fig.colorbar(scatter, ax=ax, label="Relative depth")
    return fig


def visualize_depth_points(args: argparse.Namespace) -> None:
    if not args.image.exists():
        raise FileNotFoundError(f"Файл {args.image} не найден")

    image = cv2.imread(str(args.image))
    if image is None:
        raise ValueError(f"Не удалось открыть изображение {args.image}")

    clip_percentiles: tuple[float, float] | None
    if args.no_depth_clip:
        clip_percentiles = None
    elif args.clip_percentiles is None:
        clip_percentiles = (0.5, 99.5)
    else:
        clip_percentiles = (float(args.clip_percentiles[0]), float(args.clip_percentiles[1]))

    ground_config = GroundPlaneFilterConfig(
        enabled=not args.disable_ground_filter,
        min_normal_y=float(args.ground_min_normal),
        distance_threshold=float(args.ground_distance),
        min_inlier_ratio=float(args.ground_min_ratio),
        max_iterations=int(args.ground_iterations),
    )

    try:
        generator = DepthAnythingPointCloudGenerator(
            device=args.device,
            clip_percentiles=clip_percentiles,
            ground_plane_filter=ground_config,
        )
    except ImportError as exc:  # pragma: no cover - информируем пользователя CLI
        raise RuntimeError(
            "Не установлены зависимости depth-anything-v2 и huggingface-hub"
        ) from exc
    points = generator.generate_point_cloud(
        image,
        sample_step=args.step,
        max_points=args.max_points,
    )

    if not points:
        raise RuntimeError("Модель не вернула ни одной точки")

    xs = np.array([point.x for point in points], dtype=np.float32)
    ys = np.array([point.y for point in points], dtype=np.float32)
    zs = np.array([point.z for point in points], dtype=np.float32)
    title = f"Depth Anything V2 — {args.image.name}"
    fig = _scatter_points(xs, ys, zs, args.point_size, title)

    if args.save is not None:
        fig.savefig(args.save, dpi=200, bbox_inches="tight")
    else:
        plt.show()

    plt.close(fig)


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    visualize_depth_points(args)


if __name__ == "__main__":
    main()
