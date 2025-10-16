"""Визуализация 3D-точек из ответа API поиска."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def _load_response(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _extract_points(payload: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    coords = np.array([[p["x"], p["y"], p["z"]] for p in payload], dtype=np.float32)
    scores = np.array([p.get("score", 0.0) for p in payload], dtype=np.float32)
    return coords, scores


def _transform_candidate(
    points: np.ndarray,
    rotation: list[float] | None,
    translation: list[float] | None,
    to_query: bool,
) -> np.ndarray:
    if not points.size or rotation is None or translation is None:
        return points
    rot = np.asarray(rotation, dtype=np.float32)
    if rot.size == 9:
        rot = rot.reshape(3, 3)
    elif rot.size == 16:
        rot = rot.reshape(4, 4)[:3, :3]
    else:
        raise ValueError("rotation must contain 9 or 16 values")
    t = np.asarray(translation, dtype=np.float32).reshape(3)
    if to_query:
        return (rot.T @ (points.T - t.reshape(3, 1))).T
    return (rot @ points.T + t.reshape(3, 1)).T


def plot_point_cloud(
    response: dict[str, Any],
    *,
    match_index: int,
    limit: int | None,
    save_path: Path | None,
) -> None:
    matches = response.get("matches") or []
    if not matches:
        raise ValueError("Ответ не содержит совпадений")
    if match_index < 0 or match_index >= len(matches):
        raise ValueError(f"match_index должен быть от 0 до {len(matches) - 1}")

    query_coords, query_scores = _extract_points(response.get("query_point_cloud", []))
    if limit is not None and query_coords.shape[0] > limit:
        order = np.argsort(-query_scores)[:limit]
        query_coords = query_coords[order]
        query_scores = query_scores[order]

    match = matches[match_index]
    candidate_coords, candidate_scores = _extract_points(match.get("point_cloud", []))
    if limit is not None and candidate_coords.shape[0] > limit:
        order = np.argsort(-candidate_scores)[:limit]
        candidate_coords = candidate_coords[order]
        candidate_scores = candidate_scores[order]

    candidate_in_query = _transform_candidate(
        candidate_coords,
        match.get("relative_rotation"),
        match.get("relative_translation"),
        to_query=True,
    )

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    if query_coords.size:
        ax.scatter(
            query_coords[:, 0],
            query_coords[:, 1],
            query_coords[:, 2],
            c=query_scores,
            cmap="viridis",
            s=8,
            alpha=0.9,
            label="Query",
        )

    if candidate_in_query.size:
        ax.scatter(
            candidate_in_query[:, 0],
            candidate_in_query[:, 1],
            candidate_in_query[:, 2],
            c=candidate_scores,
            cmap="plasma",
            s=8,
            alpha=0.6,
            label=f"Match #{match_index}",
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend(loc="upper right")
    ax.set_title("3D Reconstruction from Matches")
    ax.view_init(elev=30, azim=45)
    ax.grid(False)

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", dpi=200)
    else:
        plt.show()

    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot 3D points from search response")
    parser.add_argument("response", type=Path, help="Путь до JSON с ответом поиска")
    parser.add_argument(
        "--match-index", type=int, default=0, help="Номер совпадения для визуализации"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Ограничение количества точек для отображения",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Путь для сохранения изображения вместо интерактивного окна",
    )
    args = parser.parse_args()

    response = _load_response(args.response)
    plot_point_cloud(
        response,
        match_index=args.match_index,
        limit=args.limit,
        save_path=args.save,
    )


if __name__ == "__main__":
    main()
