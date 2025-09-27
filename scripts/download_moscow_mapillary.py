"""Скрипт загрузки выборки Mapillary по Московской области."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

import httpx

LOG = logging.getLogger("georag.mapillary")

API_URL = "https://graph.mapillary.com/images"
FIELDS = (
    "id,thumb_2048_url,thumb_1024_url,geometry,captured_at,compass_angle,"
    "sequence,creator,id"
)


def iter_images(
    client: httpx.Client,
    *,
    bbox: str,
    limit: int,
) -> list[dict[str, Any]]:
    """Получить список изображений Mapillary в пределах ограничителя."""

    url = API_URL
    params: dict[str, Any] | None = {
        "bbox": bbox,
        "limit": min(100, limit),
        "fields": FIELDS,
    }
    fetched = 0
    images: list[dict[str, Any]] = []

    while url and fetched < limit:
        response = client.get(url, params=params)
        response.raise_for_status()
        payload = response.json()
        for item in payload.get("data", []):
            images.append(item)
            fetched += 1
            if fetched >= limit:
                break
        url = payload.get("paging", {}).get("next")
        params = None

    return images


def download_image(client: httpx.Client, url: str, path: Path) -> None:
    """Скачать изображение и сохранить на диск."""

    response = client.get(url)
    response.raise_for_status()
    path.write_bytes(response.content)


def build_metadata(item: dict[str, Any], relative_path: str) -> dict[str, Any]:
    """Сформировать запись метаданных для ingestion."""

    geometry = item.get("geometry") or {}
    coordinates = geometry.get("coordinates") or [None, None]
    longitude, latitude = coordinates[:2]
    metadata = {
        "filename": relative_path,
        "latitude": latitude,
        "longitude": longitude,
        "captured_at": item.get("captured_at"),
        "compass_angle": item.get("compass_angle"),
        "sequence_id": (item.get("sequence") or {}).get("id"),
        "creator_id": (item.get("creator") or {}).get("id"),
    }
    return metadata


def download_dataset(
    *,
    token: str,
    output_dir: Path,
    bbox: str,
    limit: int,
) -> None:
    """Загрузить изображения Mapillary и подготовить манифест."""

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.jsonl"
    session_headers = {"Authorization": f"OAuth {token}"}

    with httpx.Client(timeout=30.0, headers=session_headers) as client:
        images = iter_images(client, bbox=bbox, limit=limit)
        if not images:
            LOG.warning("event=no_mapillary_data bbox=%s", bbox)
            return
        LOG.info("event=mapillary_candidates count=%s", len(images))
        with metadata_path.open("w", encoding="utf-8") as meta_file:
            for item in images:
                image_id = item.get("id")
                if not image_id:
                    continue
                thumb_url = item.get("thumb_2048_url") or item.get("thumb_1024_url")
                if not thumb_url:
                    LOG.warning("event=missing_thumbnail id=%s", image_id)
                    continue
                image_path = output_dir / f"{image_id}.jpg"
                if image_path.exists():
                    LOG.info("event=skip_exists path=%s", image_path)
                else:
                    download_image(client, thumb_url, image_path)
                relative = image_path.relative_to(output_dir.parent).as_posix()
                metadata = build_metadata(item, relative_path=relative)
                meta_file.write(json.dumps(metadata, ensure_ascii=False))
                meta_file.write("\n")
                LOG.info(
                    "event=image_downloaded id=%s lat=%s lon=%s", image_id, metadata["latitude"], metadata["longitude"]
                )


def main() -> None:
    parser = argparse.ArgumentParser(description="Скачать выборку Mapillary для Московской области")
    parser.add_argument(
        "--token",
        dest="token",
        default=os.getenv("MAPILLARY_TOKEN"),
        help="Mapillary access token или переменная окружения MAPILLARY_TOKEN",
    )
    parser.add_argument(
        "--output",
        dest="output",
        type=Path,
        default=Path("train_data/mapillary_moscow"),
        help="Каталог назначения",
    )
    parser.add_argument(
        "--bbox",
        dest="bbox",
        default="36.6,54.5,39.2,56.5",
        help="Границы поиска (minLon,minLat,maxLon,maxLat)",
    )
    parser.add_argument(
        "--limit",
        dest="limit",
        type=int,
        default=150,
        help="Количество изображений для загрузки",
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    if not args.token:
        raise SystemExit("Необходимо передать MAPILLARY_TOKEN через аргумент или окружение")

    download_dataset(token=args.token, output_dir=args.output, bbox=args.bbox, limit=args.limit)


if __name__ == "__main__":
    main()
