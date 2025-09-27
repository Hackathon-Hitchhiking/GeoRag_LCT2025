"""CLI-скрипт для наполнения базы изображениями из каталога."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path
from typing import Any

import torch

from app.application.feature_store import FeatureStore
from app.application.features import LocalFeatureSet, SuperPointFeatureExtractor
from app.application.geocoder import Geocoder
from app.application.global_descriptors import GlobalDescriptor, NetVLADGlobalExtractor
from app.application.ingestion import ImageIngestionService, IngestionPayload
from app.core.config import get_settings
from app.infrastructure.database.session import Database
from app.infrastructure.storage.s3 import S3Storage

LOG = logging.getLogger("georag.ingest")


def iter_images(data_dir: Path) -> list[Path]:
    supported = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return [p for p in data_dir.rglob("*") if p.suffix.lower() in supported]


def load_manifest(data_dir: Path) -> dict[str, dict[str, Any]]:
    """Загрузить объединённый манифест по изображениям.

    Поддерживаются `metadata.jsonl` и `manifest.jsonl`, каждая строка которых
    описывает одно изображение с обязательным полем `filename` (относительный
    путь от корня каталога) и произвольными дополнительными атрибутами.
    """

    manifest: dict[str, dict[str, Any]] = {}
    for name in ("metadata.jsonl", "manifest.jsonl"):
        path = data_dir / name
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            filename = payload.get("filename")
            if not filename:
                continue
            key = Path(filename).as_posix()
            manifest[key] = payload
    return manifest


def load_sidecar_metadata(image_path: Path) -> tuple[dict[str, Any] | None, float | None, float | None]:
    """Прочитать метаданные из соседних файлов, если они есть."""

    candidates = [
        image_path.with_suffix(".json"),
        image_path.with_suffix(".geojson"),
        image_path.with_suffix(".meta.json"),
    ]
    for meta_path in candidates:
        if not meta_path.exists():
            continue
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        latitude = metadata.pop("latitude", None)
        longitude = metadata.pop("longitude", None)
        return metadata or None, latitude, longitude
    return None, None, None


def compose_metadata(
    image_path: Path,
    data_dir: Path,
    manifest: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any] | None, float | None, float | None]:
    """Собрать метаданные из манифеста и побочного JSON."""

    relative_key = image_path.relative_to(data_dir).as_posix()
    manifest_entry = manifest.get(relative_key) or manifest.get(image_path.name)
    manifest_meta: dict[str, Any] = {}
    latitude: float | None = None
    longitude: float | None = None
    if manifest_entry:
        manifest_meta = {
            k: v for k, v in manifest_entry.items() if k not in {"filename", "latitude", "longitude"}
        }
        latitude = manifest_entry.get("latitude")
        longitude = manifest_entry.get("longitude")

    sidecar_meta, sidecar_lat, sidecar_lon = load_sidecar_metadata(image_path)
    metadata = {**manifest_meta, **(sidecar_meta or {})} if manifest_meta or sidecar_meta else None
    latitude = sidecar_lat if sidecar_lat is not None else latitude
    longitude = sidecar_lon if sidecar_lon is not None else longitude
    return metadata, latitude, longitude


async def ingest_directory(data_dir: Path) -> None:
    settings = get_settings()
    database = Database(settings.database_dsn, echo=settings.database_echo)
    await database.init_models()

    storage = S3Storage(
        bucket=settings.s3_bucket,
        endpoint_url=settings.s3_endpoint_url,
        region_name=settings.s3_region,
        access_key=settings.s3_access_key,
        secret_key=settings.s3_secret_key,
        use_ssl=settings.s3_use_ssl,
    )
    device = torch.device("cuda" if settings.prefer_gpu and torch.cuda.is_available() else "cpu")

    local_store = FeatureStore[LocalFeatureSet](
        storage,
        prefix=settings.s3_features_prefix,
        serializer=lambda pack: pack.to_bytes(),
        deserializer=LocalFeatureSet.from_bytes,
    )
    global_store = FeatureStore[GlobalDescriptor](
        storage,
        prefix=settings.s3_global_prefix,
        serializer=lambda desc: desc.to_bytes(),
        deserializer=GlobalDescriptor.from_bytes,
    )
    local_extractor = SuperPointFeatureExtractor(
        device=device,
        max_keypoints=settings.feature_max_keypoints,
    )
    global_extractor = NetVLADGlobalExtractor(device=device)
    geocoder = Geocoder(
        user_agent=settings.nominatim_user_agent,
        timeout=settings.nominatim_timeout,
    )
    ingestion_service = ImageIngestionService(
        database=database,
        storage=storage,
        local_store=local_store,
        global_store=global_store,
        local_extractor=local_extractor,
        global_extractor=global_extractor,
        geocoder=geocoder,
        image_prefix=settings.s3_images_prefix,
        local_feature_type=settings.local_feature_type,
        global_descriptor_type=settings.global_descriptor_type,
        matcher_type=settings.matcher_type,
    )

    try:
        images = iter_images(data_dir)
        if not images:
            LOG.warning("event=no_images_found directory=%s", data_dir)
            return
        manifest = load_manifest(data_dir)
        for path in images:
            metadata, lat, lon = compose_metadata(path, data_dir, manifest)
            payload = IngestionPayload(
                data=path.read_bytes(),
                latitude=lat,
                longitude=lon,
                metadata=metadata,
                source_name=path.stem,
            )
            try:
                stored = await ingestion_service.ingest(payload)
                LOG.info(
                    "event=image_ingested id=%s key=%s lat=%s lon=%s",
                    stored.record_id,
                    stored.image_key,
                    stored.latitude,
                    stored.longitude,
                )
            except Exception as exc:  # pragma: no cover - утилита
                LOG.exception("event=ingest_failed file=%s error=%s", path, exc)
    finally:
        await database.aclose()


def main() -> None:
    parser = argparse.ArgumentParser(description="Импортировать изображения из каталога")
    parser.add_argument(
        "data_dir",
        nargs="?",
        default="train_data",
        help="Каталог с обучающими изображениями",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    data_dir = Path(args.data_dir).resolve()
    if not data_dir.exists():
        raise SystemExit(f"Каталог {data_dir} не найден")

    asyncio.run(ingest_directory(data_dir))


if __name__ == "__main__":
    main()
