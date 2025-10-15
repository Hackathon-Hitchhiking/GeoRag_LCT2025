"""CLI-скрипт для наполнения базы изображениями из каталога."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from logging.handlers import WatchedFileHandler
from pathlib import Path
from typing import Any

import torch

from app.application.feature_store import FeatureStore
from app.application.features import LocalFeatureSet, SuperPointFeatureExtractor
from app.application.geocoder import Geocoder
from app.application.global_descriptors import NetVLADGlobalExtractor
from app.application.ingestion import ImageIngestionService, IngestionPayload
from app.core.config import get_settings
from app.infrastructure.database.session import Database
from app.infrastructure.storage import S3Storage
from app.infrastructure.vector.qdrant import QdrantVectorStore

LOG = logging.getLogger("georag.ingest")


def setup_logging(log_path: Path | None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(WatchedFileHandler(log_path, encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers,
    )


ALLOWED_METADATA_KEYS = {
    "speed",
    "angle",
    "create_timestamp",
    "device",
    "camera",
}


def normalize_metadata_dict(
    source: dict[str, Any] | None, *, strict: bool = False
) -> dict[str, Any] | None:
    """Нормализовать имена и (опционально) отфильтровать поля."""

    if not source:
        return None

    metadata = dict(source)

    angle = metadata.pop("angle", None)
    if angle is None and "compass_angle" in metadata:
        angle = metadata.pop("compass_angle")

    create_ts = metadata.pop("create_timestamp", None)
    if create_ts is None and "captured_at" in metadata:
        create_ts = metadata.pop("captured_at")

    if angle is not None:
        metadata["angle"] = angle
    if create_ts is not None:
        metadata["create_timestamp"] = create_ts

    if strict:
        metadata = {k: v for k, v in metadata.items() if k in ALLOWED_METADATA_KEYS}

    return metadata or None


def iter_images(data_dir: Path) -> list[Path]:
    supported = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return [p for p in data_dir.rglob("*") if p.suffix.lower() in supported]


def load_manifest(data_dir: Path) -> dict[str, dict[str, Any]]:
    """Загрузить объединённый манифест по изображениям.

    Поддерживаются `metadata.jsonl`, `manifest.jsonl` и `metadata.json`.
    """

    manifest: dict[str, dict[str, Any]] = {}
    for name in ("metadata.jsonl", "manifest.jsonl"):
        for path in data_dir.rglob(name):
            if not path.is_file():
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

    for json_path in data_dir.rglob("metadata.json"):
        if not json_path.is_file():
            continue
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        results = data.get("results")
        if not isinstance(results, list):
            continue
        base_dir = json_path.parent
        for item in results:
            if not isinstance(item, dict):
                continue
            image_id = item.get("id")
            if not image_id:
                continue
            relative_dir = base_dir.relative_to(data_dir)
            image_path = None
            for ext in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"):
                candidate = base_dir / f"{image_id}{ext}"
                if candidate.exists():
                    image_path = candidate.relative_to(data_dir).as_posix()
                    break
            if image_path is None:
                if relative_dir == Path('.'):
                    image_path = f"{image_id}.jpg"
                else:
                    image_path = f"{relative_dir.as_posix()}/{image_id}.jpg"

            entry = {
                "filename": image_path,
                "latitude": item.get("latitude"),
                "longitude": item.get("longitude"),
            }
            meta_raw = {
                "speed": item.get("speed"),
                "angle": item.get("angle"),
                "compass_angle": item.get("compass_angle"),
                "create_timestamp": item.get("create_timestamp"),
                "captured_at": item.get("captured_at"),
                "device": item.get("device"),
                "camera": item.get("camera"),
            }
            normalized = normalize_metadata_dict(meta_raw, strict=True)
            if normalized:
                entry.update(normalized)
            manifest[image_path] = entry
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
        metadata_raw = json.loads(meta_path.read_text(encoding="utf-8"))
        latitude = metadata_raw.pop("latitude", None)
        longitude = metadata_raw.pop("longitude", None)
        metadata = normalize_metadata_dict(metadata_raw)
        return metadata, latitude, longitude
    return None, None, None


def compose_metadata(
    image_path: Path,
    data_dir: Path,
    manifest: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any] | None, float | None, float | None]:
    """Собрать метаданные из манифеста и побочного JSON."""

    relative_key = image_path.relative_to(data_dir).as_posix()
    manifest_entry = manifest.get(relative_key) or manifest.get(image_path.name)
    latitude: float | None = None
    longitude: float | None = None
    manifest_meta: dict[str, Any] = {}
    if manifest_entry:
        manifest_meta_raw = {
            k: v for k, v in manifest_entry.items() if k not in {"filename", "latitude", "longitude"}
        }
        manifest_meta = normalize_metadata_dict(manifest_meta_raw) or {}
        latitude = manifest_entry.get("latitude")
        longitude = manifest_entry.get("longitude")
    
    sidecar_meta, sidecar_lat, sidecar_lon = load_sidecar_metadata(image_path)
    combined_meta = {**manifest_meta} if manifest_meta else {}
    if sidecar_meta:
        combined_meta.update(sidecar_meta)
    metadata = combined_meta or None
    latitude = sidecar_lat if sidecar_lat is not None else latitude
    longitude = sidecar_lon if sidecar_lon is not None else longitude
    return metadata, latitude, longitude


async def ingest_directory(data_dir: Path) -> None:
    settings = get_settings()
    database = Database(settings.database_dsn, echo=settings.database_echo)
    await database.init_models()

    storage = S3Storage(
        bucket=settings.s3_bucket,
        region=settings.s3_region,
        endpoint_url=settings.s3_endpoint_url,
        access_key=settings.s3_access_key,
        secret_key=settings.s3_secret_key,
        session_token=settings.s3_session_token,
        public_base_url=settings.s3_public_base_url,
        presign_ttl=settings.s3_presign_ttl,
        max_parallel=settings.s3_max_parallel,
    )
    if settings.compute_device:
        device = torch.device(settings.compute_device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    local_store = FeatureStore[LocalFeatureSet](
        storage,
        prefix=settings.feature_subdir,
        serializer=lambda pack: pack.to_bytes(),
        deserializer=LocalFeatureSet.from_bytes,
    )
    vector_store = QdrantVectorStore(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        collection=settings.qdrant_collection,
        vector_dim=4096,
        on_disk=settings.qdrant_on_disk,
        shard_number=settings.qdrant_shard_number,
        replication_factor=settings.qdrant_replication_factor,
    )
    await vector_store.ensure_collection()
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
        local_extractor=local_extractor,
        global_extractor=global_extractor,
        geocoder=geocoder,
        vector_store=vector_store,
        image_prefix=settings.image_subdir,
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
                    "event=image_ingested id=%s url=%s lat=%s lon=%s",
                    stored.record_id,
                    stored.image_url,
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
    parser.add_argument(
        "--log-file",
        default="logs/ingest.log",
        help="Путь до файла, куда сохранять ход импорта",
    )
    args = parser.parse_args()

    log_path = Path(args.log_file).resolve() if args.log_file else None
    setup_logging(log_path)
    data_dir = Path(args.data_dir).resolve()
    if not data_dir.exists():
        raise SystemExit(f"Каталог {data_dir} не найден")

    asyncio.run(ingest_directory(data_dir))


if __name__ == "__main__":
    main()
