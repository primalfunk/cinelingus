from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from . import __version__
from .util import read_json, utc_now, write_json


@dataclass(frozen=True)
class CacheEntry:
    media_hash: str
    media_path: Path
    cache_dir: Path
    role: str

    @property
    def manifest_path(self) -> Path:
        return self.cache_dir / "manifest.json"


def media_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_cache(base: Path, media_path: Path, role: str) -> CacheEntry:
    if not media_path.exists():
        raise FileNotFoundError(f"Missing {role}: {media_path}")
    h = media_hash(media_path)
    cache_dir = base / h / role
    cache_dir.mkdir(parents=True, exist_ok=True)
    entry = CacheEntry(media_hash=h, media_path=media_path, cache_dir=cache_dir, role=role)
    if entry.manifest_path.exists():
        existing = read_json(entry.manifest_path)
        cached_path = Path(str(existing.get("source_path") or "")).expanduser()
        try:
            matches = cached_path.resolve() == media_path.resolve()
        except OSError:
            matches = str(cached_path).casefold() == str(media_path).casefold()
        if not matches:
            raise RuntimeError(f"Cache identity mismatch for {role}: cached {cached_path}, requested {media_path}")
        if str(existing.get("role") or "") != role:
            raise RuntimeError(f"Cache role mismatch: cached {existing.get('role')}, requested {role}")
    else:
        write_json(
            entry.manifest_path,
            {
                "schema_version": "1.0",
                "tool_version": __version__,
                "media_hash": h,
                "creation_timestamp": utc_now(),
                "updated_timestamp": utc_now(),
                "role": role,
                "source_path": str(media_path),
                "processing_status": "initialized",
                "artifacts": {},
            },
        )
    return entry


def update_manifest(entry: CacheEntry, status: str, artifacts: dict[str, str]) -> None:
    if entry.manifest_path.exists():
        manifest = read_json(entry.manifest_path)
    else:
        manifest = {}
    merged_artifacts = dict(manifest.get("artifacts", {}))
    merged_artifacts.update(artifacts)
    manifest.update(
        {
            "schema_version": "1.0",
            "tool_version": __version__,
            "media_hash": entry.media_hash,
            "updated_timestamp": utc_now(),
            "role": entry.role,
            "source_path": str(entry.media_path),
            "processing_status": status,
            "artifacts": merged_artifacts,
        }
    )
    manifest.setdefault("creation_timestamp", utc_now())
    write_json(entry.manifest_path, manifest)
