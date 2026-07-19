from __future__ import annotations

import contextlib
import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path

from . import __version__
from .util import read_json, utc_now, write_json


def clear_pipeline_cache(cache_dir: Path) -> dict[str, int]:
    """Remove pipeline-owned cache children while preserving the configured root."""
    root = cache_dir.expanduser().resolve()
    if root == Path(root.anchor):
        raise ValueError("Refusing to clear a filesystem root as the pipeline cache.")
    if not root.exists():
        root.mkdir(parents=True, exist_ok=True)
        return {"files_removed": 0, "directories_removed": 0, "bytes_removed": 0}
    if not root.is_dir():
        raise ValueError(f"Pipeline cache path is not a directory: {root}")
    files = 0
    directories = 0
    byte_count = 0
    for path in root.rglob("*"):
        if path.is_file() and not path.is_symlink():
            files += 1
            with contextlib.suppress(OSError):
                byte_count += path.stat().st_size
        elif path.is_dir() and not path.is_symlink():
            directories += 1
    for child in root.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()
    return {"files_removed": files, "directories_removed": directories, "bytes_removed": byte_count}


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
