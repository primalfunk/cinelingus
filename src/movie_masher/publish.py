from __future__ import annotations

import re
import shutil
from datetime import datetime
from pathlib import Path


_PUBLISHED = re.compile(r"^cinelingus_[a-z0-9-]+_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}(?:_\d+)?\.mp4$")
_AUDIT_SUFFIXES = {".json", ".txt", ".csv"}


def _remove_non_deliverables(directory: Path, *, published_video: Path) -> None:
    """Remove render intermediates while retaining lightweight audit sidecars."""
    for child in list(directory.iterdir()):
        if child == published_video or (child.is_file() and directory == published_video.parent and _PUBLISHED.match(child.name)):
            continue
        if child.is_dir():
            _remove_non_deliverables(child, published_video=published_video)
            if not any(child.iterdir()):
                child.rmdir()
        elif child.suffix.lower() not in _AUDIT_SUFFIXES:
            child.unlink()


def publish_single_video(*, video: Path, output_dir: Path, process: str) -> Path:
    """Publish one video, remove heavy intermediates, and retain audit sidecars."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    slug = re.sub(r"[^a-z0-9]+", "-", process.lower()).strip("-") or "process"
    destination = output_dir / f"cinelingus_{slug}_{stamp}.mp4"
    suffix = 2
    while destination.exists():
        destination = output_dir / f"cinelingus_{slug}_{stamp}_{suffix}.mp4"
        suffix += 1
    if video.resolve() != destination.resolve():
        shutil.copy2(video, destination)
    _remove_non_deliverables(output_dir, published_video=destination)
    return destination
