from __future__ import annotations

import os
from pathlib import Path


def ensure_project_ffmpeg_shared_on_path(*, search_from: Path | None = None) -> Path | None:
    ffmpeg_bin = find_project_ffmpeg_shared_bin(search_from=search_from)
    if ffmpeg_bin is None:
        return None
    current_path = os.environ.get("PATH", "")
    parts = [part for part in current_path.split(os.pathsep) if part]
    ffmpeg_bin_text = str(ffmpeg_bin)
    if not any(_same_path(part, ffmpeg_bin_text) for part in parts):
        os.environ["PATH"] = ffmpeg_bin_text + os.pathsep + current_path
    return ffmpeg_bin


def find_project_ffmpeg_shared_bin(*, search_from: Path | None = None) -> Path | None:
    for root in _candidate_roots(search_from):
        tools_dir = root / "tools" / "ffmpeg"
        if not tools_dir.exists():
            continue
        for candidate in sorted(tools_dir.glob("*shared*/bin")):
            if _has_ffmpeg_shared_dlls(candidate):
                return candidate.resolve()
    return None


def _candidate_roots(search_from: Path | None) -> list[Path]:
    starts = []
    if search_from is not None:
        starts.append(search_from)
    starts.append(Path.cwd())

    roots: list[Path] = []
    seen: set[str] = set()
    for start in starts:
        try:
            current = start.resolve()
        except OSError:
            current = start
        if current.is_file():
            current = current.parent
        for candidate in [current, *current.parents]:
            key = str(candidate).lower()
            if key not in seen:
                roots.append(candidate)
                seen.add(key)
    return roots


def _has_ffmpeg_shared_dlls(path: Path) -> bool:
    return (path / "ffmpeg.exe").exists() and any(path.glob("avcodec-*.dll")) and any(path.glob("avformat-*.dll"))


def _same_path(left: str, right: str) -> bool:
    try:
        return Path(left).resolve() == Path(right).resolve()
    except OSError:
        return left.lower() == right.lower()
