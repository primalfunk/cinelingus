from pathlib import Path

import pytest

from cinelingus.cache import clear_pipeline_cache, ensure_cache
from cinelingus.util import write_json


def test_same_media_has_distinct_role_cache_directories(tmp_path: Path) -> None:
    media = tmp_path / "movie.mp4"
    media.write_bytes(b"movie")
    source = ensure_cache(tmp_path / "cache", media, "source_dialogue")
    destination = ensure_cache(tmp_path / "cache", media, "destination_video")
    assert source.cache_dir != destination.cache_dir
    assert source.cache_dir.parent == destination.cache_dir.parent


def test_cache_reuse_asserts_canonical_media_path(tmp_path: Path) -> None:
    media = tmp_path / "movie.mp4"
    other = tmp_path / "other.mp4"
    media.write_bytes(b"same")
    other.write_bytes(b"same")
    entry = ensure_cache(tmp_path / "cache", media, "source_dialogue")
    manifest = entry.manifest_path
    data = __import__("json").loads(manifest.read_text())
    data["source_path"] = str(other)
    write_json(manifest, data)
    with pytest.raises(RuntimeError, match="Cache identity mismatch"):
        ensure_cache(tmp_path / "cache", media, "source_dialogue")


def test_clear_pipeline_cache_removes_only_cache_children(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    artifact = cache / "hash" / "destination_video" / "timeline.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"artifact")
    output = tmp_path / "output" / "finished.mp4"
    output.parent.mkdir()
    output.write_bytes(b"movie")

    result = clear_pipeline_cache(cache)

    assert cache.exists()
    assert list(cache.iterdir()) == []
    assert output.read_bytes() == b"movie"
    assert result["files_removed"] == 1
    assert result["bytes_removed"] == len(b"artifact")
