from __future__ import annotations

from pathlib import Path

from cinelingus.ffmpeg_env import ensure_project_ffmpeg_shared_on_path, find_project_ffmpeg_shared_bin


def test_find_project_ffmpeg_shared_bin_prefers_shared_build(tmp_path: Path) -> None:
    ffmpeg_bin = tmp_path / "tools" / "ffmpeg" / "ffmpeg-8.1.2-full_build-shared" / "bin"
    ffmpeg_bin.mkdir(parents=True)
    (ffmpeg_bin / "ffmpeg.exe").write_text("", encoding="utf-8")
    (ffmpeg_bin / "avcodec-62.dll").write_text("", encoding="utf-8")
    (ffmpeg_bin / "avformat-62.dll").write_text("", encoding="utf-8")

    assert find_project_ffmpeg_shared_bin(search_from=tmp_path / "cache" / "abc" / "speaker_map.json") == ffmpeg_bin.resolve()


def test_ensure_project_ffmpeg_shared_on_path_prepends_once(tmp_path: Path, monkeypatch) -> None:
    ffmpeg_bin = tmp_path / "tools" / "ffmpeg" / "ffmpeg-8.1.2-full_build-shared" / "bin"
    ffmpeg_bin.mkdir(parents=True)
    (ffmpeg_bin / "ffmpeg.exe").write_text("", encoding="utf-8")
    (ffmpeg_bin / "avcodec-62.dll").write_text("", encoding="utf-8")
    (ffmpeg_bin / "avformat-62.dll").write_text("", encoding="utf-8")
    monkeypatch.setenv("PATH", "C:\\Other")

    resolved = ensure_project_ffmpeg_shared_on_path(search_from=tmp_path)
    ensure_project_ffmpeg_shared_on_path(search_from=tmp_path)

    parts = [part for part in __import__("os").environ["PATH"].split(__import__("os").pathsep) if part]
    assert resolved == ffmpeg_bin.resolve()
    assert parts[0] == str(ffmpeg_bin.resolve())
    assert parts.count(str(ffmpeg_bin.resolve())) == 1
