from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from cinelingus.evaluation_corpus import _excerpt_candidates, build_corpus_manifest, build_evaluation_plan, build_excerpt_plan
from cinelingus.util import read_json, write_json
from cinelingus.validation import validate_artifact


def _probe(duration: float = 1200.0) -> dict:
    return {
        "format": {"duration": str(duration)},
        "streams": [
            {"codec_type": "video", "codec_name": "h264", "width": 640, "height": 480, "avg_frame_rate": "24/1"},
            {"codec_type": "audio", "codec_name": "aac", "sample_rate": "48000", "channels": 2, "channel_layout": "stereo"},
        ],
    }


def test_inventory_is_read_only_and_records_unsupported_files(tmp_path: Path) -> None:
    source = tmp_path / "movies"
    source.mkdir()
    (source / "Robocop Cartoon.mp4").write_bytes(b"video-a")
    (source / "Drama.mp4").write_bytes(b"video-b")
    (source / "audio.mp3").write_bytes(b"audio")
    before = sorted(path.name for path in source.iterdir())
    output = tmp_path / "evaluation" / "corpus_manifest.json"
    cache = tmp_path / "evaluation" / "inventory_cache.json"

    manifest = build_corpus_manifest(
        source_root=source, output_path=output, inventory_cache_path=cache,
        probe_media=lambda _path: _probe(), hash_media=lambda path: ("a" if "Cartoon" in path.name else "b") * 64,
    )

    assert manifest["compatible_file_count"] == 2
    assert manifest["excluded_file_count"] == 1
    assert manifest["media"][1]["content_type"] == "animation"
    assert manifest["exclusions"][0]["reason"] == "unsupported_non_video_media"
    assert sorted(path.name for path in source.iterdir()) == before
    validate_artifact("corpus_manifest", output, Path.cwd() / "schemas")


def test_inventory_reuses_hash_and_probe_cache(tmp_path: Path) -> None:
    source = tmp_path / "movies"; source.mkdir()
    (source / "one.mp4").write_bytes(b"video")
    output = tmp_path / "manifest.json"; cache = tmp_path / "inventory.json"
    calls = {"hash": 0, "probe": 0}

    def hasher(_path): calls["hash"] += 1; return "c" * 64
    def probe(_path): calls["probe"] += 1; return _probe()

    build_corpus_manifest(source_root=source, output_path=output, inventory_cache_path=cache, probe_media=probe, hash_media=hasher)
    second = build_corpus_manifest(source_root=source, output_path=output, inventory_cache_path=cache, probe_media=probe, hash_media=hasher)

    assert calls == {"hash": 1, "probe": 1}
    assert second["media"][0]["inventory_source"] == "inventory_cache"


def test_inventory_refuses_to_write_inside_source_folder(tmp_path: Path) -> None:
    source = tmp_path / "movies"; source.mkdir()
    with pytest.raises(ValueError, match="must not be written"):
        build_corpus_manifest(
            source_root=source, output_path=source / "manifest.json",
            inventory_cache_path=tmp_path / "cache.json",
        )


def test_tier_plan_is_deterministic_bounded_and_purposeful(tmp_path: Path) -> None:
    source = tmp_path / "movies"; source.mkdir()
    names = ["Cartoon A.mp4", "Cartoon B.mp4", "Drama A.mp4", "Drama B.mp4", "Long Drama.mp4"]
    for name in names: (source / name).write_bytes(name.encode())
    manifest_path = tmp_path / "manifest.json"
    build_corpus_manifest(
        source_root=source, output_path=manifest_path, inventory_cache_path=tmp_path / "cache.json",
        probe_media=lambda path: _probe(5000.0 if "Long" in path.name else 1200.0),
        hash_media=lambda path: (path.name.encode().hex() + "0" * 64)[:64],
    )

    first = build_evaluation_plan(
        manifest_path=manifest_path, output_path=tmp_path / "plan1.json", tier="smoke", seed=7,
        max_files=4, max_pairings=3,
    )
    second = build_evaluation_plan(
        manifest_path=manifest_path, output_path=tmp_path / "plan2.json", tier="smoke", seed=7,
        max_files=4, max_pairings=3,
    )

    assert [row["media_id"] for row in first["selected_media"]] == [row["media_id"] for row in second["selected_media"]]
    assert first["selected_file_count"] <= 4
    assert first["selected_pairing_count"] <= 3
    assert all(row["purpose"] for row in first["pairings"])
    usage = Counter(
        media_id
        for row in first["pairings"]
        for media_id in (row["source_media_id"], row["destination_media_id"])
    )
    assert max(usage.values()) <= 2
    validate_artifact("corpus_evaluation_plan", tmp_path / "plan1.json", Path.cwd() / "schemas")


def test_excerpt_plan_uses_analysis_evidence_and_exact_bounded_regions(tmp_path: Path) -> None:
    digest = "d" * 64
    manifest_path = tmp_path / "manifest.json"
    write_json(manifest_path, {
        "manifest_version": "movie_corpus_manifest_v1",
        "media": [{
            "media_id": "media-dddd", "media_hash": digest, "filename": "Cartoon.mp4",
            "source_path": str(tmp_path / "source" / "Cartoon.mp4"),
            "content_type": "animation", "duration_class": "episode", "duration_seconds": 120.0,
            "analysis": {"available": True, "analysis_cache_signature": "sig-1"},
        }],
    })
    role = tmp_path / "cache" / digest / "destination_video"
    role.mkdir(parents=True)
    write_json(role / "filtered_timeline.json", {"windows": [
        {"id": "w1", "start": 4.0, "end": 5.0, "duration": 1.0, "confidence": 0.9},
        {"id": "w2", "start": 20.0, "end": 24.0, "duration": 4.0, "confidence": 0.9},
        {"id": "w3", "start": 44.0, "end": 48.0, "duration": 4.0, "confidence": 0.9},
        {"id": "w4", "start": 110.0, "end": 114.0, "duration": 4.0, "confidence": 0.9},
    ]})
    write_json(role / "performance.json", {"performances": [
        {"id": "p1", "start": 2.0, "end": 32.0, "duration": 30.0, "dialogue_density": 0.9, "estimated_turn_count": 8, "estimated_speaker_count": 2},
        {"id": "p2", "start": 60.0, "end": 100.0, "duration": 40.0, "dialogue_density": 0.4, "estimated_turn_count": 1, "estimated_speaker_count": 1},
        {"id": "p3", "start": 50.0, "end": 56.0, "duration": 6.0, "dialogue_density": 0.8, "estimated_turn_count": 4, "estimated_speaker_count": 2},
    ]})
    write_json(role / "shots.json", {"transitions": [
        {"id": "t1", "kind": "CUT", "start": 109.5, "end": 110.0},
    ]})

    first = build_excerpt_plan(
        manifest_path=manifest_path, cache_root=tmp_path / "cache", output_path=tmp_path / "first.json",
        tier="standard", seed=9, max_excerpts=7, max_total_duration=180.0,
    )
    second = build_excerpt_plan(
        manifest_path=manifest_path, cache_root=tmp_path / "cache", output_path=tmp_path / "second.json",
        tier="standard", seed=9, max_excerpts=7, max_total_duration=180.0,
    )

    assert [(row["media_id"], row["start"], row["end"], row["category"]) for row in first["excerpts"]] == [
        (row["media_id"], row["start"], row["end"], row["category"]) for row in second["excerpts"]
    ]
    assert {row["category"] for row in first["excerpts"]} >= {
        "animation_dialogue", "long_monologue", "quiet_room_tone",
    }
    assert all(2.0 <= row["duration"] <= 30.0 for row in first["excerpts"])
    assert all(row["analysis_signature"] == "sig-1" for row in first["excerpts"])
    rapid = next(row for row in _excerpt_candidates(read_json(manifest_path)["media"][0], tmp_path / "cache") if row["category"] == "rapid_speaker_exchange")
    assert rapid["evidence"]["turn_count"] >= 2
    assert rapid["evidence"]["speaker_count"] >= 2
    quiet = next(row for row in _excerpt_candidates(read_json(manifest_path)["media"][0], tmp_path / "cache") if row["category"] == "quiet_room_tone")
    assert quiet["start"] < quiet["evidence"]["dialogue_resume"] < quiet["end"]
    assert any(row["category"] == "transition_near_dialogue" for row in _excerpt_candidates(read_json(manifest_path)["media"][0], tmp_path / "cache"))
    validate_artifact("corpus_excerpt_plan", tmp_path / "first.json", Path.cwd() / "schemas")
