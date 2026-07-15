from __future__ import annotations

from pathlib import Path
import wave

import pytest

from movie_masher.filter_lab.acceptance import FilterAcceptanceError, validate_filter_output, validate_schedule_quality
from movie_masher.filter_lab.contracts import default_contract_catalog
from movie_masher.filter_lab.registry import default_filter_registry
from movie_masher.filter_lab.strategies import has_strategy


def _wav(path: Path, *, active: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sample = 3000 if active else 0
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(48000)
        handle.writeframes(sample.to_bytes(2, "little", signed=True) * 48000)


def _echo_schedule(tmp_path: Path) -> dict:
    clip_path = tmp_path / "cache" / "sourcehash" / "clips" / "c1.wav"
    clip_path.parent.mkdir(parents=True)
    clip_path.write_bytes(b"clip")
    return {
        "source_media_hash": "sourcehash",
        "render_duration": 1.0,
        "mappings": [{
            "enabled": True, "clip_id": "c1", "clip_path": str(clip_path),
            "destination_timestamp": 0.0, "planned_render_duration": 1.0,
        }],
        "filter_validation": {
            "echo_delay_matches_parameter": True,
            "repeat_limit_is_respected": True,
        },
    }


def test_contract_catalog_has_exact_registry_parity_and_generated_doc() -> None:
    catalog = default_contract_catalog()
    registry = default_filter_registry()

    assert {row.filter_id for row in catalog.contracts()} == {row.id for row in registry.definitions()}
    assert len(catalog.contracts()) == 40
    assert (Path.cwd() / "docs" / "filter_contract_catalog.md").read_text(encoding="utf-8") == catalog.render_markdown() + "\n"
    for definition in registry.definitions(implemented_only=True):
        contract = catalog.get(definition.id)
        assert contract.status == "accepted"
        if definition.execution_mode == "scheduling_strategy":
            assert has_strategy(str(definition.implementation_key))


def test_output_acceptance_reports_all_required_measurements(tmp_path: Path, monkeypatch) -> None:
    video = tmp_path / "final.mp4"
    audio = tmp_path / "replacement.wav"
    video.write_bytes(b"mp4")
    _wav(audio, active=True)
    monkeypatch.setattr(
        "movie_masher.filter_lab.acceptance.ffprobe_json",
        lambda _path: {"streams": [{"codec_type": "audio", "codec_name": "aac", "sample_rate": "48000", "channels": 2}]},
    )

    report = validate_filter_output(
        filter_id="echo",
        schedule=_echo_schedule(tmp_path),
        final_video=video,
        replacement_audio=audio,
        output_path=tmp_path / "filter_acceptance.json",
        schemas_dir=Path.cwd() / "schemas",
    )

    assert report["status"] == "pass"
    assert report["measurements"]["dialogue_coverage"] == 1.0
    assert report["measurements"]["silence_ratio"] == 0.0
    assert report["checks"]["audio_stream_verified"] is True


def test_output_acceptance_rejects_mostly_silent_audio(tmp_path: Path, monkeypatch) -> None:
    video = tmp_path / "final.mp4"
    audio = tmp_path / "replacement.wav"
    video.write_bytes(b"mp4")
    _wav(audio, active=False)
    monkeypatch.setattr(
        "movie_masher.filter_lab.acceptance.ffprobe_json",
        lambda _path: {"streams": [{"codec_type": "audio", "codec_name": "aac"}]},
    )

    with pytest.raises(FilterAcceptanceError, match="replacement_audio_has_sufficient_activity"):
        validate_filter_output(
            filter_id="echo",
            schedule=_echo_schedule(tmp_path),
            final_video=video,
            replacement_audio=audio,
            output_path=tmp_path / "filter_acceptance.json",
            schemas_dir=Path.cwd() / "schemas",
        )


def test_schedule_acceptance_rejects_sparse_clustered_repetitive_placements(tmp_path: Path) -> None:
    schedule = _echo_schedule(tmp_path)
    schedule["render_duration"] = 100.0
    schedule["acceptance_requirements"] = {
        "minimum_dialogue_coverage": 0.08,
        "timeline_bucket_count": 4,
        "minimum_occupied_timeline_buckets": 3,
        "minimum_unique_source_ratio": 0.8,
        "maximum_source_reuse": 2,
    }
    schedule["mappings"] = [
        {**schedule["mappings"][0], "destination_timestamp": timestamp}
        for timestamp in (40.0, 42.0, 44.0)
    ]

    with pytest.raises(
        FilterAcceptanceError,
        match="dialogue_coverage_sufficient, timeline_distribution_sufficient, source_repetition_within_limit",
    ):
        validate_schedule_quality(schedule)


def test_schedule_acceptance_reports_distribution_and_repetition_metrics(tmp_path: Path) -> None:
    schedule = _echo_schedule(tmp_path)
    original = schedule["mappings"][0]
    schedule["render_duration"] = 100.0
    schedule["acceptance_requirements"] = {
        "minimum_dialogue_coverage": 0.08,
        "timeline_bucket_count": 4,
        "minimum_occupied_timeline_buckets": 3,
        "minimum_unique_source_ratio": 0.8,
        "maximum_source_reuse": 2,
    }
    schedule["mappings"] = [
        {**original, "clip_id": f"c{index}", "destination_timestamp": timestamp, "planned_render_duration": 3.0}
        for index, timestamp in enumerate((5.0, 30.0, 55.0), start=1)
    ]

    quality = validate_schedule_quality(schedule)

    assert quality["checks"] == {
        "dialogue_coverage_sufficient": True,
        "timeline_distribution_sufficient": True,
        "source_repetition_within_limit": True,
    }
    assert quality["measurements"]["occupied_timeline_buckets"] == 3
    assert quality["measurements"]["unique_source_ratio"] == 1.0


def test_transposition_acceptance_rejects_reused_source_dialogue(tmp_path: Path, monkeypatch) -> None:
    video = tmp_path / "final.mp4"
    audio = tmp_path / "replacement.wav"
    video.write_bytes(b"mp4")
    _wav(audio, active=True)
    monkeypatch.setattr(
        "movie_masher.filter_lab.acceptance.ffprobe_json",
        lambda _path: {"streams": [{"codec_type": "audio", "codec_name": "aac"}]},
    )
    schedule = _echo_schedule(tmp_path)
    schedule["mappings"].append({
        **schedule["mappings"][0],
        "destination_timestamp": 0.5,
        "planned_render_duration": 0.5,
    })

    with pytest.raises(FilterAcceptanceError, match="contract_invariants_pass"):
        validate_filter_output(
            filter_id="translation.movie_masher",
            schedule=schedule,
            final_video=video,
            replacement_audio=audio,
            output_path=tmp_path / "filter_acceptance.json",
            schemas_dir=Path.cwd() / "schemas",
        )

    report = (tmp_path / "filter_acceptance.json").read_text(encoding="utf-8")
    assert '"source_dialogue_is_not_reused"' in report
    assert '"passed": false' in report
