from __future__ import annotations

from pathlib import Path
import wave

import pytest

from movie_masher.filter_lab.acceptance import FilterAcceptanceError, validate_filter_output
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
    assert len(catalog.contracts()) == 28
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
