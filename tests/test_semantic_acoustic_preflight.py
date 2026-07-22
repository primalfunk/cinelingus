import wave
from pathlib import Path

import pytest

from cinelingus.semantic.acoustic_preflight import (
    require_accepted_semantic_preflight,
    run_semantic_acoustic_preflight,
)
from cinelingus.util import write_json
from cinelingus.validation import validate_artifact


def _wav(path: Path, seconds: float = 2.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\0\0" * int(16000 * seconds))


def _screen(root: Path, clip: Path, transcript: str) -> None:
    write_json(root / "semantic_schedule_screen.json", {
        "experiment_signature": "screen-signature",
        "render_selection": ["control", "report_only", "assisted_005"],
    })
    base = {
        "editorial_placement_id": "placement-1", "window_id": "w1",
        "destination_timestamp": 0.0, "source_performance_id": "p1",
        "clip_trim_start": 0.0, "clip_trim_duration": 2.0,
        "source_transcript": transcript, "enabled": True,
    }
    write_json(root / "control_schedule.json", {"mappings": [{**base, "clip_id": "control", "clip_path": str(clip)}]})
    write_json(root / "assisted_005_schedule.json", {"mappings": [{**base, "clip_id": "semantic", "clip_path": str(clip)}]})


def test_changed_donor_preflight_accepts_complete_retranscription(tmp_path: Path) -> None:
    clip = tmp_path / "clip.wav"
    _wav(clip)
    screen = tmp_path / "screen"
    _screen(screen, clip, "hello brave world")

    def transcriber(**kwargs):
        return {"media_hash": kwargs["media_hash"], "windows": [{
            "id": "w1", "start": 1.0, "end": 3.0,
            "transcript": "hello brave world", "confidence": 0.95,
        }]}

    output = tmp_path / "proof"
    report = run_semantic_acoustic_preflight(
        screen_dir=screen, output_dir=output, semantic_variant="assisted_005",
        transcriber=transcriber,
    )

    assert report["preflight_state"] == "ACCEPTED_FOR_RENDER"
    assert report["mapping_decisions"][0]["state"] == "ACCEPTED"
    assert require_accepted_semantic_preflight(
        output / "semantic_acoustic_preflight.json",
        screen_signature="screen-signature", semantic_variant="assisted_005",
    )
    assert validate_artifact(
        "semantic_acoustic_preflight", output / "semantic_acoustic_preflight.json",
        Path(__file__).parents[1] / "schemas",
    )


def test_changed_donor_preflight_rejects_truncated_phrase(tmp_path: Path) -> None:
    clip = tmp_path / "clip.wav"
    _wav(clip)
    screen = tmp_path / "screen"
    _screen(screen, clip, "hello brave world")

    def transcriber(**kwargs):
        return {"media_hash": kwargs["media_hash"], "windows": [{
            "id": "w1", "start": 1.0, "end": 3.0,
            "transcript": "hello brave", "confidence": 0.95,
        }]}

    output = tmp_path / "proof"
    report = run_semantic_acoustic_preflight(
        screen_dir=screen, output_dir=output, semantic_variant="assisted_005",
        transcriber=transcriber,
    )

    assert report["preflight_state"] == "REJECTED_ACOUSTIC_INTEGRITY"
    assert report["rejected_source_performance_ids"] == ["p1"]
    assert report["repair_lineage"] == {}
    assert set(report["mapping_decisions"][0]["reasons"]) == {
        "word_coverage_below_threshold", "sentence_ending_missing",
    }
    with pytest.raises(ValueError, match="preflight_not_accepted"):
        require_accepted_semantic_preflight(
            output / "semantic_acoustic_preflight.json",
            screen_signature="screen-signature", semantic_variant="assisted_005",
        )
