import wave
from pathlib import Path

from cinelingus.dialogue_function.acoustic_preflight import run_function_acoustic_preflight
from cinelingus.util import write_json
from cinelingus.validation import validate_artifact


def test_function_preflight_checks_only_changed_donor(tmp_path: Path) -> None:
    clip = tmp_path / "clip.wav"
    with wave.open(str(clip), "wb") as target:
        target.setnchannels(1); target.setsampwidth(2); target.setframerate(16000); target.writeframes(b"\0\0" * 32000)
    screen = tmp_path / "screen"; screen.mkdir()
    write_json(screen / "function_schedule_screen.json", {"experiment_signature": "screen", "calibration_state": "PENDING_HUMAN_ANNOTATION"})
    base = {"mappings": [{"window_id": "w1", "clip_id": "old", "source_performance_id": "old", "destination_timestamp": 0.0}]}
    changed = {"mappings": [{"window_id": "w1", "clip_id": "new", "source_performance_id": "new", "clip_path": str(clip), "source_transcript": "Where are you?", "clip_trim_duration": 2.0, "destination_timestamp": 0.0, "planned_render_duration": 2.0}]}
    write_json(screen / "function_report_only_schedule.json", base)
    write_json(screen / "function_preserving_schedule.json", changed)
    report = run_function_acoustic_preflight(
        screen_dir=screen, output_dir=tmp_path / "output",
        transcriber=lambda **kwargs: {"media_hash": kwargs["media_hash"], "windows": [{"start": 1.0, "end": 3.0, "transcript": "Where are you?", "confidence": 0.9}]},
    )
    assert report["preflight_state"] == "ACCEPTED_FOR_RENDER"
    assert report["changed_mapping_count"] == 1
    validate_artifact("function_acoustic_preflight", tmp_path / "output/function_acoustic_preflight.json", Path("schemas"))
