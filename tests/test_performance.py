from pathlib import Path

from cinelingus.performance import build_performances, performance_windows
from cinelingus.performance_library import build_performance_library
from cinelingus.validation import validate_artifact


def test_build_performances_groups_windows_by_pause_and_classifies(tmp_path: Path) -> None:
    artifact = build_performances(
        media_hash="hash",
        role="destination_video",
        output_path=tmp_path / "performance.json",
        speaking_windows=[
            {"id": "w1", "start": 0.0, "end": 1.0, "duration": 1.0, "confidence": 0.8},
            {"id": "w2", "start": 1.5, "end": 2.5, "duration": 1.0, "confidence": 0.7},
            {"id": "w3", "start": 8.0, "end": 10.0, "duration": 2.0, "confidence": 0.9},
        ],
        shots=[
            {"id": "s1", "start": 0.0, "end": 3.0, "duration": 3.0, "confidence": 1.0},
            {"id": "s2", "start": 7.0, "end": 11.0, "duration": 4.0, "confidence": 1.0},
        ],
        max_pause=2.0,
        config_signature="sig",
    )

    assert artifact["performance_count"] == 2
    first = artifact["performances"][0]
    assert first["speaking_window_ids"] == ["w1", "w2"]
    assert first["shot_ids"] == ["s1"]
    assert first["pause_statistics"]["count"] == 1
    assert first["dialogue_density"] > 0
    assert first["speaker_sequence"] == ["A"]
    assert first["signature"]["turn_pattern"] == "A"
    assert first["signature"]["signature_version"] == "2.0"
    assert first["signature"]["average_turn_duration"] == 1.0
    assert first["signature"]["speech_continuity"] > 0
    assert first["performance_type"] in {"monologue", "dialogue_exchange"}
    assert "speaker_participation" in first["signature"]
    windows = performance_windows(artifact)
    assert windows[0]["performance_id"] == "p000001"
    assert windows[0]["performance_type_v2"] == first["performance_type"]
    assert windows[0]["estimated_energy"] == first["estimated_energy"]
    validate_artifact("performance", tmp_path / "performance.json", Path.cwd() / "schemas")



def test_build_performance_library_links_clips_and_signatures(tmp_path: Path) -> None:
    performances = {
        "config_signature": "perf_sig",
        "performances": [
            {
                "id": "p1",
                "start": 10.0,
                "end": 14.0,
                "duration": 4.0,
                "conversation_type": "exchange",
                "signature": {"performance_id": "p1", "turn_pattern": "A B"},
                "speaker_sequence": ["A", "B"],
                "turn_pattern": "A B",
            }
        ],
    }
    artifact = build_performance_library(
        media_hash="source",
        performances=performances,
        clips=[
            {"id": "c1", "path": "c1.wav", "movie_timestamp": 10.5, "duration": 1.0, "transcript": "hello"},
            {"id": "c2", "path": "c2.wav", "movie_timestamp": 20.0, "duration": 1.0, "transcript": "outside"},
        ],
        output_path=tmp_path / "performance_library.json",
        config_signature="lib_sig",
    )

    assert artifact["performance_count"] == 1
    assert artifact["performances"][0]["clip_count"] == 1
    assert artifact["performances"][0]["clips"][0]["id"] == "c1"
    assert artifact["performances"][0]["signature"]["turn_pattern"] == "A B"
    validate_artifact("performance_library", tmp_path / "performance_library.json", Path.cwd() / "schemas")
