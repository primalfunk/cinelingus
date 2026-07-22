from pathlib import Path

from cinelingus.cinematic_model.turn_coverage import audit_turn_coverage
from cinelingus.util import write_json


def test_turn_coverage_audit_reports_zero_turns_and_id_mismatch(tmp_path: Path) -> None:
    model_path = tmp_path / "film_model.json"
    write_json(model_path, {
        "film_id": "film_fixture", "media": {"filename": "fixture.mp4"},
        "speech_passages": [{"speech_passage_id": "speech_1", "linked_dialogue_turn_id": None}],
        "dialogue_turns": [],
        "performances": [{"performance_id": "performance_1", "speech_passage_references": []}],
        "provenance": [{"source_artifact_type": "timeline"}, {"source_artifact_type": "performance"}],
    })
    report = audit_turn_coverage([model_path])
    assert report["models_with_zero_turns"] == 1
    assert report["passage_assignment_percent"] == 0.0
    assert report["models"][0]["structural_id_mismatch_suspected"] is True
    assert report["models"][0]["artifact_sources_responsible"] == ["performance", "timeline"]
