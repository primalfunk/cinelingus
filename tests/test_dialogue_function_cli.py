from pathlib import Path

from cinelingus.cli import main
from cinelingus.util import write_json


def test_function_developer_cli_builds_validates_and_reports(tmp_path: Path, capsys) -> None:
    model = tmp_path / "film_model.json"
    output = tmp_path / "functions"
    write_json(model, {
        "film_id": "film_fixture", "created_from_signature": "signature", "dialogue_turns": [],
        "speech_passages": [{"speech_passage_id": "p1", "start": 0.0, "end": 1.0, "duration": 1.0, "original_transcript": "Where are you?", "language": "en", "provenance_id": "prov", "source_transcript_reference": "w1", "linked_performance_ids": []}],
    })

    assert main(["validate-function-taxonomy"]) == 0
    assert main(["build-function-bundle", "--model", str(model), "--output", str(output)]) == 0
    assert main(["validate-function-bundle", str(output / "dialogue_function_bundle.json"), "--model", str(model)]) == 0
    assert main(["report-function-bundle", str(output / "dialogue_function_bundle.json")]) == 0
    assert "does not infer emotion" in capsys.readouterr().out


def test_function_developer_cli_verifies_rendered_transcript(tmp_path: Path) -> None:
    classification = {
        "axes": {
            "surface_form": {"labels": [{"label": "interrogative", "confidence": 0.9}], "supported": True},
            "interaction_function": {"labels": [{"label": "request_information", "confidence": 0.9}], "supported": True},
            "sequence_position": {"labels": [{"label": "unavailable", "confidence": 1.0}], "supported": False},
        }, "confidence": 0.9, "ambiguity_state": "UNAMBIGUOUS", "abstention": {"abstained": False},
    }
    schedule = tmp_path / "schedule.json"
    rendered = tmp_path / "rendered.json"
    output = tmp_path / "function_render_verification.json"
    write_json(schedule, {"mappings": [{
        "window_id": "w1", "clip_id": "c1", "enabled": True,
        "dialogue_function_compatibility": {
            "available": True, "source_distribution": classification, "destination_distribution": classification,
            "normalized_function_contribution": 1.0, "confidence": 0.9,
        },
    }]})
    write_json(rendered, {"mappings": [{
        "mapping_index": 0, "window_id": "w1", "rendered_transcript": "Where are you?",
        "confidence": 0.9, "word_coverage_percentage": 100.0, "status": "pass",
    }]})

    assert main(["verify-rendered-function", "--schedule", str(schedule), "--rendered-verification", str(rendered), "--output", str(output)]) == 0
    assert output.is_file()
