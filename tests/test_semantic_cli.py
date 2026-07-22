from pathlib import Path

from cinelingus.cli import main
from cinelingus.semantic import SEMANTIC_LIMITATION
from cinelingus.util import write_json
from cinelingus.semantic.developer_cli import _artifact_rows


def test_semantic_developer_commands_build_validate_and_report(tmp_path: Path, capsys) -> None:
    model = tmp_path / "film_model.json"
    output = tmp_path / "semantic"
    write_json(model, {
        "film_id": "film_fixture", "created_from_signature": "signature",
        "speech_passages": [{
            "speech_passage_id": "speech_fixture", "start": 0.0, "end": 1.0, "duration": 1.0,
            "original_transcript": "A small test passage.", "language": "en", "provenance_id": "provenance_fixture",
        }],
    })
    assert main(["build-semantic-bundle", "--model", str(model), "--output", str(output), "--provider", "fake"]) == 0
    assert (output / "semantic_bundle.json").is_file()
    assert SEMANTIC_LIMITATION in (output / "semantic_report.txt").read_text(encoding="utf-8")
    capsys.readouterr()
    assert main(["validate-semantic-bundle", str(output / "semantic_bundle.json"), "--model", str(model)]) == 0
    assert "VALID" in capsys.readouterr().out
    assert main(["report-semantic-bundle", str(output / "semantic_bundle.json")]) == 0
    assert "transcript-vector similarity only" in capsys.readouterr().out


def test_semantic_review_developer_command_requires_and_blinds_completed_pairs(tmp_path: Path) -> None:
    control, semantic = tmp_path / "control.mp4", tmp_path / "semantic.mp4"
    control.write_bytes(b"control")
    semantic.write_bytes(b"semantic")
    cases = tmp_path / "cases.json"
    output = tmp_path / "review"
    write_json(cases, [{"case_id": "proof", "control_media": str(control), "semantic_media": str(semantic)}])

    assert main(["prepare-semantic-review", "--cases", str(cases), "--output", str(output), "--seed", "fixed"]) == 0
    assert (output / "review_manifest.json").is_file()
    assert (output / "answer_key.json").is_file()

    responses = __import__("json").loads((output / "review_responses.json").read_text(encoding="utf-8"))
    responses["cases"][0]["answers"] = {
        "semantic_relatedness": "A", "performance_fit": "B",
        "intelligibility_and_completeness": "NO_PREFERENCE", "overall_preference": "A",
    }
    write_json(output / "review_responses.json", responses)
    assert main(["finalize-semantic-review", "--package", str(output)]) == 0
    assert (output / "semantic_review_result.json").is_file()


def test_semantic_screen_accepts_source_or_destination_speech_row_namespaces() -> None:
    assert _artifact_rows({"windows": [{"id": "w1"}]}, "windows", "events") == [{"id": "w1"}]
    assert _artifact_rows({"events": [{"id": "e1"}]}, "windows", "events") == [{"id": "e1"}]
