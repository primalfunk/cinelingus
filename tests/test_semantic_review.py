from pathlib import Path

from cinelingus.semantic.review import (
    REVIEW_RESPONSES, build_blinded_semantic_review_package, finalize_blinded_semantic_review,
)
from cinelingus.util import read_json
from cinelingus.validation import validate_artifact


def test_blinded_review_package_separates_manifest_from_answer_key(tmp_path: Path) -> None:
    control, semantic = tmp_path / "control.mp4", tmp_path / "semantic.mp4"
    control.write_bytes(b"control-render")
    semantic.write_bytes(b"semantic-render")
    output = tmp_path / "review"

    manifest = build_blinded_semantic_review_package([{
        "case_id": "case_001", "control_media": control, "semantic_media": semantic,
        "destination_context": "A short destination passage.",
    }], output, seed="fixed")

    assert manifest["blinding_state"] == "BLINDED"
    assert "CONTROL" not in (output / "review_manifest.json").read_text(encoding="utf-8")
    assert "SEMANTIC" not in (output / "review_manifest.json").read_text(encoding="utf-8")
    key = read_json(output / "answer_key.json")
    assert set(key["cases"][0]["conditions"].values()) == {"CONTROL", "SEMANTIC"}
    assert len(list((output / "media").glob("*.mp4"))) == 2
    assert manifest["cases"][0]["questions"][0]["allowed_responses"] == list(REVIEW_RESPONSES)
    validate_artifact("semantic_review_manifest", output / "review_manifest.json", Path("schemas"))


def test_blinded_review_requires_completed_render_pair(tmp_path: Path) -> None:
    import pytest

    with pytest.raises(FileNotFoundError, match="Completed control and semantic renders"):
        build_blinded_semantic_review_package([{
            "case_id": "missing", "control_media": tmp_path / "a.mp4", "semantic_media": tmp_path / "b.mp4",
        }], tmp_path / "review")


def test_finalize_blinded_review_validates_and_unblinds_separate_judgments(tmp_path: Path) -> None:
    control, semantic = tmp_path / "control.mp4", tmp_path / "semantic.mp4"
    control.write_bytes(b"control-render")
    semantic.write_bytes(b"semantic-render")
    output = tmp_path / "review"
    build_blinded_semantic_review_package([{
        "case_id": "case_001", "control_media": control, "semantic_media": semantic,
    }], output, seed="fixed")
    responses = read_json(output / "review_responses.json")
    responses["reviewer_id"] = "reviewer"
    responses["cases"][0]["answers"] = {
        "semantic_relatedness": "A", "performance_fit": "B",
        "intelligibility_and_completeness": "NO_PREFERENCE", "overall_preference": "A",
    }
    from cinelingus.util import write_json
    write_json(output / "review_responses.json", responses)

    result = finalize_blinded_semantic_review(output)

    assert result["review_state"] == "COMPLETE"
    assert result["phase2_human_review_criterion"] == "PASS"
    assert result["distinguishes_semantic_relatedness_from_overall_preference"] is True
    assert result["cases"][0]["judgments"]["semantic_relatedness"]["condition"] in {"CONTROL", "SEMANTIC"}
    assert result["cases"][0]["judgments"]["performance_fit"]["condition"] in {"CONTROL", "SEMANTIC"}
    validate_artifact("semantic_review_result", output / "semantic_review_result.json", Path("schemas"))


def test_finalize_blinded_review_keeps_incomplete_review_pending(tmp_path: Path) -> None:
    control, semantic = tmp_path / "control.mp4", tmp_path / "semantic.mp4"
    control.write_bytes(b"control-render")
    semantic.write_bytes(b"semantic-render")
    output = tmp_path / "review"
    build_blinded_semantic_review_package([{
        "case_id": "case_001", "control_media": control, "semantic_media": semantic,
    }], output)

    result = finalize_blinded_semantic_review(output)

    assert result["review_state"] == "INCOMPLETE"
    assert len(result["incomplete_response_ids"]) == 4
