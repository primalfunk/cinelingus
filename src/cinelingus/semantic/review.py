from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Any

from ..util import read_json, stable_hash, utc_now, write_json

REVIEW_RESPONSES = ("A", "B", "NO_PREFERENCE", "BOTH_POOR", "UNREVIEWABLE")
REVIEW_QUESTIONS = (
    ("semantic_relatedness", "Which donor is more meaningfully related to the destination transcript?"),
    ("performance_fit", "Which donor better fits the performance?"),
    ("intelligibility_and_completeness", "Which version is more intelligible and complete?"),
    ("overall_preference", "Which version is preferred overall?"),
)


def build_blinded_semantic_review_package(
    cases: list[dict[str, Any]], output_dir: Path, *, seed: str = "phase2-semantic-review-v1",
) -> dict[str, Any]:
    """Copy completed render pairs behind deterministic A/B labels and emit a separate key."""
    if not cases:
        raise ValueError("At least one completed render pair is required")
    media_dir = output_dir / "media"
    media_dir.mkdir(parents=True, exist_ok=True)
    review_cases, key_cases, response_cases = [], [], []
    for source in sorted(cases, key=lambda row: str(row.get("case_id"))):
        case_id = str(source.get("case_id") or "")
        if not case_id:
            raise ValueError("Every review case requires a case_id")
        control, semantic = Path(source["control_media"]), Path(source["semantic_media"])
        if not control.is_file() or not semantic.is_file():
            raise FileNotFoundError(f"Completed control and semantic renders are required for {case_id}")
        semantic_is_a = int(stable_hash({"seed": seed, "case_id": case_id})[:8], 16) % 2 == 0
        assignments = {"A": semantic if semantic_is_a else control, "B": control if semantic_is_a else semantic}
        conditions = {"A": "SEMANTIC" if semantic_is_a else "CONTROL", "B": "CONTROL" if semantic_is_a else "SEMANTIC"}
        options = {}
        for label in ("A", "B"):
            suffix = assignments[label].suffix.lower() or ".bin"
            destination = media_dir / f"{case_id}_{label}{suffix}"
            shutil.copy2(assignments[label], destination)
            options[label] = {
                "media": destination.relative_to(output_dir).as_posix(),
                "sha256": _file_digest(destination),
            }
        review_cases.append({
            "case_id": case_id, "options": options,
            "destination_context": source.get("destination_context"),
            "questions": [{"question_id": identifier, "prompt": prompt, "allowed_responses": list(REVIEW_RESPONSES)} for identifier, prompt in REVIEW_QUESTIONS],
        })
        key_cases.append({"case_id": case_id, "conditions": conditions})
        response_cases.append({"case_id": case_id, "answers": {identifier: None for identifier, _ in REVIEW_QUESTIONS}, "notes": ""})
    package_signature = stable_hash({"seed": seed, "cases": review_cases})
    manifest = {
        "schema_version": "1.0", "review_version": "semantic_blinded_review_v1",
        "creation_timestamp": utc_now(), "package_signature": package_signature,
        "blinding_state": "BLINDED", "case_count": len(review_cases),
        "cases": review_cases,
        "instructions": "Review A and B without consulting answer_key.json. Semantic relatedness and overall preference are separate judgments.",
        "claim_scope": "Human judgments are evaluation evidence only and are not used for model training or fine-tuning.",
    }
    answer_key = {
        "schema_version": "1.0", "review_version": "semantic_blinded_review_v1",
        "package_signature": package_signature, "seed_signature": stable_hash(seed), "cases": key_cases,
    }
    responses = {
        "schema_version": "1.0", "review_version": "semantic_blinded_review_v1",
        "package_signature": package_signature, "reviewer_id": None, "cases": response_cases,
    }
    write_json(output_dir / "review_manifest.json", manifest)
    write_json(output_dir / "answer_key.json", answer_key)
    write_json(output_dir / "review_responses.json", responses)
    return manifest


def finalize_blinded_semantic_review(
    package_dir: Path, *, responses_path: Path | None = None, output_path: Path | None = None,
) -> dict[str, Any]:
    """Validate completed blinded answers, then unblind them into an auditable result."""
    manifest = read_json(package_dir / "review_manifest.json")
    answer_key = read_json(package_dir / "answer_key.json")
    responses = read_json(responses_path or package_dir / "review_responses.json")
    signatures = {
        manifest.get("package_signature"), answer_key.get("package_signature"), responses.get("package_signature"),
    }
    if len(signatures) != 1 or None in signatures:
        raise ValueError("Review manifest, answer key, and responses do not describe the same package")

    manifest_cases = {row["case_id"]: row for row in manifest.get("cases", [])}
    key_cases = {row["case_id"]: row for row in answer_key.get("cases", [])}
    response_cases = {row["case_id"]: row for row in responses.get("cases", [])}
    if not manifest_cases or set(manifest_cases) != set(key_cases) or set(manifest_cases) != set(response_cases):
        raise ValueError("Review case identities are missing or inconsistent")

    expected_questions = [identifier for identifier, _ in REVIEW_QUESTIONS]
    finalized_cases: list[dict[str, Any]] = []
    incomplete: list[str] = []
    for case_id in sorted(manifest_cases):
        conditions = key_cases[case_id].get("conditions") or {}
        if set(conditions) != {"A", "B"} or set(conditions.values()) != {"CONTROL", "SEMANTIC"}:
            raise ValueError(f"Invalid answer-key conditions for {case_id}")
        answers = response_cases[case_id].get("answers") or {}
        if set(answers) != set(expected_questions):
            raise ValueError(f"Responses for {case_id} must answer exactly the declared questions")
        judgments: dict[str, dict[str, Any]] = {}
        for question_id in expected_questions:
            blinded = answers[question_id]
            if blinded is None:
                incomplete.append(f"{case_id}:{question_id}")
                unblinded = None
            elif blinded not in REVIEW_RESPONSES:
                raise ValueError(f"Invalid response {blinded!r} for {case_id}:{question_id}")
            else:
                unblinded = conditions.get(blinded, blinded)
            judgments[question_id] = {"blinded_response": blinded, "condition": unblinded}
        finalized_cases.append({
            "case_id": case_id,
            "judgments": judgments,
            "notes": str(response_cases[case_id].get("notes") or ""),
        })

    state = "COMPLETE" if not incomplete else "INCOMPLETE"
    result = {
        "schema_version": "1.0",
        "review_version": "semantic_blinded_review_v1",
        "finalizer_version": "semantic_review_finalizer_v1",
        "creation_timestamp": utc_now(),
        "package_signature": manifest["package_signature"],
        "reviewer_id": responses.get("reviewer_id"),
        "review_state": state,
        "phase2_human_review_criterion": "PASS" if state == "COMPLETE" else "PENDING",
        "distinguishes_semantic_relatedness_from_overall_preference": state == "COMPLETE",
        "incomplete_response_ids": incomplete,
        "cases": finalized_cases,
        "claim_scope": "Separate blinded human judgments; no training or fine-tuning use is authorized.",
    }
    write_json(output_path or package_dir / "semantic_review_result.json", result)
    return result


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
