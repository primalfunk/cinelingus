from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from ..util import read_json


def audit_turn_coverage(model_paths: Iterable[Path]) -> dict[str, Any]:
    rows = [_audit_model(path) for path in sorted(model_paths, key=lambda item: item.as_posix().casefold())]
    passage_count = sum(row["speech_passage_count"] for row in rows)
    assigned_count = sum(row["passages_assigned_to_turns"] for row in rows)
    return {
        "schema_version": "1.0",
        "audit_version": "film_model_turn_coverage_v1",
        "model_count": len(rows),
        "speech_passage_count": passage_count,
        "dialogue_turn_count": sum(row["dialogue_turn_count"] for row in rows),
        "passages_assigned_to_turns": assigned_count,
        "passage_assignment_percent": round(100.0 * assigned_count / passage_count, 4) if passage_count else 0.0,
        "turns_with_deterministic_passage_order": sum(row["turns_with_deterministic_passage_order"] for row in rows),
        "performances_with_ordered_turns": sum(row["performances_with_ordered_turns"] for row in rows),
        "models_with_zero_turns": sum(row["dialogue_turn_count"] == 0 for row in rows),
        "models": rows,
    }


def _audit_model(path: Path) -> dict[str, Any]:
    model = read_json(path)
    passages = model.get("speech_passages") or []
    turns = model.get("dialogue_turns") or []
    performances = model.get("performances") or []
    passage_ids = {row.get("speech_passage_id") for row in passages}
    assigned = {row.get("speech_passage_id") for row in passages if row.get("linked_dialogue_turn_id")}
    performance_passage_refs = {
        ref for performance in performances for ref in performance.get("speech_passage_references") or [] if ref in passage_ids
    }
    deterministic_turns = sum(
        bool(row.get("ordered_speech_passage_references"))
        and len(row["ordered_speech_passage_references"]) == len(set(row["ordered_speech_passage_references"]))
        for row in turns
    )
    sources = sorted({
        str(row.get("source_artifact_type"))
        for row in model.get("provenance") or []
        if row.get("source_artifact_type") in {"dialogue_events", "timeline", "performance"}
    })
    mismatch = bool(performances and not performance_passage_refs and passages)
    limitations: list[str] = []
    if not turns:
        limitations.append("No ordered_turns evidence was present in the performance artifact.")
    if mismatch:
        limitations.append("Performance window IDs do not map to the canonical speech-passage source IDs.")
    elif passages and len(performance_passage_refs) < len(passages):
        limitations.append("Only part of the passage set is referenced by performances.")
    return {
        "model_path": path.resolve().as_posix(),
        "film_id": model.get("film_id"),
        "filename": (model.get("media") or {}).get("filename"),
        "speech_passage_count": len(passages),
        "dialogue_turn_count": len(turns),
        "passages_assigned_to_turns": len(assigned),
        "passage_assignment_percent": round(100.0 * len(assigned) / len(passages), 4) if passages else 0.0,
        "turns_with_deterministic_passage_order": deterministic_turns,
        "performance_count": len(performances),
        "performances_with_ordered_turns": sum(bool(row.get("dialogue_turn_references")) for row in performances),
        "passages_referenced_by_performances": len(performance_passage_refs),
        "performance_passage_coverage_percent": round(100.0 * len(performance_passage_refs) / len(passages), 4) if passages else 0.0,
        "artifact_sources_responsible": sources,
        "structural_id_mismatch_suspected": mismatch,
        "missing_structural_evidence": limitations,
    }
