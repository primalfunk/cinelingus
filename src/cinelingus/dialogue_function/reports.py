from __future__ import annotations

from pathlib import Path
from typing import Any


def render_function_bundle_report(bundle: dict[str, Any]) -> str:
    coverage = bundle.get("coverage") or {}
    counts = coverage.get("status_counts") or {}
    classifier = bundle.get("classifier") or {}
    lines = [
        "CINELINGUS DIALOGUE-FUNCTION REPORT",
        "====================================",
        "",
        f"Film ID: {bundle.get('film_id')}",
        f"Construction state: {bundle.get('construction_state')}",
        f"Taxonomy: {bundle.get('taxonomy_version')}",
        f"Classifier: {classifier.get('classifier_version')} ({classifier.get('classifier_type')})",
        f"Configuration signature: {bundle.get('configuration_signature')}",
        f"Language scope: {classifier.get('language_scope')}",
        "",
        "COVERAGE",
        "--------",
        f"Source passages: {coverage.get('source_entity_count', 0)}",
        f"Accounted passages: {coverage.get('accounted_entity_count', 0)}",
        *[f"{state.lower()}: {counts.get(state, 0)}" for state in ("CLASSIFIED", "ABSTAINED", "UNAVAILABLE", "FAILED")],
        f"Sequence position available: {coverage.get('sequence_position_available_count', 0)}",
        f"Dialogue turns: {coverage.get('source_turn_count', 0)}",
        f"Turn aggregates available: {coverage.get('turn_aggregate_available_count', 0)}",
        f"Turn aggregates unavailable: {coverage.get('turn_aggregate_unavailable_count', 0)}",
        f"Ordered function sequences available: {coverage.get('function_sequence_available_count', 0)}",
        f"Ordered function sequences unavailable: {coverage.get('function_sequence_unavailable_count', 0)}",
        "",
        "LIMITATION",
        "----------",
        "This subsystem classifies observable conversational form and function from transcript and declared bounded context. It does not infer emotion, character, relationship, scene meaning, genre, irony, comedy, or narrative purpose.",
    ]
    return "\n".join(lines) + "\n"


def write_function_bundle_report(path: Path, bundle: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_function_bundle_report(bundle), encoding="utf-8")
