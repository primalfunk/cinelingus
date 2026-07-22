from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from ..util import read_json, stable_hash, utc_now, write_json
from ..validation import validate_artifact


def aggregate_semantic_schedule_screens(
    cases: list[dict[str, Any]], *, output_path: Path, schemas_dir: Path | None = None,
) -> dict[str, Any]:
    """Aggregate schedule screens without treating missing semantic evidence as restraint."""
    if not cases:
        raise ValueError("At least one semantic schedule screen is required")
    rows = []
    for index, case in enumerate(cases, start=1):
        path = Path(case["screen"])
        screen = validate_artifact("semantic_schedule_screen", path, schemas_dir) if schemas_dir else read_json(path)
        rows.append(_case_row(case, screen, index=index))

    counts = Counter(row["case_state"] for row in rows)
    coverage_counts = Counter(row["semantic_coverage_state"] for row in rows)
    total_mappings = sum(row["mapping_count"] for row in rows)
    total_covered = sum(row["semantic_placement_count"] for row in rows)
    total_aggregate_fallback = sum(row["performance_aggregate_fallback_count"] for row in rows)
    total_direct = total_covered - total_aggregate_fallback
    total_boundary_bridge = sum(row["boundary_bridge_semantic_placement_count"] for row in rows)
    total_text_bridge = sum(row["text_bridge_semantic_placement_count"] for row in rows)
    total_exact_direct = sum(row["exact_direct_semantic_placement_count"] for row in rows)
    invariant_failures = sum(1 for row in rows if not row["invariants_passed"])
    nominees = sum(1 for row in rows if row["render_nominee"] is not None)
    if invariant_failures:
        corpus_state = "INVALID_INVARIANT_FAILURE"
    elif coverage_counts["NONE"] or coverage_counts["PARTIAL"]:
        corpus_state = "INCOMPLETE_SEMANTIC_COVERAGE"
    elif nominees:
        corpus_state = "RENDER_NOMINEES_AVAILABLE"
    else:
        corpus_state = "VALID_RESTRAINT_ONLY"
    report = {
        "schema_version": "1.0",
        "experiment_version": "semantic_corpus_screen_v1",
        "creation_timestamp": utc_now(),
        "corpus_signature": stable_hash([{key: value for key, value in row.items() if key != "screen"} for row in rows]),
        "case_count": len(rows),
        "corpus_state": corpus_state,
        "summary": {
            "invariant_failure_count": invariant_failures,
            "render_nominee_count": nominees,
            "changed_case_count": sum(1 for row in rows if row["changed_variant_count"]),
            "conflicted_case_count": sum(1 for row in rows if row["conflicted_variant_count"]),
            "coverage_state_counts": dict(sorted(coverage_counts.items())),
            "case_state_counts": dict(sorted(counts.items())),
            "mapping_count": total_mappings,
            "semantic_placement_count": total_covered,
            "direct_semantic_placement_count": total_direct,
            "performance_aggregate_fallback_count": total_aggregate_fallback,
            "exact_direct_semantic_placement_count": total_exact_direct,
            "boundary_bridge_semantic_placement_count": total_boundary_bridge,
            "text_bridge_semantic_placement_count": total_text_bridge,
            "weighted_semantic_coverage": round(total_covered / total_mappings, 6) if total_mappings else 0.0,
            "weighted_direct_semantic_coverage": round(total_direct / total_mappings, 6) if total_mappings else 0.0,
        },
        "cases": rows,
        "claim_scope": "Corpus-level schedule evidence only. Missing semantic coverage is not restraint evidence; render quality and human preference are not established.",
    }
    write_json(output_path, report)
    if schemas_dir:
        validate_artifact("semantic_corpus_screen", output_path, schemas_dir)
    return report


def _case_row(case: dict[str, Any], screen: dict[str, Any], *, index: int) -> dict[str, Any]:
    assisted = [row for row in screen.get("variants", []) if row.get("mode") == "SEMANTIC_ASSISTED" and float(row.get("weight", 0.0)) > 0.0]
    representative = max(assisted, key=lambda row: float(row.get("weight", 0.0)), default={})
    mappings = int(representative.get("mapping_count", 0) or 0)
    covered = int(representative.get("semantic_placement_count", 0) or 0)
    aggregate_fallback = min(covered, int(representative.get("performance_aggregate_fallback_count", 0) or 0))
    boundary_bridge = int(representative.get("boundary_bridge_semantic_placement_count", 0) or 0)
    text_bridge = int(representative.get("text_bridge_semantic_placement_count", 0) or 0)
    exact_direct = int(representative.get("exact_direct_semantic_placement_count", max(0, covered - aggregate_fallback - boundary_bridge - text_bridge)) or 0)
    coverage = covered / mappings if mappings else 0.0
    coverage_state = "FULL" if mappings and covered == mappings else ("PARTIAL" if covered else "NONE")
    equivalents = [value for key, value in (screen.get("invariants") or {}).items() if key.endswith("_equivalent")]
    invariants_passed = bool(equivalents) and all(value is True for value in equivalents)
    changed = [row for row in assisted if int(row.get("placements_changed", 0) or 0) > 0]
    conflicted = [row for row in changed if int(row.get("conflict_count", 0) or 0) > 0]
    render_selection = list(screen.get("render_selection") or [])
    nominee = next((value for value in render_selection if value not in {"control", "report_only"}), None)
    if not invariants_passed:
        state = "INVARIANT_FAILURE"
    elif coverage_state == "NONE":
        state = "SEMANTIC_COVERAGE_NONE"
    elif coverage_state == "PARTIAL":
        state = "SEMANTIC_COVERAGE_PARTIAL"
    elif nominee:
        state = "RENDER_NOMINEE"
    elif conflicted:
        state = "CHANGED_WITH_CONFLICTS"
    else:
        state = "SAFE_NO_SELECTION_EFFECT"
    return {
        "case_id": str(case.get("case_id") or f"case_{index:03d}"),
        "screen": str(Path(case["screen"])),
        "source_class": str(case.get("source_class") or "unspecified"),
        "destination_class": str(case.get("destination_class") or "unspecified"),
        "scheduling_mode": screen.get("scheduling_mode"),
        "invariants_passed": invariants_passed,
        "mapping_count": mappings,
        "semantic_placement_count": covered,
        "direct_semantic_placement_count": covered - aggregate_fallback,
        "performance_aggregate_fallback_count": aggregate_fallback,
        "exact_direct_semantic_placement_count": exact_direct,
        "boundary_bridge_semantic_placement_count": boundary_bridge,
        "text_bridge_semantic_placement_count": text_bridge,
        "semantic_coverage": round(coverage, 6),
        "semantic_coverage_state": coverage_state,
        "changed_variant_count": len(changed),
        "conflicted_variant_count": len(conflicted),
        "maximum_placements_changed": max((int(row.get("placements_changed", 0) or 0) for row in assisted), default=0),
        "render_nominee": nominee,
        "case_state": state,
    }
