from __future__ import annotations

from pathlib import Path
from typing import Any

from . import __version__
from .util import read_json, utc_now, write_json


DEFAULT_THRESHOLDS = {
    "accepted_residue_statuses": ["NONE_DETECTED"],
    "maximum_silence_fallback_ratio": 0.10,
    "minimum_performance_first_ratio": 0.75,
    "maximum_linewise_fallback_ratio": 0.10,
    "maximum_preserved_original_ratio": 0.25,
    "maximum_problem_count": 10,
    "require_editorial_evidence": True,
    "accepted_editorial_statuses": ["PASS", "LIMIT_REACHED"],
    "maximum_hard_gate_failure_count": 0,
    "maximum_failed_delivery_count": 0,
}


def evaluate_quality_corpus(
    *,
    manifest_path: Path,
    runs_root: Path,
    output_path: Path,
) -> dict[str, Any]:
    manifest = read_json(manifest_path)
    defaults = {**DEFAULT_THRESHOLDS, **dict(manifest.get("thresholds") or {})}
    cases = [
        _evaluate_case(case=dict(case), thresholds=defaults, runs_root=runs_root)
        for case in manifest.get("cases", [])
    ]
    report = {
        "schema_version": "1.0",
        "tool_version": __version__,
        "creation_timestamp": utc_now(),
        "manifest": str(manifest_path),
        "runs_root": str(runs_root),
        "case_count": len(cases),
        "passed_case_count": sum(1 for case in cases if case["passed"]),
        "failed_case_count": sum(1 for case in cases if not case["passed"]),
        "passed": bool(cases) and all(case["passed"] for case in cases),
        "cases": cases,
    }
    write_json(output_path, report)
    return report


def _evaluate_case(*, case: dict[str, Any], thresholds: dict[str, Any], runs_root: Path) -> dict[str, Any]:
    case_id = str(case.get("id") or "unnamed_case")
    case_thresholds = {**thresholds, **dict(case.get("thresholds") or {})}
    relative_report = Path(str(case.get("run_report") or case_id))
    if relative_report.suffix.lower() != ".json":
        relative_report = relative_report / "run_report.json"
    report_path = _resolve_within(runs_root, relative_report)
    if not report_path.exists():
        return {
            "id": case_id,
            "tags": list(case.get("tags") or []),
            "run_report": str(report_path),
            "passed": False,
            "metrics": {},
            "checks": [{"name": "run_report_exists", "passed": False, "actual": False, "expected": True}],
        }
    report = read_json(report_path)
    editorial_path = report_path.parent / "editorial_report.json"
    editorial = read_json(editorial_path) if editorial_path.exists() else {}
    schedule = dict(report.get("schedule") or {})
    performance = dict(schedule.get("performance_summary") or {})
    residue = dict(schedule.get("voice_residue_verification") or {})
    bed = dict(report.get("soundtrack_bed") or {})
    problems = dict(report.get("problem_region_report") or {})
    total = int(performance.get("destination_performance_count") or 0)
    first = sum(int(performance.get(key) or 0) for key in ("performance_couplings", "adapted_performances", "turn_sequence_matches"))
    linewise = int(performance.get("linewise_fallbacks") or 0)
    preserved = int(performance.get("preserved_original_regions") or 0)
    suppressed = int(performance.get("suppressed_unreplaced_regions") or 0)
    reconstructed = int(bed.get("reconstructed_region_count") or 0)
    silence = int(bed.get("silence_fallback_region_count") or 0)
    ambience_total = reconstructed + silence
    decisions = list(editorial.get("decisions") or [])
    hard_gate_failures = [row for row in decisions if _decision_hard_failures(row)]
    final_states = dict(editorial.get("final_state_counts") or {})
    metrics = {
        "residue_status": residue.get("status") or performance.get("voice_residue") or "UNAVAILABLE",
        "performance_first_ratio": round(first / total, 4) if total else 0.0,
        "linewise_fallback_ratio": round(linewise / total, 4) if total else 0.0,
        "preserved_original_ratio": round(preserved / total, 4) if total else 0.0,
        "suppressed_unreplaced_ratio": round(suppressed / total, 4) if total else 0.0,
        "silence_fallback_ratio": round(silence / ambience_total, 4) if ambience_total else 0.0,
        "problem_count": int(problems.get("problem_count") or 0),
        "editorial_evidence_available": bool(editorial),
        "editorial_status": editorial.get("status") or "UNAVAILABLE",
        "editorial_final_quality": editorial.get("final_quality"),
        "minimum_placement_quality": min(
            (float(row.get("overall_quality", 0.0) or 0.0) for row in decisions), default=None,
        ),
        "hard_gate_failure_count": len(hard_gate_failures),
        "hard_gate_failure_categories": sorted({
            category for row in hard_gate_failures for category in _decision_hard_failures(row)
        }),
        "failed_delivery_count": int(final_states.get("FAILED_DELIVERY") or 0),
    }
    checks = [
        _membership_check("residue_status", metrics["residue_status"], case_thresholds["accepted_residue_statuses"]),
        _minimum_check("performance_first_ratio", metrics["performance_first_ratio"], case_thresholds["minimum_performance_first_ratio"]),
        _maximum_check("linewise_fallback_ratio", metrics["linewise_fallback_ratio"], case_thresholds["maximum_linewise_fallback_ratio"]),
        _maximum_check("preserved_original_ratio", metrics["preserved_original_ratio"], case_thresholds["maximum_preserved_original_ratio"]),
        _maximum_check("silence_fallback_ratio", metrics["silence_fallback_ratio"], case_thresholds["maximum_silence_fallback_ratio"]),
        _maximum_check("problem_count", metrics["problem_count"], case_thresholds["maximum_problem_count"]),
        _boolean_requirement_check(
            "editorial_evidence_available", metrics["editorial_evidence_available"],
            bool(case_thresholds["require_editorial_evidence"]),
        ),
        _membership_check(
            "editorial_status", metrics["editorial_status"],
            case_thresholds["accepted_editorial_statuses"],
        ),
        _maximum_check(
            "hard_gate_failure_count", metrics["hard_gate_failure_count"],
            case_thresholds["maximum_hard_gate_failure_count"],
        ),
        _maximum_check(
            "failed_delivery_count", metrics["failed_delivery_count"],
            case_thresholds["maximum_failed_delivery_count"],
        ),
    ]
    return {
        "id": case_id,
        "tags": list(case.get("tags") or []),
        "run_report": str(report_path),
        "passed": all(check["passed"] for check in checks),
        "metrics": metrics,
        "checks": checks,
    }


def _resolve_within(root: Path, relative: Path) -> Path:
    root = root.resolve()
    candidate = relative if relative.is_absolute() else root / relative
    candidate = candidate.resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError(f"Corpus run report escapes runs root: {relative}")
    return candidate


def _membership_check(name: str, actual: Any, expected: list[Any]) -> dict[str, Any]:
    return {"name": name, "passed": actual in expected, "actual": actual, "expected": expected}


def _minimum_check(name: str, actual: float, expected: float) -> dict[str, Any]:
    return {"name": name, "passed": float(actual) >= float(expected), "actual": actual, "expected_minimum": expected}


def _maximum_check(name: str, actual: float, expected: float) -> dict[str, Any]:
    return {"name": name, "passed": float(actual) <= float(expected), "actual": actual, "expected_maximum": expected}


def _boolean_requirement_check(name: str, actual: bool, required: bool) -> dict[str, Any]:
    return {
        "name": name,
        "passed": bool(actual) if required else True,
        "actual": bool(actual),
        "required": bool(required),
    }


def _decision_hard_failures(decision: dict[str, Any]) -> list[str]:
    explicit = [str(value) for value in decision.get("hard_gate_failures", [])]
    if explicit:
        return sorted(set(explicit))
    categories = []
    for row in decision.get("failures", []):
        category = str(row.get("category") or "")
        coverage = float((row.get("evidence") or {}).get("coverage", 1.0) or 0.0)
        if category in {"mid_word_cut", "residual_dialogue"} or (
            category == "low_rendered_coverage" and coverage < 0.25
        ):
            categories.append(category)
    return sorted(set(categories))
