from __future__ import annotations

from pathlib import Path
from typing import Any

from ..schedule import build_schedule
from ..semantic.config import SemanticMode
from ..semantic.scheduling import SemanticScheduleContext
from ..util import stable_hash, utc_now, write_json
from .scheduling import FunctionMode, FunctionScheduleContext

FUNCTION_SCREEN_VERSION = "function_preserving_schedule_screen_v1"
TECHNICAL_FIELDS = {
    "performance": ("performance_similarity_score",),
    "duration": ("score_components", "duration_similarity"),
    "speaker": ("speaker_pattern_match",),
    "visual": ("visual_fit_score",),
    "completeness": ("cinematic_compatibility_components", "audio", "transcript_completeness"),
}


def run_function_schedule_screen(
    *, clips: list[dict[str, Any]], windows: list[dict[str, Any]],
    semantic_evidence: SemanticScheduleContext, function_evidence: FunctionScheduleContext,
    output_dir: Path, source_hash: str, destination_hash: str, max_time_stretch: float,
    semantic_weight: float = 0.05, function_weight: float = 0.15,
    scheduling_mode: str = "best_fit", best_fit_lookahead: int = 8,
    shot_boundary_mode: str = "off", cinematic_filter: str = "balanced",
    source_performances: dict[str, Any] | None = None, speaker_mapping: dict[str, Any] | None = None,
    calibration: dict[str, Any] | None = None,
) -> dict[str, Any]:
    for name, value in (("semantic", semantic_weight), ("function", function_weight)):
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{name.capitalize()} screen weight must be between 0 and 1")
    output_dir.mkdir(parents=True, exist_ok=True)
    common = {
        "clips": clips, "windows": windows, "source_hash": source_hash, "destination_hash": destination_hash,
        "max_time_stretch": max_time_stretch, "scheduling_mode": scheduling_mode,
        "best_fit_lookahead": best_fit_lookahead, "shot_boundary_mode": shot_boundary_mode,
        "cinematic_filter": cinematic_filter, "source_performances": source_performances,
        "speaker_mapping": speaker_mapping,
    }
    semantic_assisted = _semantic_mode(semantic_evidence, SemanticMode.ASSISTED, semantic_weight)
    schedules = {
        "legacy_control": build_schedule(**common, output_path=output_dir / "legacy_control_schedule.json"),
        "semantic_only": build_schedule(**common, output_path=output_dir / "semantic_only_schedule.json", semantic_context=semantic_assisted),
        "function_report_only": build_schedule(
            **common, output_path=output_dir / "function_report_only_schedule.json",
            semantic_context=semantic_assisted,
            function_context=_function_mode(function_evidence, FunctionMode.REPORT_ONLY, 0.0),
        ),
        "function_zero_weight": build_schedule(
            **common, output_path=output_dir / "function_zero_weight_schedule.json",
            semantic_context=semantic_assisted,
            function_context=_function_mode(function_evidence, FunctionMode.ASSISTED, 0.0),
        ),
        "function_preserving": build_schedule(
            **common, output_path=output_dir / "function_preserving_schedule.json",
            semantic_context=semantic_assisted,
            function_context=_function_mode(function_evidence, FunctionMode.PRESERVING, function_weight),
        ),
    }
    semantic_only, report_only, zero = schedules["semantic_only"], schedules["function_report_only"], schedules["function_zero_weight"]
    invariants = {
        "report_only_selection_equivalent_to_semantic_only": _selection_signature(report_only) == _selection_signature(semantic_only),
        "report_only_scores_equivalent_to_semantic_only": _score_signature(report_only) == _score_signature(semantic_only),
        "zero_weight_selection_equivalent_to_semantic_only": _selection_signature(zero) == _selection_signature(semantic_only),
        "zero_weight_scores_equivalent_to_semantic_only": _score_signature(zero) == _score_signature(semantic_only),
        "candidate_generation_policy": "EXISTING_LEGAL_CANDIDATES_ONLY",
        "hard_constraints_remain_authoritative": True,
        "semantic_assistance_remains_available": True,
    }
    if not all(value for key, value in invariants.items() if key.endswith("semantic_only")):
        raise RuntimeError("Dialogue-function zero-influence invariant failed")
    variants = [
        _variant(name, schedule, schedules["legacy_control"], report_only)
        for name, schedule in schedules.items() if name != "function_zero_weight"
    ]
    preserving = next(row for row in variants if row["variant_id"] == "function_preserving")
    calibration_state = (calibration or {}).get("review_state", "MISSING")
    technically_nominable = (
        preserving["placements_changed_from_report_only"] > 0
        and preserving["technical_regression_count"] == 0
        and preserving["available_function_placement_count"] == preserving["mapping_count"]
    )
    review_ready = technically_nominable and calibration_state == "COMPLETE"
    counterfactuals = _counterfactuals(
        report_only.get("mappings") or [], schedules["function_preserving"].get("mappings") or []
    )
    report = {
        "schema_version": "1.0", "experiment_version": FUNCTION_SCREEN_VERSION,
        "creation_timestamp": utc_now(),
        "experiment_signature": stable_hash({
            "source_hash": source_hash, "destination_hash": destination_hash,
            "semantic_weight": semantic_weight, "function_weight": function_weight,
            "scheduling_mode": scheduling_mode, "semantic_identity": semantic_evidence.model_identity,
            "function_identity": function_evidence.identity, "clips": clips, "windows": windows,
        }),
        "source_hash": source_hash, "destination_hash": destination_hash,
        "scheduling_mode": scheduling_mode, "semantic_weight": semantic_weight, "function_weight": function_weight,
        "semantic_model_identity": semantic_evidence.model_identity,
        "function_model_identity": function_evidence.identity,
        "calibration_state": calibration_state,
        "invariants": invariants, "variants": variants,
        "counterfactuals": counterfactuals,
        "render_selection": ["legacy_control", "semantic_only", "function_report_only", "function_preserving"] if review_ready else [],
        "render_selection_state": (
            "FOUR_WAY_CANDIDATE_SELECTED" if review_ready else
            "BLOCKED_PENDING_REVIEWED_CALIBRATION" if technically_nominable else
            "NO_CONFLICT_FREE_CHANGED_FUNCTION_CANDIDATE"
        ),
        "claim_scope": "Schedule-only provisional dialogue-function screening; render quality and human preference are not established.",
    }
    write_json(output_dir / "function_schedule_screen.json", report)
    return report


def _variant(name: str, schedule: dict[str, Any], legacy: dict[str, Any], report_only: dict[str, Any]) -> dict[str, Any]:
    mappings = schedule.get("mappings") or []
    functions = [row["dialogue_function_compatibility"] for row in mappings if (row.get("dialogue_function_compatibility") or {}).get("available")]
    regressions = _technical_regressions(report_only.get("mappings") or [], mappings)
    return {
        "variant_id": name,
        "mode": {
            "legacy_control": FunctionMode.DISABLED.value,
            "semantic_only": FunctionMode.DISABLED.value,
            "function_report_only": FunctionMode.REPORT_ONLY.value,
            "function_preserving": FunctionMode.PRESERVING.value,
        }[name],
        "mapping_count": len(mappings),
        "placements_changed_from_legacy": _changed_count(legacy.get("mappings") or [], mappings),
        "placements_changed_from_report_only": _changed_count(report_only.get("mappings") or [], mappings),
        "available_function_placement_count": len(functions),
        "confidence_covered_placement_count": sum(1 for row in functions if float(row.get("confidence", 0.0)) >= 0.62),
        "ambiguous_placement_count": sum(1 for row in functions if row.get("ambiguity")),
        "mean_function_preservation": round(sum(float(row["normalized_function_contribution"]) for row in functions) / len(functions), 6) if functions else None,
        "mean_semantic_similarity": _mean_semantic(mappings),
        "technical_regression_count": len(regressions), "technical_regressions": regressions,
        "unique_donor_count": len({row.get("clip_id") for row in mappings}),
        "schedule_file": f"{name}_schedule.json",
    }


def _counterfactuals(before: list[dict[str, Any]], after: list[dict[str, Any]]) -> dict[str, Any]:
    high_semantic_wrong_function, lower_semantic_right_function = [], []
    for index, (old, new) in enumerate(zip(before, after)):
        old_function = (old.get("dialogue_function_compatibility") or {}).get("normalized_function_contribution")
        new_function = (new.get("dialogue_function_compatibility") or {}).get("normalized_function_contribution")
        old_semantic = (old.get("semantic_compatibility") or {}).get("raw_cosine_similarity")
        new_semantic = (new.get("semantic_compatibility") or {}).get("raw_cosine_similarity")
        common = {"placement_index": index, "window_id": new.get("window_id"), "before_clip_id": old.get("clip_id"), "after_clip_id": new.get("clip_id")}
        if old_semantic is not None and old_function is not None and float(old_semantic) >= 0.75 and float(old_function) < 0.5:
            high_semantic_wrong_function.append({**common, "semantic_similarity": old_semantic, "function_compatibility": old_function})
        if _mapping_key(old) != _mapping_key(new) and None not in {old_semantic, new_semantic, old_function, new_function} and float(new_semantic) < float(old_semantic) and float(new_function) > float(old_function):
            lower_semantic_right_function.append({**common, "semantic_delta": round(float(new_semantic) - float(old_semantic), 6), "function_delta": round(float(new_function) - float(old_function), 6)})
    return {
        "high_cosine_wrong_function_candidates": high_semantic_wrong_function,
        "lower_cosine_right_function_candidates": lower_semantic_right_function,
        "review_status": "CANDIDATES_ONLY_UNTIL_HUMAN_CALIBRATION",
    }


def _technical_regressions(before: list[dict[str, Any]], after: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for index, (old, new) in enumerate(zip(before, after)):
        if _mapping_key(old) == _mapping_key(new):
            continue
        regressions = {}
        for name, path in TECHNICAL_FIELDS.items():
            left, right = _number(old, path), _number(new, path)
            if left is not None and right is not None and right < left:
                regressions[name] = round(right - left, 6)
        if regressions:
            rows.append({"placement_index": index, "window_id": new.get("window_id"), "before_clip_id": old.get("clip_id"), "after_clip_id": new.get("clip_id"), "regressions": regressions})
    return rows


def _semantic_mode(context: SemanticScheduleContext, mode: SemanticMode, weight: float) -> SemanticScheduleContext:
    return SemanticScheduleContext(
        mode, weight, context.source_by_reference, context.destination_by_reference, context.model_identity,
        context.source_by_start, context.destination_by_start, context.source_by_text, context.destination_by_text,
        context.source_by_performance, context.destination_by_performance,
    )


def _function_mode(context: FunctionScheduleContext, mode: FunctionMode, weight: float) -> FunctionScheduleContext:
    return FunctionScheduleContext(
        mode, weight, context.source_by_reference, context.destination_by_reference, context.identity,
        context.minimum_confidence, context.source_by_start, context.destination_by_start,
        context.source_by_text, context.destination_by_text,
    )


def _selection_signature(schedule: dict[str, Any]) -> list[tuple[Any, ...]]:
    return [_mapping_key(row) for row in schedule.get("mappings") or []]


def _mapping_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (row.get("window_id"), row.get("destination_performance_id"), row.get("clip_id"), row.get("source_performance_id"))


def _score_signature(schedule: dict[str, Any]) -> list[float]:
    return [float(row.get("score", 0.0)) for row in schedule.get("mappings") or []]


def _changed_count(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> int:
    maximum = max(len(left), len(right))
    return sum(index >= len(left) or index >= len(right) or _mapping_key(left[index]) != _mapping_key(right[index]) for index in range(maximum))


def _mean_semantic(rows: list[dict[str, Any]]) -> float | None:
    values = [float(value) for row in rows if (value := (row.get("semantic_compatibility") or {}).get("raw_cosine_similarity")) is not None]
    return round(sum(values) / len(values), 6) if values else None


def _number(row: dict[str, Any], path: tuple[str, ...]) -> float | None:
    value: Any = row
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
