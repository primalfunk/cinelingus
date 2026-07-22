from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from . import __version__
from .editorial import EditorialMemory, build_repair_batch
from .editorial.repair_strategies import STRATEGIES
from .util import read_json, stable_hash, utc_now, write_json


PHASE0_BENCHMARK_VERSION = "phase0_strategy_contract_benchmarks_v1"
STRATEGY_ISOLATION_REVISION = "phase0_strategy_isolation_v1"


def run_strategy_contract_benchmarks(*, output_path: Path) -> dict[str, Any]:
    """Exercise every declared repair strategy through the real pre-render router."""
    cases = [_run_contract_case(category) for category in STRATEGIES]
    report = {
        "schema_version": "1.0",
        "report_version": PHASE0_BENCHMARK_VERSION,
        "tool_version": __version__,
        "creation_timestamp": utc_now(),
        "benchmark_level": "deterministic_pre_render_contract",
        "case_count": len(cases),
        "passed_case_count": sum(bool(row["passed"]) for row in cases),
        "failed_case_count": sum(not bool(row["passed"]) for row in cases),
        "all_declared_strategies_executable": all(bool(row["passed"]) for row in cases),
        "rendered_evidence_claimed": False,
        "cases": cases,
    }
    write_json(output_path, report)
    return report


def build_rendered_strategy_coverage(
    *, contract_report_path: Path, calibration_report_paths: list[Path], output_path: Path,
) -> dict[str, Any]:
    """Join executable contracts to observed rendered attempts without overstating either."""
    contract = read_json(contract_report_path)
    contract_by_strategy = {
        str(row.get("repair_strategy")): row for row in contract.get("cases", [])
    }
    observed: dict[str, dict[str, int]] = {}
    for report_path in calibration_report_paths:
        report = read_json(report_path)
        for result in report.get("results", []):
            if not result.get("informative"):
                continue
            for row in result.get("repair_strategies", []):
                strategy = str(row.get("name") or "unknown")
                totals = observed.setdefault(strategy, {
                    "attempt_count": 0, "rendered_count": 0, "survived_count": 0,
                })
                totals["attempt_count"] += int(row.get("attempt_count") or 0)
                totals["rendered_count"] += int(row.get("rendered_count") or 0)
                totals["survived_count"] += int(row.get("survived_count") or 0)

    rows = []
    for category, spec in STRATEGIES.items():
        strategy = str(spec["strategy"])
        evidence = observed.get(strategy, {})
        contract_passed = bool(contract_by_strategy.get(strategy, {}).get("passed"))
        attempt_count = int(evidence.get("attempt_count") or 0)
        rendered_count = int(evidence.get("rendered_count") or 0)
        survived_count = int(evidence.get("survived_count") or 0)
        conservative = strategy == "conservative_uncertainty_retention"
        evidence_satisfied = attempt_count > 0 if conservative else rendered_count > 0
        status = (
            "runtime_retention_observed" if conservative and attempt_count > 0
            else "contract_only" if conservative and contract_passed
            else "rendered_repair_survived" if survived_count > 0
            else "rendered_attempt_observed" if rendered_count > 0
            else "runtime_attempt_observed" if attempt_count > 0
            else "contract_only" if contract_passed
            else "contract_failed"
        )
        rows.append({
            "failure_category": category,
            "repair_strategy": strategy,
            "contract_executable": contract_passed,
            "runtime_attempt_count": attempt_count,
            "rendered_candidate_count": rendered_count,
            "surviving_repair_count": survived_count,
            "required_evidence": "runtime_retention" if conservative else "rendered_candidate",
            "evidence_satisfied": evidence_satisfied,
            "status": status,
        })

    rendered = [row for row in rows if row["rendered_candidate_count"] > 0]
    attempted = [row for row in rows if row["runtime_attempt_count"] > 0]
    survived = [row for row in rows if row["surviving_repair_count"] > 0]
    evidence_complete = [row for row in rows if row["evidence_satisfied"]]
    report = {
        "schema_version": "1.0",
        "report_version": "phase0_rendered_strategy_coverage_v1",
        "tool_version": __version__,
        "creation_timestamp": utc_now(),
        "contract_report": str(contract_report_path.resolve()),
        "calibration_reports": [str(path.resolve()) for path in calibration_report_paths],
        "declared_strategy_count": len(rows),
        "contract_executable_count": sum(bool(row["contract_executable"]) for row in rows),
        "runtime_attempted_strategy_count": len(attempted),
        "rendered_strategy_count": len(rendered),
        "surviving_strategy_count": len(survived),
        "evidence_satisfied_strategy_count": len(evidence_complete),
        "missing_rendered_strategies": [
            row["repair_strategy"] for row in rows
            if row["required_evidence"] == "rendered_candidate" and row["rendered_candidate_count"] == 0
        ],
        "missing_strategy_evidence": [
            row["repair_strategy"] for row in rows if not row["evidence_satisfied"]
        ],
        "phase0_rendered_strategy_gate_passed": len(evidence_complete) == len(rows),
        "strategies": rows,
    }
    write_json(output_path, report)
    return report


def build_strategy_isolation_plan(
    *, prior_plan_path: Path, prior_report_path: Path, coverage_report_path: Path,
    output_path: Path, max_cases: int | None = None,
) -> dict[str, Any]:
    """Re-run observed secondary failures as explicit primary-strategy benchmarks."""
    prior_plan = read_json(prior_plan_path)
    prior_report = read_json(prior_report_path)
    coverage = read_json(coverage_report_path)
    cases_by_id = {
        str(row.get("case_id")): row for row in prior_plan.get("cases", [])
    }
    missing = set(str(value) for value in coverage.get("missing_strategy_evidence", []))
    category_by_strategy = {str(spec["strategy"]): category for category, spec in STRATEGIES.items()}
    selected = []
    unfilled = []
    limit = len(missing) if max_cases is None else max(0, int(max_cases))
    for strategy in [str(spec["strategy"]) for spec in STRATEGIES.values() if str(spec["strategy"]) in missing]:
        if len(selected) >= limit:
            break
        category = category_by_strategy[strategy]
        candidates = []
        for result in prior_report.get("results", []):
            if not result.get("informative") or str(result.get("case_id")) not in cases_by_id:
                continue
            evidence = next(
                (row for row in result.get("failure_categories", []) if row.get("name") == category),
                None,
            )
            if evidence and int(evidence.get("attempt_count") or 0) > 0:
                candidates.append((
                    -int(evidence.get("attempt_count") or 0),
                    str(result.get("case_id")), result,
                ))
        if not candidates:
            unfilled.append(strategy)
            continue
        _negative_attempts, source_case_id, _result = min(candidates)
        row = deepcopy(cases_by_id[source_case_id])
        row["case_id"] = f"strategy_{len(selected) + 1:03d}_{category}"
        row["purpose"] = f"strategy_isolation_{category}"
        row["target_failure_category"] = category
        row["target_repair_strategy"] = strategy
        row["expected_failure_modes"] = [category]
        row["strategy_isolation_source_case_id"] = source_case_id
        row["status"] = "planned"
        row["case_signature"] = stable_hash({
            "execution_revision": STRATEGY_ISOLATION_REVISION,
            "source_case_signature": cases_by_id[source_case_id].get("case_signature"),
            "target_failure_category": category,
            "target_repair_strategy": strategy,
        })
        selected.append(row)
    plan = {
        "schema_version": "1.0",
        "plan_version": "bounded_corpus_calibration_v1",
        "tool_version": __version__,
        "creation_timestamp": utc_now(),
        "execution_revision": STRATEGY_ISOLATION_REVISION,
        "tier": prior_plan.get("tier", "extended"),
        "excerpt_plan": prior_plan.get("excerpt_plan"),
        "maximum_cases": limit,
        "case_count": len(selected),
        "planned_input_duration_seconds": round(sum(
            float(case[role]["duration"]) for case in selected for role in ("destination", "source")
        ), 3),
        "source_files_read_only": True,
        "resume_required": True,
        "strategy_isolation_of": {
            "plan": str(prior_plan_path.resolve()),
            "report": str(prior_report_path.resolve()),
            "coverage": str(coverage_report_path.resolve()),
        },
        "unfilled_target_strategies": unfilled,
        "cases": selected,
    }
    write_json(output_path, plan)
    return plan


def build_observed_failure_strategy_plan(
    *, config_path: Path, editorial_report_path: Path, schedule_path: Path,
    coverage_report_path: Path, output_path: Path, max_cases: int | None = None,
    destination_context_seconds: float = 10.0, source_context_seconds: float = 15.0,
    variants_per_strategy: int = 1,
) -> dict[str, Any]:
    """Turn observed long-run placement failures into bounded strategy cases."""
    config = read_json(config_path)
    editorial = read_json(editorial_report_path)
    schedule = read_json(schedule_path)
    coverage = read_json(coverage_report_path)
    missing = set(str(value) for value in coverage.get("missing_strategy_evidence", []))
    category_by_strategy = {str(spec["strategy"]): category for category, spec in STRATEGIES.items()}
    mappings = {
        str(row.get("editorial_placement_id")): row for row in schedule.get("mappings", [])
        if row.get("editorial_placement_id")
    }
    destination_path = _config_media_path(config_path, config.get("destination_video"))
    source_path = _config_media_path(config_path, config.get("source_dialogue"))
    limit = (
        len(missing) * max(1, int(variants_per_strategy))
        if max_cases is None else max(0, int(max_cases))
    )
    selected = []
    unfilled = []
    for strategy in [str(spec["strategy"]) for spec in STRATEGIES.values() if str(spec["strategy"]) in missing]:
        if len(selected) >= limit:
            break
        category = category_by_strategy[strategy]
        candidates = []
        for decision in editorial.get("decisions", []):
            categories = {str(row.get("category")) for row in decision.get("failures", [])}
            key = str(decision.get("placement_key") or "")
            if category in categories and key in mappings:
                candidates.append((float(decision.get("overall_quality", 1.0) or 1.0), key, decision))
        if not candidates:
            unfilled.append(strategy)
            continue
        ordered = sorted(candidates, key=lambda row: (row[0], row[1]))
        for variant_index, (_quality, placement_key, decision) in enumerate(
            ordered[:max(1, int(variants_per_strategy))], start=1
        ):
            if len(selected) >= limit:
                break
            mapping = mappings[placement_key]
            destination_start = max(0.0, float(decision.get("destination_start", 0.0) or 0.0) - destination_context_seconds)
            destination_end = max(
                destination_start + 1.0,
                float(decision.get("destination_end", destination_start) or destination_start) + destination_context_seconds,
            )
            source_center = float(
                mapping.get("source_movie_timestamp", mapping.get("clip_movie_timestamp", 0.0)) or 0.0
            )
            source_start = max(0.0, source_center - source_context_seconds)
            source_duration = max(
                1.0,
                source_context_seconds * 2.0 + float(mapping.get("clip_trim_duration", 0.0) or 0.0),
            )
            destination = _observed_excerpt(
                role="destination", category=category, source_path=destination_path,
                start=destination_start, duration=destination_end - destination_start,
                placement_key=placement_key, quality=float(decision.get("overall_quality", 0.0) or 0.0),
            )
            source = _observed_excerpt(
                role="source", category=category, source_path=source_path,
                start=source_start, duration=source_duration,
                placement_key=placement_key, quality=float(decision.get("overall_quality", 0.0) or 0.0),
            )
            case = {
                "case_id": f"observed_{len(selected) + 1:03d}_{category}_v{variant_index}",
                "purpose": f"observed_strategy_isolation_{category}",
                "variant_index": variant_index,
                "destination": destination,
                "source": source,
                "expected_failure_modes": [category],
                "target_failure_category": category,
                "target_repair_strategy": strategy,
                "observed_source_placement_key": placement_key,
                "status": "planned",
            }
            case["case_signature"] = stable_hash({
                "execution_revision": "phase0_observed_failure_isolation_v1",
                "target_failure_category": category,
                "destination": destination,
                "source": source,
            })
            selected.append(case)
    plan = {
        "schema_version": "1.0",
        "plan_version": "bounded_corpus_calibration_v1",
        "tool_version": __version__,
        "creation_timestamp": utc_now(),
        "execution_revision": "phase0_observed_failure_isolation_v1",
        "tier": "extended",
        "excerpt_plan": str(editorial_report_path.resolve()),
        "maximum_cases": limit,
        "case_count": len(selected),
        "planned_input_duration_seconds": round(sum(
            float(case[role]["duration"]) for case in selected for role in ("destination", "source")
        ), 3),
        "source_files_read_only": True,
        "resume_required": True,
        "observed_failure_source": {
            "config": str(config_path.resolve()),
            "editorial_report": str(editorial_report_path.resolve()),
            "schedule": str(schedule_path.resolve()),
            "coverage": str(coverage_report_path.resolve()),
        },
        "unfilled_target_strategies": unfilled,
        "cases": selected,
    }
    write_json(output_path, plan)
    return plan


def _config_media_path(config_path: Path, value: Any) -> Path:
    path = Path(str(value or ""))
    return path if path.is_absolute() else (config_path.parent / path).resolve()


def _observed_excerpt(
    *, role: str, category: str, source_path: Path, start: float, duration: float,
    placement_key: str, quality: float,
) -> dict[str, Any]:
    end = start + duration
    return {
        "excerpt_id": f"observed_{role}_{stable_hash([str(source_path), start, end, category])[:12]}",
        "media_id": f"observed_{stable_hash(str(source_path))[:16]}",
        "filename": source_path.name,
        "source_path": str(source_path),
        "content_type": "unknown",
        "category": f"observed_{category}_{role}",
        "start": round(start, 3),
        "end": round(end, 3),
        "duration": round(duration, 3),
        "analysis_signature": stable_hash([str(source_path), start, end]),
        "evidence": {
            "placement_key": placement_key,
            "observed_failure_category": category,
            "observed_quality": round(quality, 4),
        },
    }


def _run_contract_case(category: str) -> dict[str, Any]:
    spec = STRATEGIES[category]
    mapping = _mapping_for(category)
    original = deepcopy(mapping)
    decision = {
        "placement_key": "editorial_placement_000001",
        "mapping_index": 0,
        "overall_quality": 0.3,
        "recommendation": "repair",
        "failures": [{"category": category, "severity": "high", "confidence": 1.0}],
    }
    donors = [
        _donor("clip-a", 0.3, "performance-a"),
        _donor("clip-b", 0.9, "performance-b"),
    ]

    def score_candidate(_window: dict[str, Any], clip: dict[str, Any]) -> dict[str, Any]:
        score = float(clip.get("benchmark_score", 0.0) or 0.0)
        return {
            "score": score,
            "editorial_score_model": "phase0_contract_fixture_v1",
            "editorial_components": {
                "sentence_fit": score,
                "timing_and_render_fit": score,
                "speaker_role_fit": score,
                "performance_fit": score,
                "visual_fit": score,
                "transition_cleanliness": score,
                "intelligibility": score,
                "confidence": score,
                "reuse_integrity": score,
            },
        }

    def build_mapping(window: dict[str, Any], clip: dict[str, Any], score: dict[str, Any]) -> dict[str, Any]:
        return {
            "editorial_placement_id": "editorial_placement_000001",
            "clip_id": clip["id"],
            "source_performance_id": clip.get("source_performance_id"),
            "destination_timestamp": window["start"],
            "planned_render_duration": window["duration"],
            "alignment_slot_start": window["start"],
            "alignment_slot_end": window["end"],
            "editorial_candidate_score": score["score"],
            "enabled": True,
        }

    batch = build_repair_batch(
        schedule={"mappings": [mapping], "active_filter": "balanced"},
        decisions={"decisions": [decision]}, donor_candidates=donors,
        memory=EditorialMemory(), score_candidate=score_candidate,
        build_mapping=build_mapping, maximum_repairs=1,
    )
    attempt = batch["attempts"][0]
    conservative = spec["strategy"] == "conservative_uncertainty_retention"
    actual_behavior = (
        "conservative_retention"
        if attempt.get("candidate_loss_stage") == "conservative_retention"
        else "proposal" if attempt.get("proposed") else "no_viable_alternative"
    )
    expected_behavior = "conservative_retention" if conservative else "proposal"
    passed = (
        str(attempt.get("repair_strategy")) == str(spec["strategy"])
        and actual_behavior == expected_behavior
        and ((batch["schedule"]["mappings"][0] != original) if not conservative else not batch["repairs"])
    )
    return {
        "benchmark_id": f"strategy_{category}",
        "failure_category": category,
        "repair_strategy": spec["strategy"],
        "expected_behavior": expected_behavior,
        "actual_behavior": actual_behavior,
        "candidate_family": attempt.get("candidate_family"),
        "candidate_loss_stage": attempt.get("candidate_loss_stage"),
        "mapping_changed": batch["schedule"]["mappings"][0] != original,
        "passed": bool(passed),
    }


def _mapping_for(category: str) -> dict[str, Any]:
    mapping = {
        "editorial_placement_id": "editorial_placement_000001",
        "clip_id": "clip-a",
        "source_performance_id": "performance-a",
        "destination_timestamp": 10.0,
        "planned_render_duration": 2.0,
        "alignment_slot_start": 10.0,
        "alignment_slot_end": 12.0,
        "clip_trim_start": 0.0,
        "clip_trim_duration": 2.0,
        "stretch_factor": 1.0,
        "enabled": True,
    }
    if category == "duration_failure":
        mapping.update({
            "stretch_factor": 1.2,
            "planned_render_duration": 2.4,
            "alignment_slot_end": 13.0,
        })
    return mapping


def _donor(clip_id: str, score: float, performance_id: str) -> dict[str, Any]:
    return {
        "id": clip_id,
        "duration": 2.0,
        "transcript": "This is a complete benchmark sentence.",
        "source_performance_id": performance_id,
        "benchmark_score": score,
    }
