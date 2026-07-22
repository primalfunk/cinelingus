from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

from . import __version__
from .config import load_config
from .pipeline import Pipeline
from .tools import ffprobe_json, run
from .util import read_json, stable_hash, utc_now, write_json


CALIBRATION_RECIPES = (
    {
        "id": "exchange_continuity",
        "destination": "rapid_speaker_exchange",
        "sources": ("dense_dialogue", "animation_dialogue"),
        "failure_modes": ("speaker_mismatch", "performance_mismatch", "masking"),
    },
    {
        "id": "transition_sentence_integrity",
        "destination": "transition_near_dialogue",
        "sources": ("long_monologue", "dense_dialogue"),
        "failure_modes": ("transition_artifact", "incomplete_sentence", "duration_failure"),
    },
    {
        "id": "animation_visual_role",
        "destination": "animation_dialogue",
        "sources": ("rapid_speaker_exchange", "short_fragmented_lines", "dense_dialogue"),
        "failure_modes": ("visual_mismatch", "speaker_mismatch", "performance_mismatch"),
    },
    {
        "id": "fragment_duration_edges",
        "destination": "short_fragmented_lines",
        "sources": ("long_monologue", "dense_dialogue"),
        "failure_modes": ("mid_word_cut", "duration_failure", "transition_artifact"),
    },
)

CALIBRATION_EXECUTION_REVISION = "rendered_failure_exploration_v1"


def build_calibration_plan(
    *, excerpt_plan_path: Path, output_path: Path, max_cases: int | None = None,
) -> dict[str, Any]:
    excerpt_plan = read_json(excerpt_plan_path)
    tier = str(excerpt_plan.get("tier") or "smoke")
    defaults = {"smoke": 3, "standard": 5, "extended": 10}
    limit = max(0, int(max_cases if max_cases is not None else defaults.get(tier, 3)))
    excerpts = list(excerpt_plan.get("excerpts") or [])
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for excerpt in excerpts:
        by_category[str(excerpt.get("category") or "unknown")].append(dict(excerpt))
    for rows in by_category.values():
        rows.sort(key=lambda row: (str(row.get("media_id")), float(row.get("start", 0.0))))

    cases: list[dict[str, Any]] = []
    used_pairs: set[tuple[str, str]] = set()
    destination_usage: Counter[str] = Counter()
    source_usage: Counter[str] = Counter()
    variant_counts: Counter[str] = Counter()
    while len(cases) < limit:
        added_this_round = 0
        for recipe in CALIBRATION_RECIPES:
            destinations = by_category.get(str(recipe["destination"]), [])
            source_rows = [row for category in recipe["sources"] for row in by_category.get(str(category), [])]
            options = [
                (destination, source)
                for destination in destinations
                for source in source_rows
                if destination.get("excerpt_id") != source.get("excerpt_id")
                and destination.get("source_path") != source.get("source_path")
                and _calibration_excerpt_eligible(destination, role="destination")
                and _calibration_excerpt_eligible(source, role="source")
                and (str(destination.get("excerpt_id")), str(source.get("excerpt_id"))) not in used_pairs
            ]
            if not options:
                continue
            destination, source = min(
                options,
                key=lambda pair: (
                    destination_usage[str(pair[0].get("media_id"))],
                    source_usage[str(pair[1].get("media_id"))],
                    destination_usage[str(pair[0].get("media_id"))] + source_usage[str(pair[1].get("media_id"))],
                    str(pair[0].get("media_id")), float(pair[0].get("start", 0.0)),
                    str(pair[1].get("media_id")), float(pair[1].get("start", 0.0)),
                ),
            )
            pair_key = (str(destination.get("excerpt_id")), str(source.get("excerpt_id")))
            used_pairs.add(pair_key)
            destination_usage[str(destination.get("media_id"))] += 1
            source_usage[str(source.get("media_id"))] += 1
            variant_counts[str(recipe["id"])] += 1
            case = {
                "case_id": f"case_{len(cases) + 1:03d}_{recipe['id']}",
                "purpose": recipe["id"],
                "variant_index": variant_counts[str(recipe["id"])],
                "destination": _case_excerpt(destination),
                "source": _case_excerpt(source),
                "expected_failure_modes": list(recipe["failure_modes"]),
                "status": "planned",
            }
            case["case_signature"] = stable_hash({
                "execution_revision": CALIBRATION_EXECUTION_REVISION,
                "purpose": case["purpose"], "variant_index": case["variant_index"],
                "destination": case["destination"], "source": case["source"],
                "expected_failure_modes": case["expected_failure_modes"],
            })
            cases.append(case)
            added_this_round += 1
            if len(cases) >= limit:
                break
        if not added_this_round:
            break

    plan = {
        "schema_version": "1.0",
        "plan_version": "bounded_corpus_calibration_v1",
        "tool_version": __version__,
        "creation_timestamp": utc_now(),
        "execution_revision": CALIBRATION_EXECUTION_REVISION,
        "tier": tier,
        "excerpt_plan": str(excerpt_plan_path.resolve()),
        "maximum_cases": limit,
        "case_count": len(cases),
        "planned_input_duration_seconds": round(sum(
            float(case[role]["duration"]) for case in cases for role in ("destination", "source")
        ), 3),
        "source_files_read_only": True,
        "resume_required": True,
        "cases": cases,
    }
    write_json(output_path, plan)
    return plan


def build_calibration_followup_plan(
    *, candidate_plan_path: Path, prior_plan_path: Path, prior_report_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    candidate_plan = read_json(candidate_plan_path)
    prior_plan = read_json(prior_plan_path)
    prior_report = read_json(prior_report_path)
    fallback_needs: Counter[str] = Counter(
        str(row.get("purpose") or "unknown")
        for row in prior_report.get("results", [])
        if row.get("status") == "completed" and not row.get("informative")
    )
    attempted = {_case_input_identity(case) for case in prior_plan.get("cases", [])}
    selected = []
    ordered_candidates = sorted(
        candidate_plan.get("cases", []),
        key=lambda case: (-int(case.get("variant_index", 0) or 0), str(case.get("case_id"))),
    )
    for case in ordered_candidates:
        purpose = str(case.get("purpose") or "unknown")
        if fallback_needs[purpose] <= 0 or _case_input_identity(case) in attempted:
            continue
        row = dict(case)
        row["case_id"] = f"followup_{len(selected) + 1:03d}_{purpose}"
        row["followup_reason"] = "replaces_safe_fallback_without_reusing_attempted_regions"
        selected.append(row)
        fallback_needs[purpose] -= 1
    plan = {
        "schema_version": "1.0", "plan_version": "bounded_corpus_calibration_v1",
        "tool_version": __version__, "creation_timestamp": utc_now(),
        "execution_revision": CALIBRATION_EXECUTION_REVISION,
        "tier": candidate_plan.get("tier"), "excerpt_plan": candidate_plan.get("excerpt_plan"),
        "maximum_cases": sum(1 for row in prior_report.get("results", []) if not row.get("informative")),
        "case_count": len(selected),
        "planned_input_duration_seconds": round(sum(
            float(case[role]["duration"]) for case in selected for role in ("destination", "source")
        ), 3),
        "source_files_read_only": True, "resume_required": True,
        "followup_of": {
            "plan": str(prior_plan_path.resolve()), "report": str(prior_report_path.resolve()),
            "unfilled_fallback_purposes": dict(sorted((key, value) for key, value in fallback_needs.items() if value > 0)),
        },
        "cases": selected,
    }
    write_json(output_path, plan)
    return plan


def build_calibration_supplement_plan(
    *, candidate_plan_path: Path, prior_plan_paths: list[Path], output_path: Path,
    max_cases: int,
) -> dict[str, Any]:
    candidate_plan = read_json(candidate_plan_path)
    attempted = {
        _case_input_identity(case)
        for path in prior_plan_paths
        for case in read_json(path).get("cases", [])
    }
    selected = []
    ordered_candidates = sorted(
        candidate_plan.get("cases", []),
        key=lambda case: (-int(case.get("variant_index", 0) or 0), str(case.get("case_id"))),
    )
    for case in ordered_candidates:
        if _case_input_identity(case) in attempted:
            continue
        row = dict(case)
        row["case_id"] = f"supplement_{len(selected) + 1:03d}_{row['purpose']}"
        row["followup_reason"] = "adds_untried_case_after_recipe_refinement"
        selected.append(row)
        if len(selected) >= max(0, int(max_cases)):
            break
    plan = {
        "schema_version": "1.0", "plan_version": "bounded_corpus_calibration_v1",
        "tool_version": __version__, "creation_timestamp": utc_now(),
        "execution_revision": CALIBRATION_EXECUTION_REVISION,
        "tier": candidate_plan.get("tier"), "excerpt_plan": candidate_plan.get("excerpt_plan"),
        "maximum_cases": max(0, int(max_cases)), "case_count": len(selected),
        "planned_input_duration_seconds": round(sum(
            float(case[role]["duration"]) for case in selected for role in ("destination", "source")
        ), 3),
        "source_files_read_only": True, "resume_required": True,
        "supplement_of": [str(path.resolve()) for path in prior_plan_paths],
        "cases": selected,
    }
    write_json(output_path, plan)
    return plan


def execute_calibration_plan(
    *, root: Path, plan_path: Path, output_root: Path, base_config_path: Path,
    force: bool = False,
    execute_case: Callable[[Path, dict[str, Any], Path, Path], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    plan = read_json(plan_path)
    output_root.mkdir(parents=True, exist_ok=True)
    execute_case = execute_case or _execute_case
    results = []
    for case in plan.get("cases", []):
        case = dict(case)
        signature = str(case.get("case_signature") or stable_hash(case))
        case_root = output_root / f"{case['case_id']}_{signature[:8]}"
        result_path = case_root / "calibration_case_result.json"
        prior_result = read_json(result_path) if result_path.exists() else {}
        if not force and prior_result.get("case_signature") == signature and prior_result.get("status") == "completed":
            result = prior_result
            result["resumed"] = True
        else:
            case_root.mkdir(parents=True, exist_ok=True)
            try:
                result = execute_case(root, case, case_root, base_config_path)
            except Exception as exc:  # keep the bounded corpus moving and make failure inspectable
                result = {
                    "case_id": case["case_id"], "purpose": case["purpose"],
                    "status": "failed", "error_type": type(exc).__name__, "error": str(exc),
                }
            result["case_signature"] = signature
            write_json(result_path, result)
        results.append(result)
    report = _aggregate_results(plan=plan, plan_path=plan_path, results=results, output_root=output_root)
    write_json(output_root / "calibration_report.json", report)
    return report


def _execute_case(root: Path, case: dict[str, Any], case_root: Path, base_config_path: Path) -> dict[str, Any]:
    inputs = case_root / "inputs"
    destination_path = inputs / "destination.mp4"
    source_path = inputs / "source.mp4"
    _extract_excerpt(case["destination"], destination_path)
    _extract_excerpt(case["source"], source_path)

    config_data = read_json(base_config_path)
    config_data.update({
        "destination_video": str(destination_path.resolve()),
        "source_dialogue": str(source_path.resolve()),
        "cache_dir": str((case_root / "cache").resolve()),
        "output_dir": str((case_root / "output").resolve()),
        "temp_dir": str((case_root / "temp").resolve()),
        "transcription_mode": "quality",
        "whisper_model": "medium",
        "scheduling_mode": "performance_fill",
        "editorial_refinement_enabled": True,
    })
    if case.get("target_failure_category"):
        config_data["editorial_benchmark_failure_category"] = str(case["target_failure_category"])
    config_path = case_root / "config.json"
    write_json(config_path, config_data)
    config = load_config(root, config_path)
    transformation = Pipeline(config).execute_configuration("multiworld.translation", force=False)
    output_dir = config.output_dir
    editorial = read_json(output_dir / "editorial_report.json") if (output_dir / "editorial_report.json").exists() else {}
    effectiveness = read_json(output_dir / "repair_effectiveness.json") if (output_dir / "repair_effectiveness.json").exists() else {}
    problems = read_json(output_dir / "problem_regions.json") if (output_dir / "problem_regions.json").exists() else {}
    informative = bool(editorial)
    return {
        "case_id": case["case_id"],
        "purpose": case["purpose"],
        "status": "completed",
        "resumed": False,
        "informative": informative,
        "execution_outcome": "editorial_evidence_available" if informative else "safe_fallback_no_editorial_evidence",
        "destination_category": case["destination"]["category"],
        "source_category": case["source"]["category"],
        "expected_failure_modes": list(case.get("expected_failure_modes") or []),
        "target_failure_category": case.get("target_failure_category"),
        "target_repair_strategy": case.get("target_repair_strategy"),
        "output_video": str(Path(transformation.outputs["video"]).resolve()),
        "metrics": {
            "initial_quality": editorial.get("initial_quality"),
            "final_quality": editorial.get("final_quality"),
            "quality_improvement": editorial.get("quality_improvement"),
            "placements_repaired": int(editorial.get("placements_repaired") or 0),
            "placements_unresolved": int(editorial.get("placements_rejected") or 0),
            "rendered_candidate_count": int(effectiveness.get("rendered_candidate_count") or 0),
            "surviving_repair_count": int(effectiveness.get("surviving_repair_count") or 0),
            "candidate_survival_rate": float(effectiveness.get("candidate_survival_rate") or 0.0),
            "coordinated_candidate_count": int(effectiveness.get("coordinated_candidate_count") or 0),
            "coordinated_surviving_count": int(effectiveness.get("coordinated_surviving_count") or 0),
            "resumed_from_candidate_checkpoint": bool(editorial.get("resumed_from_candidate_checkpoint")),
            "problem_count": int(problems.get("problem_count") or 0),
        },
        "failure_categories": list(effectiveness.get("by_failure_category") or []),
        "repair_strategies": list(effectiveness.get("by_repair_strategy") or []),
    }


def _extract_excerpt(excerpt: dict[str, Any], output_path: Path) -> None:
    if _media_excerpt_valid(output_path):
        return
    source = Path(str(excerpt["source_path"]))
    if not source.is_file():
        raise FileNotFoundError(f"Calibration source does not exist: {source}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = output_path.with_name(f"{output_path.stem}.partial{output_path.suffix}")
    run([
        "ffmpeg", "-y", "-v", "error", "-ss", f"{float(excerpt['start']):.3f}",
        "-i", str(source), "-t", f"{float(excerpt['duration']):.3f}",
        "-map", "0:v:0", "-map", "0:a:0?", "-c:v", "libx264", "-preset", "veryfast",
        "-crf", "22", "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart",
        "-avoid_negative_ts", "make_zero", str(partial_path),
    ])
    if not _media_excerpt_valid(partial_path):
        raise RuntimeError(f"Calibration excerpt extraction produced invalid media: {partial_path}")
    partial_path.replace(output_path)


def _media_excerpt_valid(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size <= 0:
        return False
    try:
        probe = ffprobe_json(path)
    except (OSError, RuntimeError, ValueError):
        return False
    streams = list(probe.get("streams") or [])
    duration = float((probe.get("format") or {}).get("duration", 0.0) or 0.0)
    return duration > 0.0 and any(row.get("codec_type") == "video" for row in streams)


def _case_excerpt(excerpt: dict[str, Any]) -> dict[str, Any]:
    return {
        key: excerpt.get(key) for key in (
            "excerpt_id", "media_id", "filename", "source_path", "content_type", "category",
            "start", "end", "duration", "analysis_signature", "evidence",
        )
    }


def _case_input_identity(case: dict[str, Any]) -> str:
    return stable_hash({
        role: {
            key: dict(case.get(role) or {}).get(key)
            for key in ("source_path", "start", "end", "duration", "category")
        }
        for role in ("destination", "source")
    })


def _calibration_excerpt_eligible(excerpt: dict[str, Any], *, role: str) -> bool:
    if role == "destination" and excerpt.get("category") == "animation_dialogue":
        evidence = dict(excerpt.get("evidence") or {})
        if "performance_duration" not in evidence:
            return True
        performance_duration = float(evidence.get("performance_duration", 0.0) or 0.0)
        turn_count = int(evidence.get("turn_count", 0) or 0)
        speaker_count = int(evidence.get("speaker_count", 0) or 0)
        if performance_duration < 6.0 and (turn_count < 2 or speaker_count < 2):
            return False
    return True


def _aggregate_results(*, plan: dict[str, Any], plan_path: Path, results: list[dict[str, Any]], output_root: Path) -> dict[str, Any]:
    completed = [row for row in results if row.get("status") == "completed"]
    informative = [row for row in completed if row.get("informative")]
    failure_counts: Counter[str] = Counter()
    strategy_counts: Counter[str] = Counter()
    for result in completed:
        for row in result.get("failure_categories", []):
            failure_counts[str(row.get("name") or "unknown")] += int(row.get("attempt_count") or 0)
        for row in result.get("repair_strategies", []):
            strategy_counts[str(row.get("name") or "unknown")] += int(row.get("attempt_count") or 0)
    improvements = [float(row["metrics"]["quality_improvement"]) for row in completed if row.get("metrics", {}).get("quality_improvement") is not None]
    return {
        "schema_version": "1.0", "report_version": "bounded_corpus_calibration_report_v1",
        "tool_version": __version__, "creation_timestamp": utc_now(),
        "plan": str(plan_path.resolve()), "output_root": str(output_root.resolve()),
        "tier": plan.get("tier"), "case_count": len(results),
        "completed_case_count": len(completed), "failed_case_count": len(results) - len(completed),
        "informative_case_count": len(informative),
        "safe_fallback_case_count": len(completed) - len(informative),
        "average_quality_improvement": round(sum(improvements) / len(improvements), 4) if improvements else None,
        "failure_category_attempt_counts": dict(sorted(failure_counts.items())),
        "repair_strategy_attempt_counts": dict(sorted(strategy_counts.items())),
        "results": results,
    }
