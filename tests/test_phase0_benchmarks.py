from __future__ import annotations

from pathlib import Path

from cinelingus.editorial.repair_strategies import STRATEGIES
from cinelingus.phase0_benchmarks import (
    build_observed_failure_strategy_plan, build_rendered_strategy_coverage, build_strategy_isolation_plan,
    run_strategy_contract_benchmarks,
)
from cinelingus.util import write_json
from cinelingus.validation import validate_artifact


def test_every_declared_strategy_has_an_executable_contract_benchmark(tmp_path: Path) -> None:
    path = tmp_path / "contracts.json"
    report = run_strategy_contract_benchmarks(output_path=path)

    assert report["case_count"] == len(STRATEGIES) == 12
    assert report["passed_case_count"] == 12
    assert report["failed_case_count"] == 0
    assert report["all_declared_strategies_executable"] is True
    assert report["rendered_evidence_claimed"] is False
    assert {row["failure_category"] for row in report["cases"]} == set(STRATEGIES)
    validate_artifact("phase0_strategy_benchmarks", path, Path.cwd() / "schemas")


def test_rendered_coverage_keeps_contract_and_corpus_evidence_separate(tmp_path: Path) -> None:
    contract_path = tmp_path / "contracts.json"
    run_strategy_contract_benchmarks(output_path=contract_path)
    calibration_path = tmp_path / "calibration.json"
    write_json(calibration_path, {
        "results": [{
            "informative": True,
            "repair_strategies": [
                {"name": "repair_sentence_boundaries", "attempt_count": 3, "rendered_count": 2, "survived_count": 1},
                {"name": "repair_audio_masking", "attempt_count": 1, "rendered_count": 0, "survived_count": 0},
            ],
        }],
    })
    output_path = tmp_path / "coverage.json"

    report = build_rendered_strategy_coverage(
        contract_report_path=contract_path,
        calibration_report_paths=[calibration_path], output_path=output_path,
    )

    assert report["contract_executable_count"] == 12
    assert report["runtime_attempted_strategy_count"] == 2
    assert report["rendered_strategy_count"] == 1
    assert report["surviving_strategy_count"] == 1
    assert report["evidence_satisfied_strategy_count"] == 1
    assert report["phase0_rendered_strategy_gate_passed"] is False
    assert "repair_audio_masking" in report["missing_rendered_strategies"]
    assert "conservative_uncertainty_retention" in report["missing_strategy_evidence"]
    validate_artifact("phase0_strategy_coverage", output_path, Path.cwd() / "schemas")


def test_strategy_isolation_plan_promotes_observed_secondary_failures(tmp_path: Path) -> None:
    prior_plan = tmp_path / "prior-plan.json"
    prior_report = tmp_path / "prior-report.json"
    coverage = tmp_path / "coverage.json"
    destination = {
        "source_path": "C:/movies/a.mp4", "duration": 12.0, "category": "rapid_speaker_exchange",
    }
    source = {"source_path": "C:/movies/b.mp4", "duration": 12.0, "category": "dense_dialogue"}
    write_json(prior_plan, {
        "tier": "extended", "excerpt_plan": "excerpts.json", "cases": [{
            "case_id": "case-1", "case_signature": "original", "purpose": "exchange",
            "destination": destination, "source": source, "expected_failure_modes": [], "status": "planned",
        }],
    })
    write_json(prior_report, {"results": [{
        "case_id": "case-1", "informative": True,
        "failure_categories": [
            {"name": "incomplete_sentence", "attempt_count": 1},
            {"name": "visual_mismatch", "attempt_count": 1},
        ],
    }]})
    write_json(coverage, {
        "missing_strategy_evidence": ["repair_visual_intent", "repair_audio_masking"],
    })
    output = tmp_path / "isolation.json"

    plan = build_strategy_isolation_plan(
        prior_plan_path=prior_plan, prior_report_path=prior_report,
        coverage_report_path=coverage, output_path=output,
    )

    assert plan["case_count"] == 1
    assert plan["cases"][0]["target_failure_category"] == "visual_mismatch"
    assert plan["cases"][0]["target_repair_strategy"] == "repair_visual_intent"
    assert plan["unfilled_target_strategies"] == ["repair_audio_masking"]
    validate_artifact("corpus_calibration_plan", output, Path.cwd() / "schemas")


def test_observed_failure_plan_extracts_bounded_provenance_neighborhoods(tmp_path: Path) -> None:
    config = tmp_path / "config.json"
    editorial = tmp_path / "editorial.json"
    schedule = tmp_path / "schedule.json"
    coverage = tmp_path / "coverage.json"
    write_json(config, {"destination_video": "movies/destination.mp4", "source_dialogue": "movies/source.mp4"})
    write_json(editorial, {"decisions": [{
        "placement_key": "placement-1", "destination_start": 20.0, "destination_end": 22.0,
        "overall_quality": 0.3, "failures": [{"category": "speaker_mismatch"}],
    }]})
    write_json(schedule, {"mappings": [{
        "editorial_placement_id": "placement-1", "source_movie_timestamp": 50.0,
        "clip_trim_duration": 2.0,
    }]})
    write_json(coverage, {"missing_strategy_evidence": ["repair_speaker_role", "repair_reuse_pressure"]})
    output = tmp_path / "plan.json"

    plan = build_observed_failure_strategy_plan(
        config_path=config, editorial_report_path=editorial, schedule_path=schedule,
        coverage_report_path=coverage, output_path=output,
    )

    assert plan["case_count"] == 1
    case = plan["cases"][0]
    assert case["target_failure_category"] == "speaker_mismatch"
    assert case["destination"]["start"] == 10.0
    assert case["source"]["start"] == 35.0
    assert case["source"]["duration"] == 32.0
    assert case["observed_source_placement_key"] == "placement-1"
    assert plan["unfilled_target_strategies"] == ["repair_reuse_pressure"]
    validate_artifact("corpus_calibration_plan", output, Path.cwd() / "schemas")


def test_observed_failure_plan_can_replenish_multiple_variants(tmp_path: Path) -> None:
    config = tmp_path / "config.json"
    editorial = tmp_path / "editorial.json"
    schedule = tmp_path / "schedule.json"
    coverage = tmp_path / "coverage.json"
    write_json(config, {"destination_video": "d.mp4", "source_dialogue": "s.mp4"})
    write_json(editorial, {"decisions": [
        {"placement_key": "p1", "destination_start": 20.0, "destination_end": 22.0, "overall_quality": 0.2, "failures": [{"category": "duration_failure"}]},
        {"placement_key": "p2", "destination_start": 40.0, "destination_end": 42.0, "overall_quality": 0.4, "failures": [{"category": "duration_failure"}]},
    ]})
    write_json(schedule, {"mappings": [
        {"editorial_placement_id": "p1", "source_movie_timestamp": 50.0, "clip_trim_duration": 2.0},
        {"editorial_placement_id": "p2", "source_movie_timestamp": 80.0, "clip_trim_duration": 2.0},
    ]})
    write_json(coverage, {"missing_strategy_evidence": ["repair_duration_fit"]})

    plan = build_observed_failure_strategy_plan(
        config_path=config, editorial_report_path=editorial, schedule_path=schedule,
        coverage_report_path=coverage, output_path=tmp_path / "plan.json",
        variants_per_strategy=2,
    )

    assert plan["case_count"] == 2
    assert [row["variant_index"] for row in plan["cases"]] == [1, 2]
    assert [row["observed_source_placement_key"] for row in plan["cases"]] == ["p1", "p2"]
