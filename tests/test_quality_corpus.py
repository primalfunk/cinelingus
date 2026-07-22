from pathlib import Path

import pytest

from cinelingus.quality_corpus import evaluate_quality_corpus
from cinelingus.util import write_json


def _run_report(*, residue: str = "NONE_DETECTED", silence: int = 0, problems: int = 1) -> dict:
    return {
        "schedule": {
            "performance_summary": {
                "destination_performance_count": 10,
                "performance_couplings": 7,
                "adapted_performances": 1,
                "turn_sequence_matches": 1,
                "linewise_fallbacks": 1,
                "preserved_original_regions": 0,
            },
            "voice_residue_verification": {"status": residue},
        },
        "soundtrack_bed": {
            "reconstructed_region_count": 9,
            "silence_fallback_region_count": silence,
        },
        "problem_region_report": {"problem_count": problems},
    }


def _editorial_report(*, hard: bool = False) -> dict:
    failure = {
        "category": "low_rendered_coverage", "severity": "critical",
        "evidence": {"coverage": 0.1},
    }
    return {
        "status": "LIMIT_REACHED" if hard else "PASS",
        "final_quality": 0.99,
        "final_state_counts": {"BEST_KNOWN_UNRESOLVED": int(hard), "ACCEPTED": int(not hard)},
        "decisions": [{
            "placement_key": "placement-1", "overall_quality": 0.99,
            "hard_gate_passed": not hard,
            "hard_gate_failures": ["low_rendered_coverage"] if hard else [],
            "failures": [failure] if hard else [],
        }],
    }


def test_quality_corpus_evaluates_completed_runs_against_thresholds(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    write_json(runs / "clean" / "run_report.json", _run_report())
    write_json(runs / "ghost" / "run_report.json", _run_report(residue="POSSIBLE_DESTINATION_SPEECH_DETECTED"))
    write_json(runs / "clean" / "editorial_report.json", _editorial_report())
    write_json(runs / "ghost" / "editorial_report.json", _editorial_report())
    manifest = tmp_path / "corpus.json"
    write_json(manifest, {
        "cases": [
            {"id": "clean", "run_report": "clean/run_report.json"},
            {"id": "ghost", "run_report": "ghost/run_report.json"},
        ]
    })

    report = evaluate_quality_corpus(
        manifest_path=manifest, runs_root=runs, output_path=tmp_path / "quality.json",
    )

    assert report["passed"] is False
    assert report["passed_case_count"] == 1
    assert report["failed_case_count"] == 1
    assert report["cases"][1]["metrics"]["residue_status"] == "POSSIBLE_DESTINATION_SPEECH_DETECTED"


def test_quality_corpus_hard_failure_cannot_hide_behind_high_average(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    write_json(runs / "hard" / "run_report.json", _run_report())
    write_json(runs / "hard" / "editorial_report.json", _editorial_report(hard=True))
    manifest = tmp_path / "corpus.json"
    write_json(manifest, {"cases": [{"id": "hard", "run_report": "hard/run_report.json"}]})

    report = evaluate_quality_corpus(
        manifest_path=manifest, runs_root=runs, output_path=tmp_path / "quality.json",
    )

    case = report["cases"][0]
    assert case["metrics"]["editorial_final_quality"] == 0.99
    assert case["metrics"]["hard_gate_failure_count"] == 1
    assert case["passed"] is False
    assert next(row for row in case["checks"] if row["name"] == "hard_gate_failure_count")["passed"] is False


def test_quality_corpus_requires_editorial_evidence_by_default(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    write_json(runs / "missing" / "run_report.json", _run_report())
    manifest = tmp_path / "corpus.json"
    write_json(manifest, {"cases": [{"id": "missing", "run_report": "missing/run_report.json"}]})

    report = evaluate_quality_corpus(
        manifest_path=manifest, runs_root=runs, output_path=tmp_path / "quality.json",
    )

    case = report["cases"][0]
    assert case["metrics"]["editorial_evidence_available"] is False
    assert case["passed"] is False


def test_quality_corpus_rejects_paths_outside_runs_root(tmp_path: Path) -> None:
    manifest = tmp_path / "corpus.json"
    write_json(manifest, {"cases": [{"id": "escape", "run_report": "../run_report.json"}]})

    with pytest.raises(ValueError, match="escapes runs root"):
        evaluate_quality_corpus(
            manifest_path=manifest, runs_root=tmp_path / "runs", output_path=tmp_path / "quality.json",
        )
