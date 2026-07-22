from __future__ import annotations

from pathlib import Path

from cinelingus.corpus_calibration import (
    CALIBRATION_RECIPES, build_calibration_followup_plan, build_calibration_plan,
    build_calibration_supplement_plan, execute_calibration_plan, _extract_excerpt,
)
from cinelingus.util import write_json
from cinelingus.validation import validate_artifact


def _excerpt(index: int, category: str, media: str, duration: float = 12.0) -> dict:
    return {
        "excerpt_id": f"excerpt_{index:04d}", "media_id": media,
        "filename": f"{media}.mp4", "source_path": f"C:/movies/{media}.mp4",
        "content_type": "animation" if "animation" in category else "live_action",
        "category": category, "start": float(index * 10), "end": float(index * 10) + duration,
        "duration": duration, "analysis_signature": f"sig-{media}", "evidence": {},
    }


def test_calibration_plan_builds_bounded_cross_media_strategy_cases(tmp_path: Path) -> None:
    excerpt_plan = tmp_path / "excerpts.json"
    rows = [
        _excerpt(1, "rapid_speaker_exchange", "rapid"),
        _excerpt(2, "dense_dialogue", "dense"),
        _excerpt(3, "transition_near_dialogue", "transition"),
        _excerpt(4, "long_monologue", "monologue"),
        _excerpt(5, "animation_dialogue", "animation"),
        _excerpt(6, "short_fragmented_lines", "fragment"),
        _excerpt(7, "quiet_room_tone", "quiet"),
    ]
    write_json(excerpt_plan, {"tier": "smoke", "excerpts": rows})

    plan = build_calibration_plan(excerpt_plan_path=excerpt_plan, output_path=tmp_path / "plan.json")

    assert plan["case_count"] == 3
    assert plan["execution_revision"] == "rendered_failure_exploration_v1"
    assert [row["purpose"] for row in plan["cases"]] == [
        "exchange_continuity", "transition_sentence_integrity", "animation_visual_role",
    ]
    assert all(row["destination"]["source_path"] != row["source"]["source_path"] for row in plan["cases"])
    assert all(len(row["case_signature"]) == 64 for row in plan["cases"])
    assert plan["planned_input_duration_seconds"] == 72.0
    validate_artifact("corpus_calibration_plan", tmp_path / "plan.json", Path.cwd() / "schemas")


def test_extended_calibration_plan_adds_balanced_diverse_variants(tmp_path: Path) -> None:
    excerpt_plan = tmp_path / "excerpts.json"
    categories = [
        "rapid_speaker_exchange", "dense_dialogue", "transition_near_dialogue",
        "long_monologue", "animation_dialogue", "short_fragmented_lines", "quiet_room_tone",
    ]
    rows = []
    index = 1
    for variant in range(3):
        for category in categories:
            rows.append(_excerpt(index, category, f"{category}-{variant}"))
            index += 1
    write_json(excerpt_plan, {"tier": "extended", "excerpts": rows})

    plan = build_calibration_plan(excerpt_plan_path=excerpt_plan, output_path=tmp_path / "plan.json")

    assert plan["case_count"] == 10
    purposes = [row["purpose"] for row in plan["cases"]]
    assert purposes[:4] == [row["id"] for row in CALIBRATION_RECIPES]
    assert purposes.count("exchange_continuity") == 3
    assert purposes.count("transition_sentence_integrity") == 3
    assert purposes.count("animation_visual_role") == 2
    assert purposes.count("fragment_duration_edges") == 2
    for recipe in CALIBRATION_RECIPES:
        variants = [row for row in plan["cases"] if row["purpose"] == recipe["id"]]
        assert len({row["destination"]["media_id"] for row in variants}) == len(variants)


def test_calibration_plan_rejects_marginal_single_speaker_animation_excerpt(tmp_path: Path) -> None:
    excerpt_plan = tmp_path / "excerpts.json"
    rows = [
        {**_excerpt(1, "animation_dialogue", "weak-animation"), "evidence": {"performance_duration": 2.0, "turn_count": 1, "speaker_count": 1}},
        {**_excerpt(2, "animation_dialogue", "strong-animation"), "evidence": {"performance_duration": 12.0, "turn_count": 2, "speaker_count": 2}},
        _excerpt(3, "rapid_speaker_exchange", "rapid"),
        _excerpt(4, "dense_dialogue", "dense"),
        _excerpt(5, "transition_near_dialogue", "transition"),
        _excerpt(6, "long_monologue", "monologue"),
        _excerpt(7, "short_fragmented_lines", "fragment"),
    ]
    write_json(excerpt_plan, {"tier": "standard", "excerpts": rows})

    plan = build_calibration_plan(excerpt_plan_path=excerpt_plan, output_path=tmp_path / "plan.json")

    animation = next(row for row in plan["cases"] if row["purpose"] == "animation_visual_role")
    assert animation["destination"]["media_id"] == "strong-animation"


def test_followup_plan_replaces_only_fallback_purposes_with_untried_regions(tmp_path: Path) -> None:
    def case(case_id, purpose, destination_index, source_index):
        return {
            "case_id": case_id, "purpose": purpose, "case_signature": f"sig-{case_id}",
            "destination": _excerpt(destination_index, "rapid_speaker_exchange", f"d-{destination_index}"),
            "source": _excerpt(source_index, "dense_dialogue", f"s-{source_index}"),
            "expected_failure_modes": [], "status": "planned",
        }

    prior_plan = tmp_path / "prior-plan.json"
    candidate_plan = tmp_path / "candidate-plan.json"
    prior_report = tmp_path / "prior-report.json"
    write_json(prior_plan, {"cases": [case("old-a", "exchange", 1, 2), case("old-b", "quiet", 3, 4)]})
    write_json(candidate_plan, {
        "tier": "extended", "excerpt_plan": "excerpts.json", "cases": [
            case("duplicate", "quiet", 3, 4), case("new-quiet", "quiet", 5, 6),
            case("new-exchange", "exchange", 7, 8),
        ],
    })
    write_json(prior_report, {"results": [
        {"case_id": "old-a", "purpose": "exchange", "status": "completed", "informative": True},
        {"case_id": "old-b", "purpose": "quiet", "status": "completed", "informative": False},
    ]})

    followup = build_calibration_followup_plan(
        candidate_plan_path=candidate_plan, prior_plan_path=prior_plan,
        prior_report_path=prior_report, output_path=tmp_path / "followup.json",
    )

    assert followup["case_count"] == 1
    assert followup["cases"][0]["purpose"] == "quiet"
    assert followup["cases"][0]["destination"]["media_id"] == "d-5"


def test_supplement_plan_selects_only_untried_candidate_inputs(tmp_path: Path) -> None:
    def case(case_id, destination_index, source_index):
        return {
            "case_id": case_id, "purpose": "exchange", "case_signature": f"sig-{case_id}",
            "destination": _excerpt(destination_index, "rapid_speaker_exchange", f"d-{destination_index}"),
            "source": _excerpt(source_index, "dense_dialogue", f"s-{source_index}"),
            "expected_failure_modes": [], "status": "planned",
        }

    prior = tmp_path / "prior.json"
    candidates = tmp_path / "candidates.json"
    write_json(prior, {"cases": [case("attempted", 1, 2)]})
    write_json(candidates, {
        "tier": "extended", "excerpt_plan": "excerpts.json",
        "cases": [case("duplicate", 1, 2), case("new-a", 3, 4), case("new-b", 5, 6)],
    })

    supplement = build_calibration_supplement_plan(
        candidate_plan_path=candidates, prior_plan_paths=[prior],
        output_path=tmp_path / "supplement.json", max_cases=1,
    )

    assert supplement["case_count"] == 1
    assert supplement["cases"][0]["destination"]["media_id"] == "d-3"


def test_calibration_execution_resumes_completed_cases_and_aggregates(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.json"
    write_json(plan_path, {
        "tier": "smoke", "cases": [
            {"case_id": "case_001", "purpose": "exchange"},
            {"case_id": "case_002", "purpose": "transition"},
        ],
    })
    calls = []

    def fake_execute(_root, case, _case_root, _config):
        calls.append(case["case_id"])
        return {
            "case_id": case["case_id"], "purpose": case["purpose"], "status": "completed",
            "metrics": {"quality_improvement": 0.1},
            "failure_categories": [{"name": "performance_mismatch", "attempt_count": 2}],
            "repair_strategies": [{"name": "repair_performance_structure", "attempt_count": 1}],
        }

    output = tmp_path / "output"
    first = execute_calibration_plan(
        root=tmp_path, plan_path=plan_path, output_root=output,
        base_config_path=tmp_path / "default.json", execute_case=fake_execute,
    )
    second = execute_calibration_plan(
        root=tmp_path, plan_path=plan_path, output_root=output,
        base_config_path=tmp_path / "default.json", execute_case=fake_execute,
    )

    assert calls == ["case_001", "case_002"]
    assert first["completed_case_count"] == 2
    assert first["failure_category_attempt_counts"] == {"performance_mismatch": 4}
    assert second["results"][0]["resumed"] is True


def test_calibration_execution_retries_failed_case(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.json"
    write_json(plan_path, {"tier": "smoke", "cases": [{"case_id": "case_001", "purpose": "exchange"}]})
    calls = []

    def fail_then_complete(_root, case, _case_root, _config):
        calls.append(case["case_id"])
        if len(calls) == 1:
            raise RuntimeError("transient failure")
        return {"case_id": case["case_id"], "purpose": case["purpose"], "status": "completed", "metrics": {}}

    kwargs = {
        "root": tmp_path, "plan_path": plan_path, "output_root": tmp_path / "output",
        "base_config_path": tmp_path / "default.json", "execute_case": fail_then_complete,
    }
    first = execute_calibration_plan(**kwargs)
    second = execute_calibration_plan(**kwargs)

    assert first["failed_case_count"] == 1
    assert second["completed_case_count"] == 1
    assert calls == ["case_001", "case_001"]


def test_excerpt_extraction_replaces_interrupted_invalid_output_atomically(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    output = tmp_path / "excerpt.mp4"
    output.write_bytes(b"truncated")

    def valid(path: Path) -> bool:
        return path.read_bytes() == b"complete" if path.exists() else False

    def fake_run(args):
        Path(args[-1]).write_bytes(b"complete")

    monkeypatch.setattr("cinelingus.corpus_calibration._media_excerpt_valid", valid)
    monkeypatch.setattr("cinelingus.corpus_calibration.run", fake_run)

    _extract_excerpt(
        {"source_path": str(source), "start": 1.0, "duration": 2.0}, output
    )

    assert output.read_bytes() == b"complete"
    assert not (tmp_path / "excerpt.partial.mp4").exists()
