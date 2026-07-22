from pathlib import Path
import wave

import pytest

from cinelingus.filter_lab.acceptance import _classify_silent_intervals
from cinelingus.montage import CapabilityTag, annotate_moment_audio_activity, annotate_windows_with_montage_eligibility, build_core_moments, build_full_timeline_plan, build_montage_plan, build_montage_render_acceptance, build_naive_shot_moments, build_placement_qualification, moment_artifact_with_authored_submoments, moments_with_schedule_coverage, rebase_schedule_to_montage, stable_moment_id
from cinelingus.montage_evaluation import build_montage_evaluation, build_source_start_bias_check
from cinelingus.pipeline import Pipeline, _require_montage_native_self_shuffle_result
from cinelingus.transformations import TransformationResult
from cinelingus.util import read_json, write_json
from cinelingus.validation import SCHEMA_MAP, validate_artifact


SCHEMAS = Path.cwd() / "schemas"


def _shot(identifier: str, start: float, end: float, scene: str | None = None) -> dict:
    row = {"id": identifier, "start": start, "end": end, "duration": end - start}
    if scene:
        row["scene_id"] = scene
    return row


def test_stable_moment_id_depends_on_media_boundaries_and_ordered_shots() -> None:
    first = stable_moment_id(source_media_hash="abc", start=1.0001, end=4.0, shot_ids=["s1", "s2"])
    repeated = stable_moment_id(source_media_hash="abc", start=1.0001, end=4.0, shot_ids=["s1", "s2"])
    changed = stable_moment_id(source_media_hash="abc", start=1.0001, end=4.0, shot_ids=["s2", "s1"])

    assert first == repeated
    assert first.startswith("moment_")
    assert changed != first


def test_core_groups_consecutive_shots_from_one_declared_scene() -> None:
    artifact = build_core_moments(
        source_id="film_a",
        source_media_hash="hash_a",
        shots=[_shot("s1", 0.0, 3.0, "scene_1"), _shot("s2", 3.0, 8.0, "scene_1")],
    )

    assert artifact["moment_count"] == 1
    moment = artifact["moments"][0]
    assert moment["shot_ids"] == ["s1", "s2"]
    assert moment["start"] == 0.0
    assert moment["end"] == 8.0


def test_core_preserves_shots_across_a_speech_or_transition_boundary() -> None:
    shots = [_shot("s1", 0.0, 3.0), _shot("s2", 3.0, 6.0), _shot("s3", 6.0, 9.0)]
    artifact = build_core_moments(
        source_id="film_a",
        source_media_hash="hash_a",
        shots=shots,
        speech_intervals=[{"start": 2.5, "end": 3.5}],
        transition_intervals=[{"start": 5.8, "end": 6.2, "kind": "dissolve"}],
    )

    assert artifact["moment_count"] == 1
    assert artifact["moments"][0]["shot_ids"] == ["s1", "s2", "s3"]


def test_core_never_invents_an_internal_long_take_boundary() -> None:
    artifact = build_core_moments(
        source_id="film_a",
        source_media_hash="hash_a",
        shots=[_shot("long_take", 10.0, 45.0)],
    )

    moment = artifact["moments"][0]
    assert moment["start"] == 10.0
    assert moment["end"] == 45.0
    assert moment["fallback_status"] == "PRESERVED_COMPLETE_LONG_TAKE"
    fallback = next(row for row in moment["assertions"] if row["name"] == "NO_SAFE_INTERNAL_BOUNDARY_FOUND")
    assert fallback["capability_tag"] == CapabilityTag.FALLBACK_INFERENCE.value
    assert fallback["fallback"] is True


def test_core_subdivides_long_take_only_with_overlapping_silence_and_stillness() -> None:
    artifact = build_core_moments(
        source_id="film_a",
        source_media_hash="hash_a",
        shots=[_shot("long_take", 0.0, 30.0)],
        speech_intervals=[{"start": 0.0, "end": 13.0}, {"start": 17.0, "end": 30.0}],
        stillness_intervals=[{"start": 14.0, "end": 16.0, "kind": "SUSTAINED_LOW_FRAME_DIFFERENCE"}],
    )

    assert artifact["moment_count"] == 2
    assert [row["shot_ids"] for row in artifact["moments"]] == [["long_take"], ["long_take"]]
    assert artifact["moments"][0]["end"] == artifact["moments"][1]["start"] == 15.0
    names = {row["name"] for moment in artifact["moments"] for row in moment["assertions"]}
    assert {"SILENCE_WINDOW_PRESENT", "LOW_FRAME_DIFFERENCE_AT_BOUNDARY", "VIRTUAL_BOUNDARY_CORE_SUPPORTED"} <= names


def test_core_does_not_subdivide_for_silence_without_stillness_or_inside_transition() -> None:
    kwargs = {
        "source_id": "film_a",
        "source_media_hash": "hash_a",
        "shots": [_shot("long_take", 0.0, 30.0)],
        "speech_intervals": [{"start": 0.0, "end": 13.0}, {"start": 17.0, "end": 30.0}],
    }
    silence_only = build_core_moments(**kwargs)
    guarded = build_core_moments(
        **kwargs,
        stillness_intervals=[{"start": 14.0, "end": 16.0}],
        transition_intervals=[{"start": 13.5, "end": 16.5, "kind": "GRADUAL_TRANSITION_CANDIDATE"}],
    )

    assert silence_only["moment_count"] == 1
    assert guarded["moment_count"] == 1


def test_core_groups_across_a_high_motion_boundary_without_claiming_action_understanding() -> None:
    artifact = build_core_moments(
        source_id="film_a",
        source_media_hash="hash_a",
        shots=[_shot("s1", 0.0, 3.0), _shot("s2", 3.0, 6.0)],
        boundary_stability=[{
            "boundary": 3.0,
            "status": "AVAILABLE",
            "low_boundary_motion": False,
            "capability_tag": "CORE_HEURISTIC",
            "evidence_name": "LOW_FRAME_DIFFERENCE_AT_BOUNDARY",
            "confidence": 0.9,
        }],
    )

    assert artifact["moment_count"] == 1
    names = {row["name"] for row in artifact["moments"][0]["assertions"]}
    assert "COMPLETE_ACTION" not in names


def test_core_artifact_uses_literal_evidence_language_and_validates(tmp_path: Path) -> None:
    artifact = build_core_moments(
        source_id="film_a",
        source_media_hash="hash_a",
        shots=[_shot("s1", 0.0, 4.0)],
    )
    names = {assertion["name"] for moment in artifact["moments"] for assertion in moment["assertions"]}

    assert "COMPLETE_ACTION" not in names
    assert "COMPLETE_GESTURE" not in names
    assert "SHOT_SEQUENCE_PRESERVED" in names

    path = tmp_path / "cinematic_moments.json"
    write_json(path, artifact)
    assert validate_artifact("cinematic_moments", path, SCHEMAS)["moment_count"] == 1


def test_new_montage_schemas_are_registered_and_parseable() -> None:
    for artifact_type in (
        "cinematic_moments",
        "montage_plan",
        "montage_evaluation",
        "montage_calibration_manifest",
        "montage_render_acceptance",
    ):
        path = SCHEMAS / SCHEMA_MAP[artifact_type]
        assert read_json(path)["type"] == "object"


def test_repository_adr_indexes_all_approved_authority_documents() -> None:
    adr = (Path.cwd() / "docs" / "architecture" / "adr-001-montage-composition-foundation.md").read_text(encoding="utf-8")

    assert "../montage_I_design.txt" in adr
    assert "../montage_II_foundation.txt" in adr
    assert "../montage_III_addendum.txt" in adr
    assert "EXPERIMENTAL" in adr


def test_corpus_resolver_example_contains_only_portable_placeholders() -> None:
    example = read_json(Path.cwd() / "config" / "montage_corpus_resolver.example.json")

    assert example["schema_version"] == "1.0"
    assert all("Reference Film" in path for path in example["sources"].values())


def test_naive_sampler_preserves_each_complete_shot_without_safety_claims() -> None:
    artifact = build_naive_shot_moments(
        source_id="film_a",
        source_media_hash="hash_a",
        shots=[_shot("s1", 0.0, 3.0), _shot("s2", 3.0, 8.0)],
    )

    assert artifact["moment_count"] == 2
    assert [row["shot_ids"] for row in artifact["moments"]] == [["s1"], ["s2"]]
    assert artifact["moments"][0]["assertions"][0]["name"] == "NAIVE_COMPLETE_SHOT"


def test_montage_plan_is_deterministic_structured_and_experimental(tmp_path: Path) -> None:
    artifact = build_core_moments(
        source_id="film_a",
        source_media_hash="hash_a",
        shots=[_shot(f"s{index}", index * 4.0, (index + 1) * 4.0) for index in range(12)],
    )
    laws = {
        "visual": "broad_sampling",
        "temporal": "seeded_reordering",
        "dialogue": "unchanged",
        "requested_audio": "ORIGINAL_REALITY",
        "actual_audio_method": "ORIGINAL_REALITY",
    }

    first = build_montage_plan(
        filter_id="translation.self_shuffle",
        filter_contract_version="1.0.0",
        moment_artifacts=[artifact],
        target_duration=45.0,
        minimum_moments=4,
        random_seed=17,
        governing_relationship="repetition",
        laws=laws,
    )
    repeated = build_montage_plan(
        filter_id="translation.self_shuffle",
        filter_contract_version="1.0.0",
        moment_artifacts=[artifact],
        target_duration=45.0,
        minimum_moments=4,
        random_seed=17,
        governing_relationship="repetition",
        laws=laws,
    )

    assert first == repeated
    assert first["verdict"] == "EXPERIMENTAL"
    assert first["structural_roles"][0] == "BEGINNING"
    assert first["structural_roles"][-2:] == ["CLIMAX", "RESOLUTION"]
    assert first["source_participation"]["film_a"]["share"] == 1.0
    path = tmp_path / "montage_plan.json"
    write_json(path, first)
    assert validate_artifact("montage_plan", path, SCHEMAS)["planner_version"] == "montage_planner_v10"
    assert first["opening_selection"]["timeline_position_primary_tiebreaker"] is False
    assert "SEEDED_DIVERSITY" in first["opening_selection"]["selection_basis"]


def test_montage_planner_packs_complete_moments_within_target_when_possible() -> None:
    artifact = build_core_moments(
        source_id="film_a", source_media_hash="hash_a",
        shots=[_shot(f"s{index}", index * 5.0, (index + 1) * 5.0) for index in range(20)],
    )
    plan = build_montage_plan(
        filter_id="translation.self_shuffle", filter_contract_version="2.0",
        moment_artifacts=[artifact], target_duration=40.0, minimum_moments=8,
        random_seed=11, governing_relationship="self_recollection",
        laws={"visual": "safe", "temporal": "seeded", "dialogue": "reassigned", "requested_audio": "ambient", "actual_audio_method": "minimal"},
    )

    assert len(plan["selected_moments"]) == 8
    assert plan["actual_duration"] == 40.0
    assert not plan["fallback_decisions"]


def test_montage_audio_qualification_rejects_sustained_source_dead_air(tmp_path: Path) -> None:
    artifact = build_core_moments(
        source_id="film_a", source_media_hash="hash_a",
        shots=[_shot("s1", 0.0, 1.0), _shot("s2", 1.0, 2.0)],
    )
    audio = tmp_path / "analysis.wav"
    with wave.open(str(audio), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(48000)
        active = (3000).to_bytes(2, "little", signed=True)
        silent = (0).to_bytes(2, "little", signed=True)
        handle.writeframes(active * 48000 + silent * 48000)
    qualified = annotate_moment_audio_activity(artifact, audio_path=audio)
    schedule = {"mappings": [
        {"enabled": True, "destination_timestamp": 0.1, "planned_render_duration": 0.5},
        {"enabled": True, "destination_timestamp": 1.1, "planned_render_duration": 0.5},
    ]}

    eligible = moments_with_schedule_coverage(qualified, schedule)

    assert eligible["moment_count"] == 1
    assert eligible["moments"][0]["audio_activity"]["eligible"] is True
    assert eligible["candidate_rejections"][0]["reason"] == "SOURCE_SOUNDTRACK_SUSTAINED_DEAD_AIR"
    path = tmp_path / "cinematic_moments.json"
    write_json(path, qualified)
    validate_artifact("cinematic_moments", path, SCHEMAS)


def test_red_dwarf_regression_rescues_local_complete_shot_without_relaxing_final_threshold(tmp_path: Path) -> None:
    shots = [
        _shot("dialogue_shot", 0.0, 4.0, "scene_1"),
        _shot("silent_shot", 4.0, 8.0, "scene_1"),
        _shot("tail_shot", 8.0, 12.0, "scene_1"),
    ]
    artifact = build_core_moments(
        source_id="red_dwarf_fixture",
        source_media_hash="red_dwarf_hash",
        shots=shots,
    )
    audio = tmp_path / "red_dwarf_regression.wav"
    with wave.open(str(audio), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(48000)
        active = (3000).to_bytes(2, "little", signed=True)
        silent = (0).to_bytes(2, "little", signed=True)
        handle.writeframes(active * (4 * 48000) + silent * (8 * 48000))
    qualified_moments = annotate_moment_audio_activity(artifact, audio_path=audio)
    windows = [
        {"id": "legal_dialogue", "start": 1.0, "end": 3.0, "duration": 2.0},
        {"id": "unsafe_dialogue", "start": 5.0, "end": 7.0, "duration": 2.0},
    ]

    qualification = build_placement_qualification(
        windows,
        qualified_moments,
        shots_artifact={"shots": shots, "transitions": []},
        audio_path=audio,
    )

    legal = [row for row in qualification["windows"] if row["montage_placement_eligible"]]
    assert qualified_moments["moments"][0]["audio_activity"]["eligible"] is False
    assert len(legal) == 1
    assert legal[0]["montage_qualification_stage"] == "LOCAL_COMPLETE_SHOT"
    assert qualification["submoments"][0]["shot_ids"] == ["dialogue_shot"]
    assert qualification["report"]["preferred_legal_placements"] == 0
    assert qualification["report"]["local_legal_placements"] == 1
    assert qualification["report"]["locally_rejected_submoments"] == 1
    assert qualification["report"]["deterministic_rescue_invoked"] is True
    assert qualification["report"]["final_output_dead_air_limit_seconds"] == 0.75

    schedule = {
        "mappings": [{
            "enabled": True,
            "clip_id": "line_1",
            "destination_timestamp": legal[0]["start"],
            "planned_render_duration": legal[0]["duration"],
            "montage_moment_id": legal[0]["montage_moment_id"],
        }],
        "placement_qualification": qualification["report"],
    }
    planning_artifact = moment_artifact_with_authored_submoments(
        qualified_moments,
        qualification,
        schedule,
    )
    eligible = moments_with_schedule_coverage(
        planning_artifact,
        schedule,
        include_audio_safe_context=True,
    )
    kwargs = {
        "filter_id": "multiworld.contagion",
        "filter_contract_version": "2.0",
        "moment_artifacts": [eligible],
        "target_duration": 4.0,
        "minimum_moments": 1,
        "random_seed": 11,
        "governing_relationship": "contagion",
        "laws": {
            "visual": "complete_shots",
            "temporal": "seeded",
            "dialogue": "contagious",
            "requested_audio": "source",
            "actual_audio_method": "source_bed",
        },
        "schedule": schedule,
    }
    first = build_montage_plan(**kwargs)
    second = build_montage_plan(**kwargs)
    assert first == second
    assert first["selected_moments"][0]["shot_ids"] == ["dialogue_shot"]
    assert first["placement_qualification"]["deterministic_rescue_invoked"] is True


def test_identical_media_timeline_reuses_one_canonical_transcription(tmp_path: Path) -> None:
    pipeline = object.__new__(Pipeline)
    destination_dir = tmp_path / "destination"
    source_dir = tmp_path / "source"
    destination_dir.mkdir()
    source_dir.mkdir()
    entry_type = type("Entry", (), {})
    pipeline.destination = entry_type()
    pipeline.destination.media_hash = "same_hash"
    pipeline.destination.cache_dir = destination_dir
    pipeline.destination.manifest_path = destination_dir / "manifest.json"
    pipeline.destination.role = "destination_video"
    pipeline.destination.media_path = tmp_path / "same.mp4"
    pipeline.source = entry_type()
    pipeline.source.media_hash = "same_hash"
    pipeline.source.cache_dir = source_dir
    pipeline.source.manifest_path = source_dir / "manifest.json"
    pipeline.source.role = "source_dialogue"
    pipeline.source.media_path = pipeline.destination.media_path
    write_json(pipeline.destination.manifest_path, {"schema_version": "1.0", "media_hash": "same_hash", "artifacts": {}})
    write_json(pipeline.source.manifest_path, {"schema_version": "1.0", "media_hash": "same_hash", "artifacts": {}})
    pipeline.schemas_dir = SCHEMAS
    pipeline.cancel_check = None
    messages = []
    pipeline.logger = type("Logger", (), {"info": lambda self, message: messages.append(message)})()
    source_events = {
        "schema_version": "1.0",
        "tool_version": "test",
        "media_hash": "same_hash",
        "creation_timestamp": "2026-07-15T00:00:00+00:00",
        "detector": "fixture",
        "config_signature": "canonical_source_signature",
        "events": [{"id": "line_1", "start": 1.0, "end": 2.0, "duration": 1.0, "transcript": "hello", "confidence": 1.0}],
    }

    first = pipeline.build_identical_media_timeline_from_source(source_events=source_events)
    repeated = pipeline.build_identical_media_timeline_from_source(source_events=source_events)

    assert first["windows"] == source_events["events"]
    assert first["canonical_analysis_reuse"]["enabled"] is True
    assert repeated == first
    assert any("reused canonical identical-media transcription" in message for message in messages)


def test_silent_interval_provenance_distinguishes_source_behavior_seams_and_gaps() -> None:
    schedule = {
        "montage_audio_segments": [
            {"moment_id": "moment_a", "output_start": 0.0, "output_end": 5.0},
            {"moment_id": "moment_b", "output_start": 5.2, "output_end": 10.0},
        ]
    }
    intervals = [
        {"start": 1.0, "end": 1.4, "duration": 0.4},
        {"start": 4.9, "end": 5.3, "duration": 0.4},
        {"start": 10.2, "end": 11.0, "duration": 0.8},
    ]

    classified = _classify_silent_intervals(intervals, schedule)

    assert [row["classification"] for row in classified] == [
        "SOURCE_AUTHORED_AUDIO_BEHAVIOR",
        "ASSEMBLY_SEAM",
        "UNMAPPED_OUTPUT_GAP",
    ]
    assert classified[0]["moment_ids"] == ["moment_a"]
    assert classified[1]["moment_ids"] == ["moment_a", "moment_b"]
    assert classified[2]["moment_ids"] == []
    assert classified[2]["exceeds_dead_air_limit"] is True


def test_sparse_filter_montage_uses_audio_safe_context_without_losing_authored_material(tmp_path: Path) -> None:
    artifact = build_core_moments(
        source_id="film_a",
        source_media_hash="hash_a",
        shots=[
            _shot(f"s{index}", index * 10.0, (index + 1) * 10.0, f"scene_{index}")
            for index in range(5)
        ],
    )
    for moment in artifact["moments"]:
        moment["audio_activity"] = {"eligible": True}
    schedule = {"mappings": [{
        "clip_id": "authored_line",
        "enabled": True,
        "destination_timestamp": 1.0,
        "planned_render_duration": 2.0,
    }]}

    eligible = moments_with_schedule_coverage(
        artifact,
        schedule,
        include_audio_safe_context=True,
    )
    plan = build_montage_plan(
        filter_id="multiworld.contagion",
        filter_contract_version="2.0",
        moment_artifacts=[eligible],
        target_duration=40.0,
        minimum_moments=4,
        random_seed=7,
        governing_relationship="contagion",
        laws={
            "visual": "safe",
            "temporal": "seeded",
            "dialogue": "contagious",
            "requested_audio": "source",
            "actual_audio_method": "source_bed",
        },
        schedule=schedule,
    )

    assert eligible["moment_count"] == 5
    assert eligible["authored_placement_moment_count"] == 1
    assert eligible["context_moment_count"] == 4
    assert plan["actual_duration"] == 40.0
    assert plan["material_utilization"]["utilization_ratio"] == 1.0
    assert plan["material_utilization"]["selected_authored_moment_count"] == 1
    assert plan["material_utilization"]["selected_context_moment_count"] == 3
    path = tmp_path / "montage_plan.json"
    write_json(path, plan)
    validate_artifact("montage_plan", path, SCHEMAS)


def test_montage_planner_rejects_trivial_underuse_of_safe_material() -> None:
    artifact = build_core_moments(
        source_id="film_a",
        source_media_hash="hash_a",
        shots=[
            _shot(f"s{index}", index * 10.0, (index + 1) * 10.0, f"scene_{index}")
            for index in range(5)
        ],
    )
    for moment in artifact["moments"]:
        moment["audio_activity"] = {"eligible": True}
    eligible = moments_with_schedule_coverage(
        artifact,
        {"mappings": [{"enabled": True, "destination_timestamp": 1.0, "planned_render_duration": 2.0}]},
        include_audio_safe_context=True,
    )

    with pytest.raises(ValueError, match="safe material must be substantially utilized"):
        build_montage_plan(
            filter_id="multiworld.contagion",
            filter_contract_version="2.0",
            moment_artifacts=[eligible],
            target_duration=40.0,
            minimum_moments=4,
            maximum_moments=1,
            random_seed=7,
            governing_relationship="contagion",
            laws={
                "visual": "safe",
                "temporal": "seeded",
                "dialogue": "contagious",
                "requested_audio": "source",
                "actual_audio_method": "source_bed",
            },
        )


def test_destination_windows_are_annotated_before_filter_placement() -> None:
    artifact = build_core_moments(
        source_id="film_a", source_media_hash="hash_a",
        shots=[_shot("safe", 0.0, 5.0, "safe_scene"), _shot("unsafe", 5.0, 10.0, "unsafe_scene")],
    )
    artifact["moments"][0]["audio_activity"] = {"eligible": True}
    artifact["moments"][1]["audio_activity"] = {"eligible": False}
    windows = [
        {"id": "safe_window", "start": 1.0, "end": 3.0, "duration": 2.0},
        {"id": "unsafe_window", "start": 6.0, "end": 8.0, "duration": 2.0},
        {"id": "boundary_window", "start": 4.0, "end": 6.0, "duration": 2.0},
    ]

    rows = annotate_windows_with_montage_eligibility(windows, artifact)

    assert rows[0]["montage_placement_eligible"] is True
    assert rows[0]["montage_eligibility_reason"] == "COMPLETE_AUDIO_SAFE_MOMENT"
    assert rows[1]["montage_placement_eligible"] is False
    assert rows[1]["montage_eligibility_reason"] == "SOURCE_SOUNDTRACK_SUSTAINED_DEAD_AIR"
    assert rows[2]["montage_moment_id"] is None
    assert rows[2]["montage_eligibility_reason"] == "NOT_CONTAINED_IN_COMPLETE_CINEMATIC_MOMENT"


def test_montage_planner_relaxes_count_instead_of_exceeding_duration() -> None:
    artifact = build_core_moments(
        source_id="film_a", source_media_hash="hash_a",
        shots=[_shot(f"s{index}", index * 20.0, (index + 1) * 20.0) for index in range(4)],
    )
    plan = build_montage_plan(
        filter_id="translation.self_shuffle", filter_contract_version="2.0",
        moment_artifacts=[artifact], target_duration=45.0, minimum_moments=4,
        random_seed=5, governing_relationship="self_recollection",
        laws={"visual": "safe", "temporal": "seeded", "dialogue": "reassigned", "requested_audio": "ambient", "actual_audio_method": "minimal"},
    )

    assert plan["actual_duration"] <= 45.0
    assert len(plan["selected_moments"]) == 2
    assert plan["fallback_decisions"][0]["action"] == "RELAXED_MINIMUM_MOMENT_COUNT"


def test_montage_planner_shortens_video_to_non_repeated_source_audio() -> None:
    artifact = build_core_moments(
        source_id="film_a", source_media_hash="hash_a",
        shots=[_shot("s1", 0.0, 4.0, "scene_1"), _shot("s2", 4.0, 8.0, "scene_2")],
    )
    plan = build_montage_plan(
        filter_id="multiworld.translation", filter_contract_version="2.0",
        moment_artifacts=[artifact], target_duration=20.0, minimum_moments=2,
        random_seed=5, governing_relationship="translation",
        laws={"visual": "safe", "temporal": "seeded", "dialogue": "translated", "requested_audio": "source", "actual_audio_method": "source_bed"},
        schedule={"mappings": [
            {"clip_id": "line_1", "enabled": True},
            {"clip_id": "line_2", "enabled": True},
        ]},
        configured_minimum_duration=12.0,
    )

    assert plan["actual_duration"] == 8.0
    assert plan["duration_resolution"] == {
        "policy": "TARGET_IS_CEILING_SHORTEN_TO_NON_REPEATED_SOURCE_AUDIO",
        "requested_target_duration": 20.0,
        "configured_minimum_duration": 12.0,
        "available_non_repeating_montage_duration": 8.0,
        "resolved_duration": 8.0,
        "shortened": True,
        "configured_minimum_relaxed": True,
    }
    assert plan["repetition_policy"]["authorized"] is False
    assert plan["repetition_policy"]["observed_repeated_placement_count"] == 0
    assert plan["fallback_decisions"][-1]["action"] == "SHORTENED_TARGET_VIDEO_TO_AVAILABLE_SOURCE_AUDIO"
    assert plan["fallback_decisions"][-1]["reason"] == "INSUFFICIENT_NON_REPEATED_SOURCE_AUDIO_FOR_TARGET_DURATION"


def test_montage_planner_rejects_repetition_without_filter_plan_authorization() -> None:
    artifact = build_core_moments(
        source_id="film_a", source_media_hash="hash_a", shots=[_shot("s1", 0.0, 5.0)],
    )
    kwargs = {
        "filter_id": "translation.self_shuffle", "filter_contract_version": "2.0",
        "moment_artifacts": [artifact], "target_duration": 5.0, "minimum_moments": 1,
        "random_seed": 1, "governing_relationship": "self_recollection",
        "laws": {"visual": "safe", "temporal": "seeded", "dialogue": "reassigned", "requested_audio": "source", "actual_audio_method": "source_bed"},
        "schedule": {"mappings": [
            {"clip_id": "line_1", "enabled": True},
            {"clip_id": "line_1", "enabled": True},
        ]},
    }

    with pytest.raises(ValueError, match="without explicit authorization"):
        build_montage_plan(**kwargs)

    plan = build_montage_plan(
        **kwargs,
        repetition_authorized=True,
        repetition_authorization_basis="FILTER_PARAMETER:allow_line_reuse",
    )
    assert plan["repetition_policy"]["authorized"] is True
    assert plan["repetition_policy"]["authorization_basis"] == "FILTER_PARAMETER:allow_line_reuse"
    assert plan["repetition_policy"]["observed_repeated_source_clip_ids"] == ["line_1"]
    assert plan["repetition_policy"]["observed_repeated_placement_count"] == 1


def test_render_acceptance_passes_tolerant_conformance_without_promoting_verdict(tmp_path: Path) -> None:
    artifact = build_core_moments(source_id="film_a", source_media_hash="hash_a", shots=[_shot("s1", 0.0, 5.0)])
    plan = build_montage_plan(
        filter_id="translation.self_shuffle", filter_contract_version="2.0", moment_artifacts=[artifact],
        target_duration=5.0, minimum_moments=1, random_seed=1, governing_relationship="self_recollection",
        laws={"visual": "safe", "temporal": "seeded", "dialogue": "reassigned", "requested_audio": "ambient", "actual_audio_method": "minimal"},
    )
    path = tmp_path / "montage_render_acceptance.json"
    acceptance = build_montage_render_acceptance(
        plan=plan,
        encoded_probe={"format": {"duration": "5.018"}, "streams": [{"codec_type": "video"}, {"codec_type": "audio"}]},
        output_path=path,
    )

    assert acceptance["acceptance_status"] == "PASS"
    assert acceptance["plan_verdict"] == "EXPERIMENTAL"
    validate_artifact("montage_render_acceptance", path, SCHEMAS)


def test_self_shuffle_result_guard_rejects_legacy_or_failed_artifacts(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="missing"):
        _require_montage_native_self_shuffle_result(TransformationResult(transformation_id="self_shuffle", outputs={}))

    plan_path = tmp_path / "montage_plan.json"
    acceptance_path = tmp_path / "montage_render_acceptance.json"
    write_json(plan_path, {"filter_id": "translation.self_shuffle", "verdict": "EXPERIMENTAL"})
    write_json(acceptance_path, {"filter_id": "translation.self_shuffle", "acceptance_status": "FAIL"})
    result = TransformationResult(
        transformation_id="self_shuffle", outputs={},
        artifacts={"montage_plan": plan_path, "montage_render_acceptance": acceptance_path},
    )
    with pytest.raises(RuntimeError, match="invalid filter identity or production verdict"):
        _require_montage_native_self_shuffle_result(result)


def test_self_shuffle_result_guard_accepts_full_timeline_pass(tmp_path: Path) -> None:
    plan_path = tmp_path / "montage_plan.json"
    acceptance_path = tmp_path / "montage_render_acceptance.json"
    write_json(plan_path, {"filter_id": "translation.self_shuffle", "verdict": "PRODUCTION_READY"})
    write_json(acceptance_path, {"filter_id": "translation.self_shuffle", "acceptance_status": "PASS"})

    _require_montage_native_self_shuffle_result(TransformationResult(
        transformation_id="self_shuffle", outputs={},
        artifacts={"montage_plan": plan_path, "montage_render_acceptance": acceptance_path},
    ))


def test_legacy_run_mutation_self_shuffle_redirects_to_transformation_engine(tmp_path: Path) -> None:
    pipeline = object.__new__(Pipeline)
    messages = []
    pipeline.logger = type("Logger", (), {"info": lambda self, message: messages.append(message)})()
    expected = TransformationResult(
        transformation_id="self_shuffle",
        outputs={"video": tmp_path / "montage.mp4"},
        artifacts={"montage_plan": tmp_path / "plan.json", "montage_render_acceptance": tmp_path / "acceptance.json"},
    )
    calls = []
    pipeline.execute_transformation = lambda transformation_id, force=False, parameters=None: calls.append((transformation_id, force, parameters)) or expected

    result = pipeline.run_mutation("self_shuffle", force=True, parameters={"seed": 17})

    assert calls == [("translation.self_shuffle", True, {"seed": 17})]
    assert result["video"] == tmp_path / "montage.mp4"
    assert result["montage_plan"] == tmp_path / "plan.json"
    assert "full-timeline" in messages[0]


def test_removed_best_short_entrypoint_is_rejected(tmp_path: Path) -> None:
    pipeline = object.__new__(Pipeline)
    messages = []
    pipeline.logger = type("Logger", (), {"info": lambda self, message: messages.append(message)})()
    with pytest.raises(ValueError, match="Best Short was removed"):
        pipeline.run_best_short_remix(app_mode="Cinelingus", mutation_id="flashback")


def test_full_timeline_plan_uses_complete_anchor_and_curtails_to_audio() -> None:
    plan = build_full_timeline_plan(
        filter_id="multiworld.translation",
        filter_contract_version="1.0.0",
        anchor_source_id="film_a",
        anchor_media_hash="anchorhash",
        anchor_duration=7200.0,
        supporting_audio_durations=[5400.0],
        shot_ids=["s1", "s2"],
        random_seed=1,
        governing_relationship="Translation",
        laws={
            "visual": "COMPLETE_ANCHOR_TIMELINE_FROM_ZERO",
            "temporal": "ANCHOR_CHRONOLOGY_PRESERVED",
            "dialogue": "transfer dialogue",
            "requested_audio": "TRANSLATION_LAW",
            "actual_audio_method": "CONTINUOUS_SOURCE_SOUNDTRACK_BED",
        },
        schedule={"mappings": []},
    )

    assert plan["actual_duration"] == 5400.0
    assert plan["selected_moments"][0]["start"] == 0.0
    assert plan["selected_moments"][0]["end"] == 5400.0
    assert plan["duration_resolution"]["policy"] == "FULL_SOURCE_TIMELINE_LIMITED_BY_SUPPORTING_AUDIO"
    assert plan["fallback_decisions"][0]["action"] == "CURTAILED_VIDEO_TO_SUPPORTING_AUDIO"


def test_montage_plan_records_explicit_minimum_count_fallback() -> None:
    artifact = build_core_moments(
        source_id="film_a",
        source_media_hash="hash_a",
        shots=[_shot("s1", 0.0, 4.0), _shot("s2", 4.0, 8.0)],
    )
    plan = build_montage_plan(
        filter_id="translation.self_shuffle",
        filter_contract_version="1.0.0",
        moment_artifacts=[artifact],
        target_duration=45.0,
        minimum_moments=4,
        random_seed=1,
        governing_relationship="repetition",
        laws={"visual": "broad_sampling", "temporal": "seeded_reordering", "dialogue": "unchanged", "requested_audio": "ORIGINAL_REALITY", "actual_audio_method": "ORIGINAL_REALITY"},
    )

    assert plan["fallback_decisions"][0]["action"] == "RELAXED_MINIMUM_MOMENT_COUNT"
    assert plan["fallback_decisions"][0]["capability_tag"] == "FALLBACK_INFERENCE"


def test_self_shuffle_coverage_and_rebase_preserve_source_timestamps() -> None:
    artifact = build_core_moments(
        source_id="film_a",
        source_media_hash="hash_a",
        shots=[_shot("s1", 0.0, 4.0), _shot("s2", 4.0, 8.0), _shot("s3", 8.0, 12.0)],
    )
    schedule = {"mappings": [
        {"clip_id": "c1", "enabled": True, "destination_timestamp": 1.0},
        {"clip_id": "c3", "enabled": True, "destination_timestamp": 9.0},
    ]}
    eligible = moments_with_schedule_coverage(artifact, schedule)
    plan = build_montage_plan(
        filter_id="translation.self_shuffle",
        filter_contract_version="2.0",
        moment_artifacts=[eligible],
        target_duration=8.0,
        minimum_moments=2,
        random_seed=3,
        governing_relationship="self_recollection",
        laws={"visual": "safe", "temporal": "seeded", "dialogue": "reassigned", "requested_audio": "ambient", "actual_audio_method": "minimal"},
    )
    rebased = rebase_schedule_to_montage(schedule, plan)

    assert eligible["moment_count"] == 2
    assert rebased["self_shuffle_render_strategy"] == "shot_aware_montage_v1"
    assert rebased["montage_plan_verdict"] == "EXPERIMENTAL"
    assert {row["source_destination_timestamp"] for row in rebased["mappings"]} == {1.0, 9.0}
    assert all(row.get("montage_moment_id") for row in rebased["mappings"])


def test_rebase_moves_speech_evidence_and_coverage_into_montage_coordinates() -> None:
    schedule = {
        "destination_speech_regions": [
            {"id": "w1", "start": 10.0, "end": 12.0, "duration": 2.0, "transcript": "original destination words"},
        ],
        "destination_performance_fills": [{
            "destination_performance_id": "p1", "start": 10.0, "duration": 2.0,
            "target_coverage": 0.8, "speech_windows": [
                {"id": "w1", "start": 10.0, "end": 12.0, "duration": 2.0},
            ],
        }],
        "mappings": [{
            "clip_id": "c1", "enabled": True, "destination_timestamp": 10.0,
            "planned_render_duration": 2.0, "alignment_mode": "speech_window_snap",
            "alignment_source_window_ids": ["w1"], "alignment_slot_start": 10.0,
            "alignment_slot_end": 12.0, "window_id": "w1",
        }],
    }
    plan = {
        "selected_moments": [{
            "id": "m1", "start": 8.0, "end": 14.0, "montage_index": 0,
            "source_id": "film_a", "source_media_hash": "hash_a",
        }],
        "verdict": "EXPERIMENTAL",
    }

    rebased = rebase_schedule_to_montage(schedule, plan)

    assert rebased["destination_speech_regions"][0]["start"] == 2.0
    assert rebased["destination_speech_regions"][0]["end"] == 4.0
    rebased_id = rebased["destination_speech_regions"][0]["id"]
    assert rebased["mappings"][0]["alignment_source_window_ids"] == [rebased_id]
    assert rebased["mappings"][0]["alignment_slot_start"] == 2.0
    assert rebased["destination_performance_fills"][0]["coverage"] == 1.0
    assert rebased["destination_performance_fills"][0]["uncovered_speech_window_count"] == 0


def test_self_shuffle_excludes_dialogue_that_would_cross_a_moment_boundary() -> None:
    artifact = build_core_moments(
        source_id="film_a", source_media_hash="hash_a",
        shots=[_shot("s1", 0.0, 4.0), _shot("s2", 4.0, 8.0)],
    )
    schedule = {"mappings": [{
        "clip_id": "crossing", "enabled": True, "destination_timestamp": 3.0,
        "planned_render_duration": 2.0,
    }]}

    assert moments_with_schedule_coverage(artifact, schedule)["moment_count"] == 0


def test_pipeline_materializes_and_reuses_core_moment_artifact(tmp_path: Path) -> None:
    pipeline = object.__new__(Pipeline)
    pipeline.destination = type(
        "Destination",
        (),
        {
            "media_hash": "hash_a",
            "cache_dir": tmp_path / "cache",
            "manifest_path": tmp_path / "cache" / "manifest.json",
            "role": "destination_video",
            "media_path": tmp_path / "film.mp4",
        },
    )()
    pipeline.destination.cache_dir.mkdir(parents=True)
    write_json(
        pipeline.destination.manifest_path,
        {"schema_version": "1.0", "media_hash": "hash_a", "artifacts": {}},
    )
    pipeline.schemas_dir = SCHEMAS
    pipeline.cancel_check = None
    messages = []
    pipeline.logger = type("Logger", (), {"info": lambda self, message: messages.append(message)})()
    visual = {"shots": {"shots": [_shot("s1", 0.0, 4.0), _shot("s2", 4.0, 8.0)]}}
    timeline = {"windows": [{"start": 3.5, "end": 4.5}]}

    first = pipeline.build_cinematic_moments(visual=visual, timeline=timeline)
    repeated = pipeline.build_cinematic_moments(visual=visual, timeline=timeline)

    assert first["moment_count"] == 1
    assert repeated == first
    assert any("reused cinematic moments" in message for message in messages)
    assert (pipeline.destination.cache_dir / "cinematic_moments.json").exists()


def test_evaluation_reports_category_metrics_and_naive_comparison_without_overclaiming(tmp_path: Path) -> None:
    core = [
        {"id": "a", "accepted": True, "human_label": "POSITIVE", "categories": ["positive_boundaries", "speech_heavy", "hard_cuts"], "failure_codes": [], "fallback": False},
        {"id": "b", "accepted": True, "human_label": "POSITIVE", "categories": ["positive_boundaries", "low_motion"], "failure_codes": [], "fallback": False},
        {"id": "c", "accepted": False, "human_label": "NEGATIVE", "categories": ["negative_boundaries", "high_motion"], "failure_codes": ["SUBJECT_MOTION_INTERRUPTED"], "fallback": True},
    ]
    naive = [
        {"id": "a", "accepted": True, "human_label": "POSITIVE", "categories": ["positive_boundaries", "speech_heavy", "hard_cuts"], "failure_codes": ["MID_WORD"], "fallback": False},
        {"id": "b", "accepted": True, "human_label": "POSITIVE", "categories": ["positive_boundaries", "low_motion"], "failure_codes": [], "fallback": False},
        {"id": "c", "accepted": True, "human_label": "NEGATIVE", "categories": ["negative_boundaries", "high_motion"], "failure_codes": ["SUBJECT_MOTION_INTERRUPTED"], "fallback": False},
    ]
    report = build_montage_evaluation(
        core_records=core,
        naive_records=naive,
        configuration={"boundary_threshold": 0.06},
        source_manifest_version="corpus-v1",
        corpus_split_version="held-out-v1",
        planner_version="montage_planner_v1",
        filter_contract_version="1.0.0",
        model_inventory=[],
        capability_availability={"core": True, "enhanced_vision": False, "enhanced_audio": False},
        random_seed=1,
        plan_reproducible=True,
        montage_checks={"minimum_moment_or_fallback": True, "source_participation": True, "complete_provenance": True},
    )

    assert report["verdict"] == "EXPERIMENTAL"
    assert report["category_metrics"]["speech_heavy"]["sample_count"] == 1
    assert report["naive_sampler_comparison"]["severe_speech_failure_reduction"] == 1.0
    assert report["naive_sampler_comparison"]["human_acceptability_improvement_percentage_points"] > 0
    assert report["production_readiness_checks"]["held_out_sample_size_sufficient"] is False
    assert report["production_readiness_checks"]["source_start_bias"] is False
    path = tmp_path / "montage_evaluation.json"
    write_json(path, report)
    assert validate_artifact("montage_evaluation", path, SCHEMAS)["verdict"] == "EXPERIMENTAL"


def test_openings_vary_across_seeded_plans_without_destination_intro_privilege() -> None:
    artifact = build_core_moments(
        source_id="film_a", source_media_hash="hash_a",
        shots=[_shot(f"s{index}", index * 5.0, (index + 1) * 5.0) for index in range(24)],
    )
    laws = {"visual": "safe", "temporal": "seeded", "dialogue": "reassigned", "requested_audio": "ambient", "actual_audio_method": "minimal"}
    records = []
    for seed in range(20):
        plan = build_montage_plan(
            filter_id="translation.self_shuffle", filter_contract_version="2.0", moment_artifacts=[artifact],
            target_duration=60.0, minimum_moments=8, random_seed=seed,
            governing_relationship="self_recollection", laws=laws,
        )
        records.append(plan["opening_selection"])

    check = build_source_start_bias_check(records)
    assert check["passed"] is True
    assert check["distinct_opening_count"] >= 4
    assert check["normalized_timeline_span"] >= 0.5
