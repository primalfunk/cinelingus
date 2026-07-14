from __future__ import annotations

from pathlib import Path

from movie_masher.dialogue_reel import build_dialogue_scene_artifact, build_scene_pair_candidates, build_vignette_reel_report, build_vignette_schedule, offset_vignette_schedule, select_vignette_reel
from movie_masher.intervals import covered_speech_duration
from movie_masher.validation import validate_artifact


def test_dialogue_scene_artifact_converts_performances() -> None:
    artifact = build_dialogue_scene_artifact(
        media_hash="hash",
        role="destination_video",
        performances={
            "performances": [
                {
                    "id": "p1",
                    "start": 10.0,
                    "end": 18.0,
                    "duration": 8.0,
                    "speaker_sequence": ["A", "B", "A"],
                    "estimated_speaker_count": 2,
                    "dialogue_event_ids": ["d1", "d2"],
                    "pause_statistics": {"average": 0.4},
                    "words_per_second": 2.0,
                    "estimated_energy": 0.7,
                    "silence_ratio": 0.2,
                }
            ]
        },
    )

    scene = artifact["scenes"][0]
    assert scene["scene_id"] == "p1"
    assert scene["speaker_transitions"] == ["A->B", "B->A"]
    assert scene["silence_percentage"] == 20.0


def test_scene_pair_candidates_score_and_reject_close_self_shuffle() -> None:
    destination = build_dialogue_scene_artifact(
        media_hash="same",
        role="destination_video",
        performances={"performances": [_perf("dp1", 100.0, ["A", "B"])]},
    )
    source = build_dialogue_scene_artifact(
        media_hash="same",
        role="source_dialogue",
        performances={"performances": [_perf("sp1", 110.0, ["A", "B"]), _perf("sp2", 180.0, ["A", "B"])]},
    )
    schedule = {
        "mappings": [
            _mapping("dp1", "sp1", 100.0, 110.0, 0),
            _mapping("dp1", "sp2", 100.0, 180.0, 1),
        ]
    }

    candidates = build_scene_pair_candidates(
        schedule=schedule,
        source_scenes=source,
        destination_scenes=destination,
        self_shuffle=True,
        minimum_temporal_separation=30.0,
    )

    assert [row["source_scene_id"] for row in candidates["candidates"]] == ["sp2"]
    assert candidates["rejected_candidates"][0]["reason_rejected"] == "self_shuffle_temporal_separation"


def test_scene_pair_candidates_resolve_schedule_local_source_group_by_timestamp() -> None:
    destination = build_dialogue_scene_artifact(
        media_hash="same",
        role="destination_video",
        performances={"performances": [_perf("dp1", 100.0, ["A", "B"])]},
    )
    source = build_dialogue_scene_artifact(
        media_hash="same",
        role="source_dialogue",
        performances={"performances": [_perf("sp1", 180.0, ["A", "B"])]},
    )
    schedule = {"mappings": [_mapping("dp1", "source_group_000001", 100.0, 182.0, 0)]}

    candidates = build_scene_pair_candidates(
        schedule=schedule,
        source_scenes=source,
        destination_scenes=destination,
        self_shuffle=True,
        minimum_temporal_separation=30.0,
    )

    assert candidates["candidate_count"] == 1
    assert candidates["candidates"][0]["source_scene_id"] == "sp1"
    assert candidates["candidates"][0]["source_scene_resolution"] == ["source_timestamp"]


def test_scene_pair_candidates_report_unresolved_source_scene() -> None:
    destination = build_dialogue_scene_artifact(
        media_hash="same",
        role="destination_video",
        performances={"performances": [_perf("dp1", 100.0, ["A"])]},
    )
    schedule = {"mappings": [_mapping("dp1", "missing", 100.0, 500.0, 0)]}

    candidates = build_scene_pair_candidates(
        schedule=schedule,
        source_scenes={"scenes": []},
        destination_scenes=destination,
    )

    assert candidates["candidate_count"] == 0
    assert candidates["rejected_candidates"][0]["reason_rejected"] == "source_scene_not_found"


def test_scene_pair_candidate_renders_all_mappings_for_destination_scene() -> None:
    destination = build_dialogue_scene_artifact(
        media_hash="same",
        role="destination_video",
        performances={"performances": [_perf("dp1", 100.0, ["A", "B"])]},
    )
    source = build_dialogue_scene_artifact(
        media_hash="same",
        role="source_dialogue",
        performances={"performances": [_perf("sp1", 180.0, ["A"]), _perf("sp2", 220.0, ["B"])]},
    )
    first = _mapping("dp1", "sp1", 100.0, 180.0, 0)
    second = _mapping("dp1", "sp2", 104.0, 220.0, 1)
    first["planned_render_duration"] = first["clip_trim_duration"] = 4.0
    second["planned_render_duration"] = second["clip_trim_duration"] = 4.0

    candidates = build_scene_pair_candidates(
        schedule={"mappings": [first, second]},
        source_scenes=source,
        destination_scenes=destination,
        self_shuffle=True,
    )

    assert candidates["candidate_count"] == 2
    assert candidates["candidates"][0]["mapping_indices"] == [0, 1]
    assert candidates["candidates"][0]["pair_mapping_count"] == 1
    assert candidates["candidates"][0]["dialogue_coverage"] == 1.0


def test_scene_pair_candidates_reject_sparse_destination_scene() -> None:
    destination = build_dialogue_scene_artifact(
        media_hash="same",
        role="destination_video",
        performances={"performances": [_perf("dp1", 100.0, ["A"])]},
    )
    source = build_dialogue_scene_artifact(
        media_hash="same",
        role="source_dialogue",
        performances={"performances": [_perf("sp1", 180.0, ["A"])]},
    )
    mapping = _mapping("dp1", "sp1", 100.0, 180.0, 0)
    mapping["planned_render_duration"] = mapping["clip_trim_duration"] = 2.0

    candidates = build_scene_pair_candidates(
        schedule={"mappings": [mapping]},
        source_scenes=source,
        destination_scenes=destination,
    )

    assert candidates["candidate_count"] == 0
    assert candidates["rejected_candidates"][0]["reason_rejected"] == "insufficient_dialogue_coverage"


def test_select_vignette_reel_chooses_multiple_non_reused_moments() -> None:
    candidates = {
        "candidates": [
            _candidate("c1", "dp1", "sp1", 10, 18, 0.9),
            _candidate("c2", "dp2", "sp2", 40, 48, 0.85),
            _candidate("c3", "dp1", "sp3", 70, 78, 0.99),
        ]
    }

    reel = select_vignette_reel(
        candidates=candidates,
        target_duration_seconds=14,
        minimum_duration_seconds=8,
        maximum_duration_seconds=30,
    )

    assert reel["selection_status"] == "multi_vignette"
    assert [row["id"] for row in reel["selected_vignettes"]] == ["c1", "c2"]


def test_select_vignette_reel_never_exceeds_hard_maximum_while_seeking_minimum() -> None:
    candidates = {
        "candidates": [
            _candidate("c1", "dp1", "sp1", 0, 40, 0.9),
            _candidate("c2", "dp2", "sp2", 50, 320, 0.85),
            _candidate("c3", "dp3", "sp3", 330, 410, 0.8),
        ]
    }

    reel = select_vignette_reel(
        candidates=candidates,
        target_duration_seconds=180,
        minimum_duration_seconds=120,
        maximum_duration_seconds=300,
    )

    assert reel["actual_scene_duration_seconds"] == 120
    assert [row["id"] for row in reel["selected_vignettes"]] == ["c1", "c3"]
    assert reel["rejected_candidates"][0]["reason_rejected"] == "would_exceed_maximum_duration"


def test_build_vignette_schedule_rebases_independent_moment() -> None:
    schedule = {
        "mappings": [
            {
                "enabled": True,
                "destination_timestamp": 40.0,
                "planned_render_duration": 2.0,
                "clip_trim_duration": 2.0,
                "alignment_slot_start": 40.0,
                "alignment_slot_end": 42.0,
            }
        ]
    }
    vignette = {"id": "v1", "mapping_indices": [0], "destination_start": 39.5, "destination_duration": 5.0}

    vignette_schedule = build_vignette_schedule(schedule, vignette, padding=0.5)

    assert vignette_schedule["mappings"][0]["destination_timestamp"] == 1.0
    assert vignette_schedule["source_candidate_id"] == "v1"


def test_offset_vignette_schedule_preserves_sequential_reel_coverage() -> None:
    first = {'mappings': [{
        'destination_timestamp': 0.5,
        'planned_render_duration': 4.0,
        'alignment_slot_start': 0.5,
        'alignment_slot_end': 4.5,
    }]}
    second = offset_vignette_schedule(first, offset_seconds=5.0)
    combined = first['mappings'] + second['mappings']

    assert second['mappings'][0]['destination_timestamp'] == 5.5
    assert second['mappings'][0]['alignment_slot_start'] == 5.5
    assert second['mappings'][0]['alignment_slot_end'] == 9.5
    assert covered_speech_duration(combined, []) == 8.0


def _perf(perf_id: str, start: float, sequence: list[str]) -> dict:
    return {
        "id": perf_id,
        "start": start,
        "end": start + 8.0,
        "duration": 8.0,
        "speaker_sequence": sequence,
        "estimated_speaker_count": len(set(sequence)),
        "pause_statistics": {"average": 0.5},
        "words_per_second": 2.0,
        "estimated_energy": 0.5,
        "silence_ratio": 0.2,
    }


def _mapping(destination: str, source: str, destination_timestamp: float, source_timestamp: float, index: int) -> dict:
    return {
        "enabled": True,
        "_schedule_index": index,
        "destination_performance_id": destination,
        "source_performance_id": source,
        "destination_timestamp": destination_timestamp,
        "clip_movie_timestamp": source_timestamp,
        "planned_render_duration": 6.0,
        "clip_trim_duration": 6.0,
        "score": 0.8,
    }


def _candidate(candidate_id: str, destination: str, source: str, start: float, end: float, score: float) -> dict:
    return {
        "id": candidate_id,
        "destination_scene_id": destination,
        "source_scene_id": source,
        "destination_start": start,
        "destination_end": end,
        "destination_duration": end - start,
        "overall_score": score,
        "mapping_indices": [0],
    }


def test_vignette_reel_report_is_short_report_compatible(tmp_path: Path) -> None:
    reel = {
        "target_duration_seconds": 180,
        "actual_scene_duration_seconds": 16,
        "selection_status": "multi_vignette",
        "selected_vignettes": [
            _candidate("c1", "dp1", "sp1", 10, 18, 0.9),
            _candidate("c2", "dp2", "sp2", 40, 48, 0.85),
        ],
        "rejected_candidates": [],
    }

    report = build_vignette_reel_report(
        reel=reel,
        candidates={"candidate_count": 2},
        destination_scenes={"scene_count": 2},
        source_scenes={"scene_count": 2},
        output_video=tmp_path / "FINAL_reel.mp4",
        output_audio=tmp_path / "reel.wav",
        output_path=tmp_path / "output_report.json",
        vignette_outputs=[],
    )

    assert report["selection_summary"]["vignette_count"] == 2
    assert report["selected_mode"] == "dialogue_reel"
    validate_artifact("short_remix_report", tmp_path / "output_report.json", Path.cwd() / "schemas")
