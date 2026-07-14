from pathlib import Path

from movie_masher.cir import (
    CIR_OBJECT_TYPES,
    OBJECT_DIALOGUE_CLIP,
    OBJECT_RENDER_OPERATION,
    OBJECT_SHOT,
    OBJECT_TRANSFORMATION_REPORT,
    OBJECT_TRANSFORMATION_PLAN,
    build_cinematic_index,
)
from movie_masher.validation import validate_artifact


def test_build_cinematic_index_links_artifacts_and_counts(tmp_path: Path) -> None:
    root = tmp_path
    destination_cache = root / "cache" / "desthash"
    source_cache = root / "cache" / "sourcehash"
    output = root / "output"
    destination_cache.mkdir(parents=True)
    source_cache.mkdir(parents=True)
    output.mkdir()
    (output / "movie_masher").mkdir()
    for path in [
        destination_cache / "movie.json",
        source_cache / "movie.json",
        source_cache / "dialogue_events.json",
        source_cache / "filtered_dialogue_events.json",
        source_cache / "clip_library.json",
        source_cache / "performance.json",
        destination_cache / "performance.json",
        destination_cache / "timeline.json",
        destination_cache / "filtered_timeline.json",
        destination_cache / "replacement_schedule.json",
        destination_cache / "shots.json",
        destination_cache / "visual_report.json",
        output / "movie_masher" / "transformation_report.json",
        output / "movie_masher" / "transformation_plan.json",
    ]:
        path.write_text("{}")

    schedule = {
        "transformation_name": "movie_masher",
        "transformation_history": [
            {"verb": "select", "description": "select", "inputs": ["a"], "outputs": ["b"]}
        ],
        "mappings": [
            {
                "enabled": True,
                "render_operations": [
                    {"operation": "trim"},
                    {"operation": "render"},
                ],
            }
        ],
    }

    index = build_cinematic_index(
        root=root,
        output_path=output / "cinematic_index.json",
        destination_movie={"media_hash": "desthash", "path": "dest.mp4", "duration": 10.0},
        source_movie={"media_hash": "sourcehash", "path": "source.mp4", "duration": 8.0},
        source_events={"events": [{"id": "e1"}, {"id": "e2"}]},
        filtered_source_events={"filter_stats": {"usable_count": 1}},
        clip_library={"clips": [{"id": "c1"}]},
        destination_timeline={"windows": [{"id": "w1"}]},
        filtered_destination_timeline={"filter_stats": {"usable_count": 1}},
        schedule=schedule,
        audio_output=output / "replacement_dialogue.wav",
        video_output=output / "movie_masher_output.mp4",
        run_report_json=output / "run_report.json",
        schedule_report_csv=output / "schedule_report.csv",
        destination_cache=destination_cache,
        source_cache=source_cache,
        shots={"shots": [{"id": "shot_000001"}, {"id": "shot_000002"}]},
        visual_report={"average_shot_duration": 5.0},
        source_performances={"performances": [{"id": "sp1"}]},
        destination_performances={"performances": [{"id": "dp1"}, {"id": "dp2"}]},
        transformation_report=output / "movie_masher" / "transformation_report.json",
        transformation_plan=output / "movie_masher" / "transformation_plan.json",
    )

    assert OBJECT_DIALOGUE_CLIP in CIR_OBJECT_TYPES
    assert index["counts"]["dialogue_clips"] == 1
    assert index["counts"]["render_operations"] == 2
    assert index["counts"]["source_performances"] == 1
    assert index["counts"]["destination_performances"] == 2
    assert index["counts"]["shots"] == 2
    assert index["counts"]["average_shot_duration"] == 5.0
    assert index["counts"]["transformation_reports"] == 1
    assert index["counts"]["transformation_plans"] == 1
    assert index["outputs"]["transformation_report"] == "output/movie_masher/transformation_report.json"
    assert index["outputs"]["transformation_plan"] == "output/movie_masher/transformation_plan.json"
    assert index["transformation"]["name"] == "movie_masher"
    assert any(item["cir_object_type"] == OBJECT_RENDER_OPERATION for item in index["artifacts"])
    assert any(item["cir_object_type"] == OBJECT_SHOT for item in index["artifacts"])
    assert any(item["cir_object_type"] == OBJECT_TRANSFORMATION_REPORT for item in index["artifacts"])
    assert any(item["cir_object_type"] == OBJECT_TRANSFORMATION_PLAN for item in index["artifacts"])
    validate_artifact("cinematic_index", output / "cinematic_index.json", Path.cwd() / "schemas")
