from pathlib import Path

from movie_masher.mutations import (
    MUTATION_CHOICES,
    build_drift_schedule,
    build_echo_schedule,
    build_mutation_plan,
    build_mutation_report,
    build_self_shuffle_schedule,
    enforce_self_shuffle_changed_lines,
    get_mutation,
)
from movie_masher.validation import validate_artifact


def test_mutation_registry_loads_initial_filters() -> None:
    assert {"echo", "drift", "self_shuffle"}.issubset(set(MUTATION_CHOICES))
    assert get_mutation("echo").display_name == "Echo"


def test_echo_and_drift_create_distinct_mutation_schedules() -> None:
    clips = [
        {"id": "c1", "path": "c1.wav", "movie_timestamp": 1.0, "duration": 1.0, "transcript": "first"},
        {"id": "c2", "path": "c2.wav", "movie_timestamp": 10.0, "duration": 2.0, "transcript": "second"},
    ]

    echo = build_echo_schedule(clips=clips, duration=40.0, parameters={"delay_seconds": 5.0, "repeat_frequency": 1, "max_repeats": 10, "duck_original_at_echoes": True})
    drift = build_drift_schedule(clips=clips, duration=40.0, parameters={"starting_offset": 1.0, "maximum_offset": 9.0})

    assert echo["mutation_id"] == "echo"
    assert drift["mutation_id"] == "drift"
    assert echo["mappings"][0]["destination_timestamp"] == 6.0
    assert drift["mappings"][0]["destination_timestamp"] > 1.0
    assert {m["mutation_operation"] for m in echo["mappings"]} == {"echo"}
    assert {m["mutation_operation"] for m in drift["mappings"]} == {"drift"}


def test_mutation_plan_and_report_validate(tmp_path: Path) -> None:
    plan_path = tmp_path / "mutation_plan.json"
    report_path = tmp_path / "mutation_report.json"
    video = tmp_path / "echo_output.mp4"
    audio = tmp_path / "echo_audio.wav"
    video.write_text("video")
    audio.write_text("audio")
    schedule = {"render_duration": 12.0, "mappings": [{"mutation_operation": "echo"}]}

    build_mutation_plan(
        mutation_id="echo",
        source_media_hash="hash",
        source_path=tmp_path / "source.mp4",
        selected_objects=[{"id": "c1"}],
        operations=[{"operation": "echo"}],
        placements=[{"clip_id": "c1", "destination_timestamp": 5.0}],
        render_strategy={"audio": "echo"},
        expected_output_path=video,
        output_path=plan_path,
        parameters={"delay_seconds": 5.0},
    )
    build_mutation_report(
        mutation_id="echo",
        source_path=tmp_path / "source.mp4",
        source_media_hash="hash",
        parameters={"delay_seconds": 5.0},
        plan_path=plan_path,
        output_video=video,
        output_audio=audio,
        schedule=schedule,
        output_path=report_path,
    )

    validate_artifact("mutation_plan", plan_path, Path.cwd() / "schemas")
    validate_artifact("mutation_report", report_path, Path.cwd() / "schemas")



def test_echo_defaults_are_audible_mutation_defaults() -> None:
    echo = get_mutation("echo")

    assert echo.default_parameters["repeat_frequency"] == 1
    assert echo.default_parameters["max_repeats"] >= 60
    assert echo.default_parameters["duck_original_at_echoes"] is True


def test_drift_defaults_create_visible_late_film_offset() -> None:
    drift = get_mutation("drift")

    assert drift.default_parameters["maximum_offset"] >= 12.0


def test_self_shuffle_disables_or_repairs_unchanged_original_lines() -> None:
    clips = [
        {"id": "same", "path": "same.wav", "movie_timestamp": 10.0, "duration": 2.0, "speaker_id": "speaker_001", "transcript": "same"},
        {"id": "other", "path": "other.wav", "movie_timestamp": 30.0, "duration": 2.0, "speaker_id": "speaker_001", "transcript": "other"},
    ]
    schedule = {
        "mappings": [
            {
                "enabled": True,
                "clip_id": "same",
                "clip_path": "same.wav",
                "destination_timestamp": 10.0,
                "alignment_slot_start": 10.0,
                "alignment_slot_end": 12.0,
                "planned_render_duration": 2.0,
                "clip_trim_duration": 2.0,
                "destination_speaker_id": "speaker_001",
                "source_speaker_id": "speaker_001",
                "selection_reason": "self_shuffle",
            }
        ]
    }

    enforce_self_shuffle_changed_lines(schedule=schedule, clips=clips)

    mapping = schedule["mappings"][0]
    assert mapping["clip_id"] == "other"
    assert mapping["self_shuffle_unchanged_line"] is False
    assert schedule["self_shuffle_policy"]["repaired_unchanged_mappings"] == 1


def test_self_shuffle_threads_source_performances_into_scheduler(monkeypatch, tmp_path: Path) -> None:
    from movie_masher import mutations

    captured = {}
    source_performances = {"performances": [{"id": "sp1"}]}

    def fake_build_schedule(**kwargs):
        captured.update(kwargs)
        return {"mappings": []}

    monkeypatch.setattr(mutations, "build_schedule", fake_build_schedule)
    build_self_shuffle_schedule(
        clips=[],
        windows=[],
        media_hash="hash",
        max_time_stretch=0.1,
        output_path=tmp_path / "schedule.json",
        seed=1,
        best_fit_lookahead=8,
        cinematic_filter="balanced",
        source_performances=source_performances,
    )

    assert captured["source_performances"] is source_performances


def test_self_shuffle_disables_unchanged_line_when_no_replacement_exists() -> None:
    clips = [
        {"id": "same", "path": "same.wav", "movie_timestamp": 10.0, "duration": 2.0, "speaker_id": "speaker_001", "transcript": "same"},
    ]
    schedule = {
        "mappings": [
            {
                "enabled": True,
                "clip_id": "same",
                "clip_path": "same.wav",
                "destination_timestamp": 10.0,
                "alignment_slot_start": 10.0,
                "alignment_slot_end": 12.0,
                "planned_render_duration": 2.0,
                "clip_trim_duration": 2.0,
                "selection_reason": "self_shuffle",
            }
        ]
    }

    enforce_self_shuffle_changed_lines(schedule=schedule, clips=clips)

    mapping = schedule["mappings"][0]
    assert mapping["enabled"] is False
    assert mapping["self_shuffle_unchanged_line"] is True
    assert schedule["self_shuffle_policy"]["disabled_unchanged_mappings"] == 1


def test_self_shuffle_mutation_renders_dialogue_only(monkeypatch, tmp_path: Path) -> None:
    from movie_masher import mutations

    original = tmp_path / "movie.mp4"
    audio = tmp_path / "self_shuffle.wav"
    video = tmp_path / "self_shuffle.mp4"
    clip = tmp_path / "clip.wav"
    original.write_text("movie")
    clip.write_text("clip")
    calls = []

    def fake_dialogue(**kwargs):
        calls.append(("dialogue", kwargs))
        kwargs["output_path"].write_text("dialogue")

    def fake_original(**kwargs):
        calls.append(("original", kwargs))

    def fake_mux(**kwargs):
        calls.append(("mux", kwargs))
        kwargs["output_path"].write_text("video")

    monkeypatch.setattr(mutations, "render_dialogue_wav", fake_dialogue)
    monkeypatch.setattr(mutations, "render_schedule_over_original_audio", fake_original)
    monkeypatch.setattr(mutations, "mux_video", fake_mux)

    schedule = {
        "mutation_id": "self_shuffle",
        "mappings": [
            {
                "enabled": True,
                "clip_path": str(clip),
                "destination_timestamp": 0.0,
                "clip_trim_start": 0.0,
                "clip_trim_duration": 1.0,
                "stretch_factor": 1.0,
            }
        ],
    }

    mutations.render_mutation_media(
        original_media=original,
        schedule=schedule,
        duration=5.0,
        audio_output=audio,
        video_output=video,
        sample_rate=48000,
        channels=2,
        target_lufs=-18.0,
        fade_duration=0.015,
    )

    assert [name for name, _ in calls] == ["dialogue", "mux"]
    assert schedule["self_shuffle_render_strategy"] == "dialogue_only_v1"
