from pathlib import Path

import pytest

from movie_masher.pipeline import Pipeline, _attach_performance_speech_windows, _speech_mute_regions, _validate_best_short_render_contract


def test_render_preview_uses_selected_mapping_indices(monkeypatch, tmp_path: Path) -> None:
    config = type(
        "Config",
        (),
        {
            "output_dir": tmp_path / "output",
            "render_sample_rate": 48000,
            "render_channels": 2,
            "target_lufs": -18.0,
            "audio_fade_duration": 0.015,
            "cinematic_filter": "balanced",
        },
    )()
    pipeline = object.__new__(Pipeline)
    pipeline.config = config
    pipeline.destination = type("Dest", (), {"media_path": tmp_path / "dest.mp4"})()
    pipeline.logger = type("Logger", (), {"info": lambda self, message: None})()
    schedule = {
        "mappings": [
            {
                "clip_path": str(tmp_path / "a.wav"),
                "destination_timestamp": 10.0,
                "planned_render_duration": 2.0,
                "clip_trim_duration": 2.0,
                "stretch_factor": 1.0,
            },
            {
                "clip_path": str(tmp_path / "b.wav"),
                "destination_timestamp": 30.0,
                "planned_render_duration": 3.0,
                "clip_trim_duration": 3.0,
                "stretch_factor": 1.0,
            },
        ]
    }
    calls = {}

    monkeypatch.setattr(pipeline, "schedule", lambda force=False: schedule)
    monkeypatch.setattr(pipeline, "_inspect_one", lambda entry, force=False: {"duration": 100.0})

    def fake_render_dialogue_wav(**kwargs):
        calls["audio_duration"] = kwargs["duration"]
        calls["preview_timestamp"] = kwargs["schedule"]["mappings"][0]["destination_timestamp"]
        kwargs["output_path"].parent.mkdir(parents=True, exist_ok=True)
        kwargs["output_path"].write_text("wav")

    def fake_mux_video_segment(**kwargs):
        calls["video_start"] = kwargs["start_time"]
        calls["video_duration"] = kwargs["duration"]
        kwargs["output_path"].parent.mkdir(parents=True, exist_ok=True)
        kwargs["output_path"].write_text("mp4")

    monkeypatch.setattr("movie_masher.pipeline.render_dialogue_wav", fake_render_dialogue_wav)
    monkeypatch.setattr("movie_masher.pipeline.mux_video_segment", fake_mux_video_segment)

    result = pipeline.render_preview([1], video=True)

    assert calls["video_start"] == 29.0
    assert calls["video_duration"] == 5.0
    assert calls["audio_duration"] == 5.0
    assert calls["preview_timestamp"] == 1.0
    assert Path(result["video"]).exists()


def test_movie_masher_audio_render_is_dialogue_only(monkeypatch, tmp_path: Path) -> None:
    config = type(
        "Config",
        (),
        {
            "output_dir": tmp_path / "output",
            "render_sample_rate": 48000,
            "render_channels": 2,
            "target_lufs": -18.0,
            "audio_fade_duration": 0.015,
        },
    )()
    pipeline = object.__new__(Pipeline)
    pipeline.config = config
    pipeline.destination = type("Dest", (), {"cache_dir": tmp_path / "cache", "media_path": tmp_path / "dest.mp4"})()
    pipeline.logger = type("Logger", (), {"info": lambda self, message: None})()
    pipeline.destination.cache_dir.mkdir(parents=True)
    calls = {}

    def fake_render_dialogue_wav(**kwargs):
        calls["dialogue_only"] = True
        kwargs["output_path"].parent.mkdir(parents=True, exist_ok=True)
        kwargs["output_path"].write_text("wav")

    def fail_original_bed(**_kwargs):
        raise AssertionError("Movie Masher audio should not preserve the original bed")

    monkeypatch.setattr("movie_masher.pipeline.render_dialogue_wav", fake_render_dialogue_wav)
    monkeypatch.setattr("movie_masher.pipeline.render_schedule_over_original_audio", fail_original_bed)

    output = pipeline.render_audio_from_schedule(
        schedule={"mappings": [{"enabled": True, "destination_timestamp": 1.0, "planned_render_duration": 2.0}]},
        dest_movie={"duration": 10.0},
        force=True,
    )

    assert output.exists()
    assert calls["dialogue_only"] is True


def test_render_problem_region_previews_cuts_final_output(monkeypatch, tmp_path: Path) -> None:
    config = type(
        "Config",
        (),
        {
            "output_dir": tmp_path / "output",
        },
    )()
    pipeline = object.__new__(Pipeline)
    pipeline.config = config
    pipeline.destination = type("Dest", (), {})()
    pipeline.logger = type("Logger", (), {"info": lambda self, message: None})()
    config.output_dir.mkdir(parents=True)
    (config.output_dir / "movie_masher_output.mp4").write_text("video")
    stale_dir = config.output_dir / "previews" / "problem_regions"
    stale_dir.mkdir(parents=True)
    stale_file = stale_dir / "problem_stale.mp4"
    stale_file.write_text("old")
    (config.output_dir / "problem_regions.json").write_text(
        '{"problems":[{"problem_type":"fallback_mapping","severity":"medium","start":10.0,"end":12.0,"performance_id":"p1","mapping_indices":[3],"reason":"test"}]}'
    )
    calls = []

    monkeypatch.setattr(pipeline, "_inspect_one", lambda entry, force=False: {"duration": 20.0})

    def fake_extract_video_segment(**kwargs):
        calls.append(kwargs)
        kwargs["output_path"].parent.mkdir(parents=True, exist_ok=True)
        kwargs["output_path"].write_text("clip")

    monkeypatch.setattr("movie_masher.pipeline.extract_video_segment", fake_extract_video_segment)

    result = pipeline.render_problem_region_previews()

    assert len(result["previews"]) == 1
    assert calls[0]["start_time"] == 9.0
    assert calls[0]["duration"] == 4.0
    assert not stale_file.exists()
    assert Path(result["manifest"]).exists()
    assert Path(result["text"]).exists()



def test_attach_performance_speech_windows_recovers_referenced_filtered_window() -> None:
    rows = [
        {
            "id": "p1",
            "start": 10.0,
            "duration": 2.0,
            "speaking_window_ids": ["w_filtered"],
        }
    ]
    timeline = [
        {
            "id": "w_filtered",
            "start": 10.0,
            "end": 12.0,
            "duration": 2.0,
            "usable": False,
            "reject_reason": "repeated_text",
            "confidence": 0.2,
        }
    ]

    enriched = _attach_performance_speech_windows(rows, timeline)

    assert enriched[0]["speech_windows"][0]["id"] == "w_filtered"
    assert enriched[0]["speech_windows"][0]["source_kind"] == "recovered_filtered_speech_window"
    assert enriched[0]["speech_windows"][0]["recovered"] is True



def test_speech_mute_regions_prefer_snapped_speech_slots() -> None:
    schedule = {
        "destination_performance_fills": [
            {"start": 0.0, "duration": 10.0},
        ],
        "mappings": [
            {
                "enabled": True,
                "alignment_mode": "speech_window_snap",
                "alignment_slot_start": 1.0,
                "alignment_slot_end": 2.0,
            },
            {
                "enabled": True,
                "alignment_mode": "speech_window_snap",
                "alignment_slot_start": 4.0,
                "alignment_slot_end": 5.0,
            },
            {
                "enabled": True,
                "alignment_mode": "speech_window_snap",
                "alignment_slot_start": 4.0,
                "alignment_slot_end": 5.0,
            },
        ],
    }

    regions = _speech_mute_regions(schedule, padding=0.25, merge_gap=0.1, duration=10.0)

    assert regions == [
        {"start": 0.75, "duration": 1.5},
        {"start": 3.75, "duration": 1.5},
    ]


def test_speech_mute_regions_falls_back_to_performance_regions() -> None:
    schedule = {
        "destination_performance_fills": [
            {"start": 0.1, "duration": 2.0},
            {"start": 2.45, "duration": 1.0},
            {"start": 9.8, "duration": 1.0},
        ]
    }

    regions = _speech_mute_regions(schedule, padding=0.35, merge_gap=0.25, duration=10.0)

    assert regions == [
        {"start": 0.0, "duration": 3.8},
        {"start": 9.45, "duration": 0.55},
    ]


def test_speech_mute_regions_falls_back_to_mapping_bounds() -> None:
    schedule = {
        "mappings": [
            {"enabled": True, "destination_timestamp": 5.0, "planned_render_duration": 2.0},
            {"enabled": False, "destination_timestamp": 8.0, "planned_render_duration": 2.0},
        ]
    }

    assert _speech_mute_regions(schedule, padding=0.25, duration=20.0) == [
        {"start": 4.75, "duration": 2.5}
    ]


def test_run_all_generates_problem_previews_after_success(monkeypatch, tmp_path: Path) -> None:
    pipeline = object.__new__(Pipeline)
    video = tmp_path / "movie.mp4"
    calls = []
    pipeline.logger = type("Logger", (), {"info": lambda self, message: calls.append(message)})()

    class Result:
        outputs = {"video": video}

    monkeypatch.setattr(pipeline, "execute_transformation", lambda transformation_id, force=False: Result())
    monkeypatch.setattr(pipeline, "render_problem_region_previews", lambda max_regions=10: {"previews": [{"path": "a.mp4"}]})

    assert pipeline.run_all(force=True) == video
    assert any("problem preview clips: 1" in message for message in calls)


def test_run_all_keeps_finished_video_when_problem_previews_fail(monkeypatch, tmp_path: Path) -> None:
    pipeline = object.__new__(Pipeline)
    video = tmp_path / "movie.mp4"
    calls = []
    pipeline.logger = type("Logger", (), {"info": lambda self, message: calls.append(message)})()

    class Result:
        outputs = {"video": video}

    def fail_preview(max_regions=10):
        raise FileNotFoundError("missing final video")

    monkeypatch.setattr(pipeline, "execute_transformation", lambda transformation_id, force=False: Result())
    monkeypatch.setattr(pipeline, "render_problem_region_previews", fail_preview)

    assert pipeline.run_all(force=False) == video
    assert any("problem preview generation skipped" in message for message in calls)


def test_best_short_render_contract_rejects_unchanged_audio_output() -> None:
    schedule = {"mappings": [{"enabled": True, "destination_timestamp": 1.0, "planned_render_duration": 2.0}]}

    with pytest.raises(ValueError, match="produced no mute regions"):
        _validate_best_short_render_contract(
            short_schedule=schedule,
            mute_regions=[],
            duration=10.0,
            candidate_id="bad",
        )


def test_best_short_render_contract_accepts_in_window_muted_output() -> None:
    schedule = {"mappings": [{"enabled": True, "destination_timestamp": 1.0, "planned_render_duration": 2.0}]}

    _validate_best_short_render_contract(
        short_schedule=schedule,
        mute_regions=[{"start": 0.8, "duration": 2.4}],
        duration=10.0,
        candidate_id="good",
    )


def test_best_short_render_contract_allows_dialogue_only_without_mute_regions() -> None:
    schedule = {"mappings": [{"enabled": True, "destination_timestamp": 1.0, "planned_render_duration": 2.0}]}

    _validate_best_short_render_contract(
        short_schedule=schedule,
        mute_regions=[],
        duration=10.0,
        candidate_id="dialogue-only",
        require_mute_regions=False,
    )
