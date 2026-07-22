from pathlib import Path

import pytest

from cinelingus.pipeline import Pipeline, _apply_hybrid_speech_mask, _attach_performance_speech_windows, _speech_mute_regions


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

    monkeypatch.setattr("cinelingus.pipeline.render_dialogue_wav", fake_render_dialogue_wav)
    monkeypatch.setattr("cinelingus.pipeline.mux_video_segment", fake_mux_video_segment)

    result = pipeline.render_preview([1], video=True)

    assert calls["video_start"] == 29.0
    assert calls["video_duration"] == 5.0
    assert calls["audio_duration"] == 5.0
    assert calls["preview_timestamp"] == 1.0
    assert Path(result["video"]).exists()


def test_translation_audio_render_is_dialogue_only(monkeypatch, tmp_path: Path) -> None:
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
        raise AssertionError("Translation audio should not preserve the original bed")

    monkeypatch.setattr("cinelingus.pipeline.render_dialogue_wav", fake_render_dialogue_wav)
    monkeypatch.setattr("cinelingus.pipeline.render_schedule_over_original_audio", fail_original_bed)

    output = pipeline.render_audio_from_schedule(
        schedule={"mappings": [{"enabled": True, "destination_timestamp": 1.0, "planned_render_duration": 2.0}]},
        dest_movie={"duration": 10.0},
        force=True,
    )

    assert output.exists()
    assert calls["dialogue_only"] is True


def test_translation_audio_render_runs_post_render_residue_verification(monkeypatch, tmp_path: Path) -> None:
    config = type("Config", (), {
        "output_dir": tmp_path / "output",
        "render_sample_rate": 48000, "render_channels": 2, "target_lufs": -18.0,
        "audio_fade_duration": 0.015, "original_duck_db": -28.0,
        "dialogue_suppression": "hard_mute", "suppression_padding": 0.04,
        "background_reconstruction": "neighboring_non_speech_with_adaptive_crossfades",
        "verify_voice_residue": True, "speech_backend": "whisper",
        "whisper_model": "medium", "whisper_language": "en", "transcription_mode": "quality",
    })()
    pipeline = object.__new__(Pipeline)
    pipeline.config = config
    pipeline.destination = type("Dest", (), {
        "cache_dir": tmp_path / "cache", "media_path": tmp_path / "dest.mp4", "media_hash": "destination-hash",
    })()
    pipeline.logger = type("Logger", (), {"info": lambda self, message: None})()
    pipeline.destination.cache_dir.mkdir(parents=True)
    observed = {}

    def fake_render(**kwargs):
        kwargs["schedule"]["background_reconstruction_report"] = {"reconstructed_region_count": 1}
        kwargs["output_path"].parent.mkdir(parents=True, exist_ok=True)
        kwargs["output_path"].write_bytes(b"0" * 100)

    def fake_transcribe(**kwargs):
        observed["media_hash"] = kwargs["media_hash"]
        return {"windows": []}

    monkeypatch.setattr("cinelingus.pipeline.render_schedule_over_original_audio", fake_render)
    monkeypatch.setattr("cinelingus.pipeline.transcribe_with_whisper", fake_transcribe)
    schedule = {
        "destination_speech_regions": [],
        "mappings": [{"enabled": True, "destination_timestamp": 1.0, "planned_render_duration": 2.0}],
    }

    pipeline.render_audio_from_schedule(
        schedule=schedule, dest_movie={"duration": 10.0}, force=True, persist_schedule=False,
    )

    assert observed["media_hash"]
    assert schedule["voice_residue_verification"]["status"] == "INCONCLUSIVE"


def test_translation_audio_render_corrects_and_reverifies_residue_once(monkeypatch, tmp_path: Path) -> None:
    config = type("Config", (), {
        "output_dir": tmp_path / "output",
        "render_sample_rate": 48000, "render_channels": 2, "target_lufs": -18.0,
        "audio_fade_duration": 0.015, "original_duck_db": -28.0,
        "dialogue_suppression": "hard_mute", "suppression_padding": 0.04,
        "background_reconstruction": "neighboring_non_speech_with_adaptive_crossfades",
        "verify_voice_residue": True, "residue_correction_passes": 1, "residue_correction_padding": 0.12,
        "speech_backend": "whisper", "whisper_model": "medium", "whisper_language": "en",
        "transcription_mode": "quality",
    })()
    pipeline = object.__new__(Pipeline)
    pipeline.config = config
    pipeline.destination = type("Dest", (), {
        "cache_dir": tmp_path / "cache", "media_path": tmp_path / "dest.mp4", "media_hash": "destination-hash",
    })()
    pipeline.logger = type("Logger", (), {"info": lambda self, message: None})()
    pipeline.destination.cache_dir.mkdir(parents=True)
    calls = {"renders": 0, "transcriptions": 0}

    def fake_render(**kwargs):
        calls["renders"] += 1
        kwargs["schedule"]["background_reconstruction_report"] = {"reconstructed_region_count": 1}
        kwargs["output_path"].parent.mkdir(parents=True, exist_ok=True)
        kwargs["output_path"].write_bytes(bytes([calls["renders"]]) * 100)

    def fake_transcribe(**_kwargs):
        calls["transcriptions"] += 1
        if calls["transcriptions"] == 1:
            return {"windows": [{
                "start": 1.0, "end": 2.0, "transcript": "bring the blue lantern downstairs", "confidence": 0.9,
            }]}
        return {"windows": []}

    monkeypatch.setattr("cinelingus.pipeline.render_schedule_over_original_audio", fake_render)
    monkeypatch.setattr("cinelingus.pipeline.transcribe_with_whisper", fake_transcribe)
    schedule = {
        "unmatched_policy": "suppress_original_dialogue",
        "destination_speech_regions": [{
            "id": "line", "start": 1.0, "end": 2.0, "duration": 1.0,
            "transcript": "bring the blue lantern downstairs",
        }],
        "mappings": [],
    }

    pipeline.render_audio_from_schedule(
        schedule=schedule, dest_movie={"duration": 3.0}, force=True, persist_schedule=False,
    )

    assert calls == {"renders": 2, "transcriptions": 2}
    assert schedule["voice_residue_verification"]["status"] == "NONE_DETECTED"
    assert schedule["residue_correction_report"]["completed_passes"] == 1
    assert schedule["residue_correction_regions"][0]["source_kind"] == "post_render_residue_correction"


def test_translation_audio_render_resumes_after_corrective_render_checkpoint(monkeypatch, tmp_path: Path) -> None:
    config = type("Config", (), {
        "output_dir": tmp_path / "output",
        "render_sample_rate": 48000, "render_channels": 2, "target_lufs": -18.0,
        "audio_fade_duration": 0.015, "original_duck_db": -28.0,
        "dialogue_suppression": "hard_mute", "suppression_padding": 0.04,
        "background_reconstruction": "neighboring_non_speech_with_adaptive_crossfades",
        "verify_voice_residue": True, "residue_correction_passes": 1, "residue_correction_padding": 0.12,
        "speech_backend": "whisper", "whisper_model": "medium", "whisper_language": "en",
        "transcription_mode": "quality",
    })()
    pipeline = object.__new__(Pipeline)
    pipeline.config = config
    pipeline.destination = type("Dest", (), {
        "cache_dir": tmp_path / "cache", "media_path": tmp_path / "dest.mp4", "media_hash": "destination-hash",
    })()
    pipeline.logger = type("Logger", (), {"info": lambda self, message: None})()
    pipeline.destination.cache_dir.mkdir(parents=True)
    calls = {"renders": 0, "transcriptions": 0}

    def fake_render(**kwargs):
        calls["renders"] += 1
        kwargs["schedule"]["background_reconstruction_report"] = {"reconstructed_region_count": 1}
        kwargs["output_path"].parent.mkdir(parents=True, exist_ok=True)
        kwargs["output_path"].write_bytes(bytes([calls["renders"]]) * 100)

    def interrupted_transcribe(**_kwargs):
        calls["transcriptions"] += 1
        if calls["transcriptions"] == 1:
            return {"windows": [{
                "start": 1.0, "end": 2.0, "transcript": "bring the blue lantern downstairs", "confidence": 0.9,
            }]}
        raise KeyboardInterrupt("simulated interruption after corrective render")

    monkeypatch.setattr("cinelingus.pipeline.render_schedule_over_original_audio", fake_render)
    monkeypatch.setattr("cinelingus.pipeline.transcribe_with_whisper", interrupted_transcribe)
    schedule = {
        "unmatched_policy": "suppress_original_dialogue",
        "destination_speech_regions": [{
            "id": "line", "start": 1.0, "end": 2.0, "duration": 1.0,
            "transcript": "bring the blue lantern downstairs",
        }],
        "mappings": [],
    }

    try:
        pipeline.render_audio_from_schedule(
            schedule=schedule, dest_movie={"duration": 3.0}, force=True, persist_schedule=False,
        )
    except KeyboardInterrupt:
        pass

    checkpoint = config.output_dir / "residue_correction_checkpoint.json"
    assert checkpoint.exists()
    assert calls["renders"] == 2

    monkeypatch.setattr("cinelingus.pipeline.transcribe_with_whisper", lambda **_kwargs: {"windows": []})
    pipeline.render_audio_from_schedule(
        schedule=schedule, dest_movie={"duration": 3.0}, force=True, persist_schedule=False,
    )

    assert calls["renders"] == 2
    assert schedule["residue_correction_report"]["completed_passes"] == 1
    assert schedule["residue_correction_report"]["passes"][0]["resumed_from_checkpoint"] is True


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
    (config.output_dir / "translation_output.mp4").write_text("video")
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

    monkeypatch.setattr("cinelingus.pipeline.extract_video_segment", fake_extract_video_segment)

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
            "transcript": "Destination words retained for residue contrast.",
        }
    ]

    enriched = _attach_performance_speech_windows(rows, timeline)

    assert enriched[0]["speech_windows"][0]["id"] == "w_filtered"
    assert enriched[0]["speech_windows"][0]["source_kind"] == "recovered_filtered_speech_window"
    assert enriched[0]["speech_windows"][0]["recovered"] is True
    assert enriched[0]["speech_windows"][0]["transcript"] == "Destination words retained for residue contrast."



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


def test_hybrid_speech_mask_expands_boundaries_and_adds_trusted_diarization() -> None:
    schedule = {
        "unmatched_policy": "suppress_original_dialogue",
        "destination_speech_regions": [{
            "id": "whisper", "start": 1.0, "end": 2.0, "duration": 1.0, "confidence": 0.8,
        }]
    }

    enriched = _apply_hybrid_speech_mask(
        schedule,
        acoustic_windows=[{"start": 0.9, "end": 2.5, "duration": 1.6}],
        diarization_segments=[{"start": 4.0, "end": 5.0, "duration": 1.0, "speaker_id": "speaker_1", "confidence": 0.9}],
    )

    assert enriched["destination_speech_regions"][0]["start"] == 0.9
    assert enriched["destination_speech_regions"][0]["end"] == 2.18
    assert enriched["destination_speech_regions"][1]["source_kind"] == "trusted_diarization_speech_window"
    assert enriched["speech_mask_report"]["acoustic_boundary_expansion_count"] == 1
    assert enriched["speech_mask_report"]["trusted_diarization_region_count"] == 1


def test_speech_mute_regions_expand_low_confidence_recovered_tails_more_aggressively() -> None:
    schedule = {
        "destination_speech_regions": [{
            "id": "speech", "start": 2.0, "end": 2.4, "duration": 0.4,
            "confidence": 0.5, "source_kind": "recovered_filtered_speech_window", "recovered": True,
        }],
        "mappings": [{
            "enabled": True, "alignment_mode": "speech_window_snap",
            "alignment_slot_start": 2.0, "alignment_slot_end": 2.4,
            "alignment_source_window_ids": ["speech"],
        }],
    }

    regions = _speech_mute_regions(schedule, padding=0.04, duration=5.0, adaptive=True)

    assert regions == [{"start": 1.86, "duration": 0.77}]
    report = schedule["suppression_padding_report"]
    assert report["strategy"] == "confidence_aware_asymmetric_padding_v1"
    assert report["regions"][0]["leading_padding"] == 0.14
    assert report["regions"][0]["trailing_padding"] == 0.23


def test_speech_mute_regions_honor_local_residue_repair_padding() -> None:
    schedule = {
        "unmatched_policy": "suppress_original_dialogue",
        "destination_speech_regions": [{
            "id": "speech", "start": 2.0, "end": 3.0, "duration": 1.0,
            "confidence": 1.0, "source_kind": "detected_speech_window",
        }],
        "mappings": [{
            "enabled": True, "destination_timestamp": 2.0, "planned_render_duration": 1.0,
            "suppression_leading_padding": 0.12, "suppression_trailing_padding": 0.22,
        }],
    }

    regions = _speech_mute_regions(schedule, padding=0.04, duration=5.0, adaptive=True)

    assert regions == [{"start": 1.88, "duration": 1.34}]
    report = schedule["suppression_padding_report"]
    assert report["regions"][0]["leading_padding"] == 0.12
    assert report["regions"][0]["trailing_padding"] == 0.22


def test_speech_mute_regions_include_unmatched_destination_speech_under_hard_policy() -> None:
    schedule = {
        "unmatched_policy": "suppress_original_dialogue",
        "destination_speech_regions": [
            {"id": "matched", "start": 1.0, "end": 2.0, "duration": 1.0},
            {"id": "unmatched", "start": 4.0, "end": 5.0, "duration": 1.0},
        ],
        "mappings": [{
            "enabled": True,
            "alignment_mode": "speech_window_snap",
            "alignment_source_window_ids": ["matched"],
        }],
    }

    assert _speech_mute_regions(schedule) == [
        {"start": 1.0, "duration": 1.0},
        {"start": 4.0, "duration": 1.0},
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


