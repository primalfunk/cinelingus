from pathlib import Path

from cinelingus.config import AppConfig
from cinelingus.reports import build_run_report, write_report_files


def test_write_report_files_creates_json_txt_and_csv(tmp_path: Path) -> None:
    config = AppConfig(
        root=tmp_path,
        destination_video=tmp_path / "destination.mp4",
        source_dialogue=tmp_path / "source.mp4",
        cache_dir=tmp_path / "cache",
        output_dir=tmp_path / "output",
        temp_dir=tmp_path / "temp",
        speech_backend="whisper",
        transcription_mode="fast_preview",
        whisper_model="tiny",
        whisper_language=None,
        quality_modes={},
        filter_min_duration=0.35,
        filter_max_duration=12.0,
        filter_min_confidence=0.2,
        filter_min_chars_per_second=0.5,
        filter_max_chars_per_second=28.0,
        filter_repeated_text_window=4,
        silence_noise_db=-35,
        silence_min_duration=0.35,
        min_speech_duration=0.25,
        merge_gap=0.25,
        max_time_stretch=0.1,
        scheduling_mode="strict_order",
        best_fit_lookahead=8,
        render_sample_rate=48000,
        render_channels=2,
        target_lufs=-18.0,
        audio_fade_duration=0.015,
        original_duck_db=-28.0,
        visual_scene_threshold=0.35,
        visual_min_shot_duration=0.5,
        shot_boundary_mode="soft",
        cinematic_filter="balanced",
        enable_speaker_awareness=True,
        speaker_diarization_backend="heuristic",
        speaker_diarization_model="pyannote/speaker-diarization-community-1",
        speaker_diarization_device="auto",
        prefer_high_speaker_confidence=True,
        prefer_clean_dialogue_timing=True,
        prefer_funny_or_surprising_matches=True,
    )
    audio = config.output_dir / "replacement_dialogue.wav"
    video = config.output_dir / "translation_output.mp4"
    audio.parent.mkdir(parents=True)
    audio.write_text("audio")
    video.write_text("video")
    schedule = {
        "scheduling_mode": "strict_order",
        "transformation_name": "translation",
        "transformation_history": [{"verb": "select", "description": "select test", "inputs": ["a"], "outputs": ["b"]}],
        "mappings": [
            {
                "window_id": "w1",
                "clip_id": "c1",
                "enabled": True,
                "destination_timestamp": 1.0,
                "clip_trim_duration": 2.0,
                "planned_render_duration": 2.0,
                "stretch_factor": 1.0,
                "score": 0.75,
                "selection_reason": "next_source_clip_in_order",
                "timing_strategy": "pad_trailing_silence",
                "skipped_source_clips": 0,
                "source_transcript": "hello",
                "alignment_mode": "speech_window_snap",
                "alignment_slot_start": 1.0,
                "alignment_slot_end": 3.0,
                "alignment_spillover_seconds": 0.0,
                "source_speaker_id": "speaker_001",
                "destination_speaker_id": "speaker_001",
                "speaker_match_preserved": True,
                "mapped_destination_speaker_id": "speaker_001",
                "speaker_mapping_followed": True,
            },
            {
                "window_id": "w2",
                "clip_id": "c2",
                "enabled": False,
                "destination_timestamp": 3.0,
                "clip_trim_duration": 1.0,
                "planned_render_duration": 1.0,
                "stretch_factor": 1.0,
                "score": 0.25,
                "selection_reason": "next_source_clip_in_order",
                "timing_strategy": "pad_trailing_silence",
                "skipped_source_clips": 0,
                "source_transcript": "disabled",
                "alignment_mode": "performance_fill_fallback",
            }
        ],
    }
    report = build_run_report(
        config=config,
        source_hash="sourcehash",
        destination_hash="desthash",
        destination_movie={"duration": 10.0, "resolution": "640x480"},
        source_movie={"duration": 8.0, "resolution": "640x480"},
        source_events={"events": [{"id": "e1", "speaker_id": "speaker_001", "duration": 1.0}, {"id": "e2", "speaker_id": "speaker_002", "duration": 2.0}]},
        filtered_source_events={"events": [{"id": "e1"}], "filter_stats": {"usable_count": 1, "rejected_count": 1}},
        clip_library={"clips": [{"id": "c1"}]},
        destination_timeline={"windows": [{"id": "w1", "speaker_id": "speaker_001", "duration": 2.0}, {"id": "w2", "speaker_id": "speaker_002", "duration": 1.0}]},
        filtered_destination_timeline={"windows": [{"id": "w1"}, {"id": "w2"}], "filter_stats": {"usable_count": 2, "rejected_count": 0}},
        schedule=schedule,
        problem_region_report={
            "problem_count": 2,
            "summary": {
                "fallback_mapping_count": 1,
                "underfilled_performance_count": 1,
                "low_fit_mapping_count": 0,
            },
        },
        editorial_highlights={
            "summary": {
                "evaluated_performances": 3,
                "average_editorial_score": 0.67,
                "positive_highlight_count": 4,
                "needs_review_count": 1,
            },
            "highlights": {
                "most_convincing": [{"performance_id": "p1"}],
                "funniest": [{"performance_id": "p2"}],
                "most_awkward": [],
            },
        },
        source_speaker_map={
            "diagnostics": {
                "status": "fallback",
                "effective_backend": "heuristic_timing_v1",
                "fallback_used": True,
                "fallback_reason": "pyannote diarization produced no usable speaker segments; fell back to heuristic speaker labels",
                "labeled_item_count": 2,
                "speech_item_count": 2,
            },
            "warnings": ["pyannote diarization produced no usable speaker segments; fell back to heuristic speaker labels"],
        },
        destination_speaker_map={
            "diagnostics": {
                "status": "strong",
                "effective_backend": "pyannote.audio",
                "fallback_used": False,
                "labeled_item_count": 2,
                "speech_item_count": 2,
            },
            "warnings": [],
        },
        audio_output=audio,
        video_output=video,
    )

    paths = write_report_files(report, schedule, config.output_dir)

    assert paths["json"].exists()
    assert paths["txt"].exists()
    assert paths["csv"].exists()
    report_text = paths["txt"].read_text()
    csv_text = paths["csv"].read_text()
    assert "scheduled mappings: 2" in report_text
    assert "enabled mappings: 1" in report_text
    assert "name: translation" in report_text
    assert "CIR index:" in report_text
    assert "disabled mappings: 1" in report_text
    assert "shot boundary mode: soft" in report_text
    assert "speech-snapped mappings: 1" in report_text
    assert "unique muted speech spans: 1" in report_text
    assert "Problem Regions" in report_text
    assert "Soundtrack Bed" in report_text
    assert "Speakers" in report_text
    assert "source speakers: 2" in report_text
    assert "source speaker status: fallback via heuristic_timing_v1" in report_text
    assert "destination speaker status: strong via pyannote.audio" in report_text
    assert "speaker-map placements: 1 / 1" in report_text
    assert "duck dB: -28.0" in report_text
    assert "total problems: 2" in report_text
    assert "Editorial Highlights" in report_text
    assert "evaluated performances: 3" in report_text
    assert "editorial_highlights.json" in report["outputs"]["editorial_highlights"]
    assert "taste_profile.json" in report["outputs"]["taste_profile"]
    assert "problem_regions.json" in report["outputs"]["problem_regions"]
    assert "window_id,clip_id,enabled" in csv_text
    assert report["counts"]["skipped_windows"] == 1
    assert report["counts"]["enabled_mappings"] == 1
    assert report["counts"]["disabled_mappings"] == 1
    assert report["schedule"]["average_score"] == 0.75
    assert report["schedule"]["alignment"]["speech_snapped_mappings"] == 1
    assert report["schedule"]["alignment"]["unique_speech_slots"] == 1
    assert report["schedule"]["alignment"]["merged_duck_region_count"] == 1
    assert report["soundtrack_bed"]["continuous_original_bed"] is True
    assert report["soundtrack_bed"]["duck_db"] == -28.0
    assert report["speakers"]["source_dialogue"]["speaker_count"] == 2
    assert report["speakers"]["source_diagnostics"]["fallback_used"] is True
    assert "source speaker diarization used fallback" in report["warnings"][-1]
    assert report["speakers"]["schedule_preservation"]["same_speaker_count"] == 1
    assert report["speakers"]["speaker_mapping"]["speaker_mapping_followed_count"] == 1

