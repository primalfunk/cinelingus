from movie_masher.gui import (
    compact_path,
    completed_run_truth_summary,
    finished_output_folder,
    completion_summary,
    diarization_chunk_progress,
    accessible_control_labels,
    emblem_asset_path,
    emblem_variant_asset_path,
    format_clock_duration,
    heartbeat_stage_message,
    heartbeat_stage_progress,
    highlight_row_tags,
    highlight_row_values,
    highlight_rows,
    plain_status_for_log_line,
    quality_detail,
    quality_preset_label,
    quality_preset_mode,
    quality_runtime_warning,
    required_input_fields,
    reported_output_duration,
    responsive_layout,
    should_emit_console_heartbeat,
    speaker_diarization_detail,
    run_truth_summary,
    single_film_input_needs_explicit_choice,
    summarize_output_dir,
    stage_key_for_log_line,
    stage_sequence_key,
    summarize_whisper_model_used,
    workflow_uses_target_length,
    open_path_or_reveal,
)
from movie_masher.util import write_json


def test_open_path_reveals_file_when_windows_player_launch_fails(tmp_path, monkeypatch):
    movie = tmp_path / "finished.mp4"
    movie.write_bytes(b"mp4")
    calls = []
    monkeypatch.setattr("movie_masher.gui.os.startfile", lambda _path: (_ for _ in ()).throw(OSError("broken association")))
    monkeypatch.setattr("movie_masher.gui.subprocess.Popen", lambda command: calls.append(command))

    result = open_path_or_reveal(movie)

    assert result == "revealed"
    assert calls == [["explorer.exe", "/select,", str(movie)]]


def test_summarize_output_dir_reports_problem_preview_status(tmp_path):
    output = tmp_path / "output"
    preview_dir = output / "previews" / "problem_regions"
    preview_dir.mkdir(parents=True)
    write_json(
        output / "problem_regions.json",
        {
            "problem_count": 3,
            "summary": {"fallback_mapping_count": 1, "undercovered_speech_window_count": 2},
            "problems": [],
        },
    )
    write_json(preview_dir / "problem_region_previews.json", {"preview_count": 3, "previews": []})

    summary = summarize_output_dir(output)

    assert summary["problem_count"] == 3
    assert summary["fallback_count"] == 1
    assert summary["undercovered_count"] == 2
    assert summary["preview_count"] == 3
    assert summary["preview_dir"] == str(preview_dir)
    assert "3 preview clip(s) ready" in summary["message"]


def test_summarize_output_dir_handles_clean_report(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    write_json(output / "problem_regions.json", {"problem_count": 0, "summary": {}, "problems": []})

    summary = summarize_output_dir(output)

    assert summary["problem_count"] == 0
    assert summary["message"] == "No problem regions reported."


def test_summarize_output_dir_prefers_editorial_highlights(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    write_json(
        output / "editorial_highlights.json",
        {
            "summary": {"needs_review_count": 2},
            "highlights": {
                "most_convincing": [{"performance_id": "p1"}],
                "funniest": [{"performance_id": "p2"}, {"performance_id": "p3"}],
                "most_awkward": [{"performance_id": "p4"}],
            },
        },
    )
    write_json(output / "problem_regions.json", {"problem_count": 4, "summary": {}, "problems": []})

    summary = summarize_output_dir(output)

    assert summary["highlight_count"] == 4
    assert summary["needs_attention_count"] == 2
    assert summary["problem_count"] == 4
    assert summary["message"].startswith("Highlights ready")
    assert "2 performance(s) need review" in summary["message"]


def test_highlight_rows_filters_and_formats_buckets():
    highlights = {
        "highlights": {
            "most_convincing": [
                {
                    "performance_id": "p1",
                    "start": 1.2345,
                    "duration": 2.3456,
                    "editorial_score": 0.81234,
                    "editorial_label": "Convincing",
                    "component": "believability",
                    "review_status": "unreviewed",
                    "mapping_indices": [0],
                }
            ],
            "needs_attention": [
                {
                    "performance_id": "p2",
                    "start": None,
                    "duration": None,
                    "editorial_score": 0.2,
                    "editorial_label": "Needs Review",
                    "component": "review_priority",
                    "review_status": "needs_revision",
                    "mapping_indices": [3],
                }
            ],
        }
    }

    rows = highlight_rows(highlights)
    needs = highlight_rows(highlights, "needs_attention")

    assert [row["performance_id"] for row in rows] == ["p1", "p2"]
    assert [row["performance_id"] for row in needs] == ["p2"]
    assert highlight_row_values(rows[0]) == ("Most Convincing", "p1", 1.234, 2.346, 0.812, "Convincing", "believability", "unreviewed")
    assert highlight_row_tags(rows[0]) == ("convincing",)
    assert highlight_row_tags(needs[0]) == ("needs",)


def test_single_film_input_guard_rejects_unchanged_project_default(tmp_path):
    default = tmp_path / "source" / "movie_1.mp4"
    chosen = tmp_path / "chosen.mp4"

    assert single_film_input_needs_explicit_choice("Self Shuffle", default, default, selected_by_user=False) is True
    assert single_film_input_needs_explicit_choice("Self Shuffle", default, default, selected_by_user=True) is False
    assert single_film_input_needs_explicit_choice("Self Shuffle", chosen, default, selected_by_user=False) is False
    assert single_film_input_needs_explicit_choice("Movie Masher", default, default, selected_by_user=False) is False


def test_run_truth_summary_distinguishes_movie_masher_and_single_film(tmp_path):
    destination = tmp_path / "destination.mp4"
    source = tmp_path / "source.mp4"
    output = tmp_path / "output"

    movie_masher = run_truth_summary(
        transformation="Movie Masher",
        destination=destination,
        source=source,
        output_dir=output,
        quality="Balanced",
        matching_style="Balanced",
    )
    single = run_truth_summary(
        transformation="Self Shuffle",
        destination=destination,
        source=source,
        output_dir=output,
        quality="Balanced",
        matching_style="Balanced",
    )

    assert "Experiment: Movie Masher" in movie_masher
    assert "Anchor Film:" in movie_masher
    assert "Film B:" in movie_masher
    assert "Film:" in single
    assert "Source dialogue:" not in single
    assert "Previous observations are reused only when material and settings match" in single


def test_completed_run_truth_summary_prefers_mutation_report(tmp_path):
    output = tmp_path / "output" / "mutations" / "self_shuffle" / "self_shuffle_output.mp4"
    output.parent.mkdir(parents=True)
    write_json(
        output.parent / "mutation_report.json",
        {
            "source_film": str(tmp_path / "chosen-film.mp4"),
            "mutation_filter": {"display_name": "Self Shuffle"},
            "outputs": {"video": str(output)},
        },
    )

    summary = completed_run_truth_summary(output, tmp_path / "output", "Self Shuffle")

    assert "Self Shuffle" in summary
    assert "chosen-film.mp4" in summary
    assert "self_shuffle_output.mp4" in summary


def test_completed_run_truth_summary_prefers_run_report_for_movie_masher(tmp_path):
    output = tmp_path / "output" / "movie_masher_output.mp4"
    output.parent.mkdir(parents=True)
    write_json(
        output.parent / "run_report.json",
        {
            "inputs": {
                "destination_video": {"path": str(tmp_path / "dest.mp4")},
                "source_dialogue": {"path": str(tmp_path / "source.mp4")},
            },
            "outputs": {"video_path": str(output)},
        },
    )

    summary = completed_run_truth_summary(output, output.parent, "Movie Masher")

    assert "Movie Masher" in summary
    assert "dest.mp4" in summary
    assert "source.mp4" in summary
    assert "movie_masher_output.mp4" in summary


def test_compact_path_keeps_filename_visible(tmp_path):
    long_path = tmp_path / "very" / "long" / "folder" / "name" / "selected-film-with-readable-name.mp4"

    compact = compact_path(long_path, max_chars=70)

    assert compact.startswith("...")
    assert compact.endswith("selected-film-with-readable-name.mp4")


def test_required_input_fields_are_transformation_specific():
    assert required_input_fields("Movie Masher") == ("anchor_film", "film_2", "output")
    assert required_input_fields("Self Shuffle") == ("anchor_film", "output")
    assert required_input_fields("Echo") == ("anchor_film", "output")




def test_finished_output_folder_prefers_actual_render_location(tmp_path):
    output = tmp_path / "output" / "best_short" / "movie.mp4"
    fallback = tmp_path / "output"

    assert finished_output_folder(output, fallback) == output.parent
    assert finished_output_folder(None, fallback) == fallback

def test_quality_preset_mapping_is_user_facing():
    assert quality_preset_mode("Fast Preview") == "fast_preview"
    assert quality_preset_mode("Balanced") == "balanced"
    assert quality_preset_mode("High Accuracy") == "quality"
    assert quality_preset_label("quality") == "Precision"
    assert quality_preset_mode("High Accuracy") == "quality"


def test_plain_status_for_log_line_hides_technical_language():
    assert plain_status_for_log_line("[time] transcribing destination timeline with Whisper") == "Examining recurring voices"
    assert plain_status_for_log_line("muxing final video") == "Completing the cinematic artifact"


def test_completion_summary_includes_quality_model_and_output(tmp_path):
    output = tmp_path / "movie.mp4"
    summary = completion_summary(
        output=output,
        output_dir=tmp_path,
        transformation="Movie Masher",
        quality_preset="Balanced",
        whisper_model="small",
        started_at=None,
    )

    assert "Mode: Movie Masher" in summary
    assert "movie.mp4" in summary
    assert "Fidelity: Balanced" in summary
    assert "Whisper" not in summary


def test_summarize_whisper_model_used_reports_fallback_warning(tmp_path):
    write_json(
        tmp_path / "run_report.json",
        {
            "config": {"whisper_model": "medium"},
            "source_events": {
                "whisper_model": "small",
                "whisper_model_warning": "Requested Whisper model 'medium' could not be used; fell back to 'small'.",
            },
        },
    )

    model, warning = summarize_whisper_model_used(tmp_path, "medium")

    assert model == "small"
    assert "fell back" in warning


def test_heartbeat_stage_message_reports_silent_long_running_stage():
    assert heartbeat_stage_message(stage="Finding spoken dialogue...", idle_seconds=5) == "Finding spoken dialogue..."
    message = heartbeat_stage_message(stage="Finding spoken dialogue...", idle_seconds=65)
    assert "No new observations" in message
    assert "01:05" in message


def test_heartbeat_stage_progress_pulses_without_claiming_completion():
    assert heartbeat_stage_progress(0) == 10.0
    assert 10.0 <= heartbeat_stage_progress(999) < 90.0


def test_diarization_chunks_report_real_fractional_progress():
    assert diarization_chunk_progress("diarizing_destination_chunk_26_of_249") == (26, 249)
    assert diarization_chunk_progress("validating_destination_speakers") is None


def test_detailed_stages_map_to_compact_progress_milestones():
    assert stage_sequence_key("clips") == "source_dialogue"
    assert stage_sequence_key("render_video") == "render_audio"
    assert stage_sequence_key("finalize") == "finalize"


def test_quality_detail_names_purpose_and_model():
    assert quality_detail("Balanced") == "A measured balance of speed and fidelity."
    assert quality_detail("High Accuracy") == "A more exacting examination for final work. This examination may require substantially more time."


def test_quality_runtime_warning_flags_high_accuracy_on_cpu():
    assert quality_runtime_warning("Balanced", {"available": True, "cuda_available": False}) is None
    warning = quality_runtime_warning("High Accuracy", {"available": True, "cuda_available": False})
    assert warning is not None
    assert "without accelerated examination" in warning
    assert "Whisper" not in warning
    assert quality_runtime_warning("High Accuracy", {"available": True, "cuda_available": True}) is None


def test_console_heartbeat_emission_is_disabled_to_prevent_journal_flooding():
    assert should_emit_console_heartbeat(idle_seconds=10, now=100.0, last_heartbeat_at=0.0) is False
    assert should_emit_console_heartbeat(idle_seconds=31, now=100.0, last_heartbeat_at=80.0) is False
    assert should_emit_console_heartbeat(idle_seconds=31, now=100.0, last_heartbeat_at=60.0) is False


def test_speaker_diarization_detail_is_user_facing():
    class Config:
        enable_speaker_awareness = True
        speaker_diarization_backend = "heuristic"

    assert speaker_diarization_detail(Config()) == "Recurring voices will be estimated from temporal structure."


def test_stage_key_for_log_line_drives_progress_checklist():
    assert stage_key_for_log_line("transcribing source dialogue with Whisper") == "source_dialogue"
    assert stage_key_for_log_line("rendered video: out.mp4") == "render_video"


def test_completed_run_truth_summary_includes_short_remix_audio_provenance(tmp_path):
    output = tmp_path / "output" / "best_short" / "FINAL_movie_masher_best_short.mp4"
    audio = tmp_path / "output" / "best_short" / "movie_masher_best_short.wav"
    output.parent.mkdir(parents=True)
    write_json(
        output.parent / "output_report.json",
        {
            "outputs": {"video": str(output), "audio": str(audio)},
            "selection_summary": {"candidate_id": "candidate_001"},
            "audio_provenance": {
                "status": "pass",
                "inputs": {
                    "destination_video": str(tmp_path / "visual.mp4"),
                    "source_dialogue": str(tmp_path / "dialogue.mp4"),
                },
            },
        },
    )

    summary = completed_run_truth_summary(output, tmp_path / "output", "Movie Masher")

    assert "FINAL_movie_masher_best_short.mp4" in summary
    assert "visual.mp4" in summary
    assert "dialogue.mp4" in summary
    assert "Audio check: pass" in summary


def test_completed_run_truth_summary_includes_dialogue_reel_vignette_count(tmp_path):
    output = tmp_path / "output" / "best_short" / "runs" / "run" / "FINAL_reel.mp4"
    output.parent.mkdir(parents=True)
    write_json(
        output.parent / "output_report.json",
        {
            "outputs": {"video": str(output), "audio": str(output.with_suffix(".wav"))},
            "selection_summary": {"candidate_id": "scene_pair_001", "vignette_count": 3},
        },
    )

    summary = completed_run_truth_summary(output, tmp_path / "output", "Movie Masher")

    assert "Vignettes: 3" in summary
    assert "Candidate: scene_pair_001" in summary


def test_responsive_layout_hides_hero_before_controls_become_cramped():
    assert responsive_layout(1120) == "wide"
    assert responsive_layout(820) == "compact"


def test_target_length_only_applies_to_short_remixes():
    assert workflow_uses_target_length("Best Short Remix") is True
    assert workflow_uses_target_length("Full Movie Remix") is False


def test_missing_emblem_uses_documented_optional_asset_path(tmp_path):
    path = emblem_asset_path(tmp_path)

    assert path == tmp_path / "assets" / "cinelingus_emblem.png"
    assert not path.exists()


def test_emblem_variants_have_stable_role_specific_paths(tmp_path):
    assert emblem_variant_asset_path(tmp_path, compact=True) == tmp_path / "assets" / "cinelingus_emblem_header.png"
    assert emblem_variant_asset_path(tmp_path, compact=False) == tmp_path / "assets" / "cinelingus_emblem_hero.png"


def test_accessible_labels_remain_plain_and_explicit():
    labels = accessible_control_labels()

    assert labels["begin"] == "Start Cinelingus processing job"
    assert labels["technical_record"] == "Show exact technical processing log"


def test_completion_summary_reports_duration_and_fallback_truthfully(tmp_path):
    output = tmp_path / "run" / "artifact.mp4"
    output.parent.mkdir()
    output.write_bytes(b"mp4")
    write_json(output.parent / "output_report.json", {"actual_duration": 297.25})

    summary = completion_summary(
        output=output,
        output_dir=tmp_path,
        transformation="Movie Masher",
        quality_preset="Balanced",
        whisper_model="small",
        started_at=None,
        model_warning="backend fallback",
    )

    assert reported_output_duration(output, tmp_path) == 297.25
    assert "Final duration: 04:57" in summary
    assert "alternate method was used" in summary
    assert "Whisper" not in summary


def test_operator_clock_uses_stable_in_place_format():
    assert format_clock_duration(30) == "00:30"
    assert format_clock_duration(494) == "08:14"
