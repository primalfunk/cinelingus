from cinelingus.problem_report import build_problem_region_report


def test_problem_region_report_lists_fallback_and_underfilled(tmp_path):
    schedule = {
        "destination_performance_fills": [
            {
                "destination_performance_id": "p1",
                "start": 10.0,
                "duration": 5.0,
                "coverage": 0.4,
                "target_coverage": 0.9,
                "coverage_basis": "speech_windows",
                "speech_window_count": 3,
                "covered_speech_window_count": 1,
                "uncovered_speech_window_count": 2,
                "stop_reason": "remaining_gap_has_no_fitting_whole_line",
            }
        ],
        "mappings": [
            {
                "enabled": True,
                "window_id": "p1",
                "destination_performance_id": "p1",
                "clip_id": "c1",
                "destination_timestamp": 10.0,
                "planned_render_duration": 1.0,
                "alignment_mode": "performance_fill_fallback",
                "score": 0.8,
                "visual_fit_score": 0.9,
                "source_transcript": "hello",
            },
            {
                "enabled": True,
                "window_id": "p2",
                "destination_performance_id": "p2",
                "clip_id": "c2",
                "destination_timestamp": 20.0,
                "planned_render_duration": 1.0,
                "alignment_mode": "speech_window_snap",
                "score": 0.3,
                "visual_fit_score": 0.7,
            },
        ],
    }

    report = build_problem_region_report(
        schedule=schedule,
        output_json=tmp_path / "problem_regions.json",
        output_csv=tmp_path / "problem_regions.csv",
        output_txt=tmp_path / "problem_regions.txt",
    )

    assert report["summary"]["fallback_mapping_count"] == 1
    assert report["summary"]["underfilled_performance_count"] == 1
    assert report["summary"]["uncovered_speech_performance_count"] == 1
    assert report["summary"]["low_fit_mapping_count"] == 1
    assert (tmp_path / "problem_regions.json").exists()
    assert "fallback mappings: 1" in (tmp_path / "problem_regions.txt").read_text()
    assert "preview --mapping 0" in (tmp_path / "problem_regions.csv").read_text()


def test_problem_region_report_flags_undercovered_speech_slots(tmp_path):
    schedule = {
        "destination_performance_fills": [
            {
                "destination_performance_id": "p1",
                "start": 10.0,
                "duration": 4.0,
                "coverage": 1.0,
                "target_coverage": 0.9,
                "coverage_basis": "speech_windows",
                "speech_windows": [
                    {"id": "w1", "start": 10.0, "end": 12.0, "duration": 2.0},
                    {"id": "w2", "start": 12.0, "end": 14.0, "duration": 2.0},
                ],
                "speech_window_count": 2,
                "covered_speech_window_count": 2,
                "uncovered_speech_window_count": 0,
                "stop_reason": "target_coverage_met",
            }
        ],
        "mappings": [
            {
                "enabled": True,
                "window_id": "p1",
                "destination_performance_id": "p1",
                "clip_id": "c1",
                "destination_timestamp": 10.0,
                "planned_render_duration": 2.0,
                "alignment_mode": "speech_window_snap",
                "alignment_source_window_ids": ["w1"],
                "score": 0.8,
                "visual_fit_score": 0.9,
            }
        ],
    }

    report = build_problem_region_report(
        schedule=schedule,
        output_json=tmp_path / "problem_regions.json",
        output_csv=tmp_path / "problem_regions.csv",
        output_txt=tmp_path / "problem_regions.txt",
    )

    assert report["summary"]["underfilled_performance_count"] == 0
    assert report["summary"]["undercovered_speech_window_count"] == 1
    assert report["problems"][0]["problem_type"] == "undercovered_speech_window"
    assert report["problems"][0]["window_id"] == "w2"
    assert report["problems"][0]["coverage"] == 0.0
    assert "undercovered speech windows: 1" in (tmp_path / "problem_regions.txt").read_text()


def test_problem_report_prioritizes_residue_ambience_and_boundary_review(tmp_path):
    schedule = {
        "mappings": [],
        "destination_performance_fills": [],
        "destination_speech_regions": [{
            "id": "speech", "start": 4.0, "end": 5.0, "duration": 1.0,
        }],
        "voice_residue_verification": {
            "status": "POSSIBLE_DESTINATION_SPEECH_DETECTED",
            "regions": [{
                "destination_region_id": "speech", "start": 4.0, "end": 5.0,
                "possible_residue": True, "destination_similarity": 0.9,
                "donor_similarity": 0.2, "rendered_transcript": "original words",
            }],
        },
        "background_reconstruction_report": {
            "silence_fallback_targets": [{"start": 7.0, "end": 8.0}],
        },
        "suppression_padding_report": {
            "regions": [{
                "speech_region_id": "speech", "confidence": 0.4,
                "source_kind": "recovered_filtered_speech_window",
                "leading_padding": 0.12, "trailing_padding": 0.2,
            }],
        },
    }

    report = build_problem_region_report(
        schedule=schedule,
        output_json=tmp_path / "problem_regions.json",
        output_csv=tmp_path / "problem_regions.csv",
        output_txt=tmp_path / "problem_regions.txt",
    )

    assert report["problems"][0]["problem_type"] == "possible_destination_speech_residue"
    assert report["problems"][0]["severity"] == "critical"
    assert report["summary"]["possible_residue_count"] == 1
    assert report["summary"]["ambience_silence_fallback_count"] == 1
    assert report["summary"]["uncertain_speech_boundary_count"] == 1
