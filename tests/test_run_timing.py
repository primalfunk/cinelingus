from movie_masher.run_timing import completed_stage_text, estimate_overall_remaining


def test_overall_eta_uses_run_progress_not_current_stage_progress() -> None:
    assert estimate_overall_remaining(60.0, 25.0) == 180.0


def test_stage_completion_text_records_duration() -> None:
    assert completed_stage_text("Find dialogue", 65.0) == "[x] Find dialogue - 01:05"
