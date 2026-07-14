from movie_masher.progress import ProgressState, format_progress_status


def test_progress_state_formats_percent_and_eta():
    state = ProgressState.start("speech", "Speech detection", total=10)
    state.started_at -= 10
    state.update(current=5)

    message = format_progress_status(state)

    assert "Speech detection" in message
    assert "50% complete" in message
    assert "Remaining:" in message


def test_progress_state_handles_unknown_eta():
    state = ProgressState.start("load", "Source loading")

    assert "Estimating" in format_progress_status(state)



def test_progress_update_can_change_stage_label():
    state = ProgressState.start("old", "Old", total=2)
    state.update(current=1, stage_id="new", stage_label="New", status_message="working")

    assert state.stage_id == "new"
    assert "New" in format_progress_status(state)
    assert "working" in format_progress_status(state)
