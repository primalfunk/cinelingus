from movie_masher.intervals import covered_speech_duration


def test_coverage_uses_interval_union_and_target_intersection() -> None:
    mappings = [
        {"destination_timestamp": 1.0, "planned_render_duration": 3.0},
        {"destination_timestamp": 2.0, "planned_render_duration": 3.0},
    ]
    windows = [{"start": 0.0, "end": 4.0}]
    assert covered_speech_duration(mappings, windows) == 3.0
