import pytest

from movie_masher.transformation_verbs import (
    compress_duration,
    movie_masher_transformation_plan,
    self_shuffle_transformation_plan,
    remove_disabled_mappings,
    repeat_selection,
    select_dialogue_clips,
    select_speaking_windows,
    shuffle_selection,
    stretch_duration,
)


def test_movie_masher_plan_uses_core_verbs() -> None:
    verbs = [step["verb"] for step in movie_masher_transformation_plan()]
    assert verbs == ["select", "select", "place", "replace", "render"]


def test_self_shuffle_plan_uses_single_film_verbs() -> None:
    verbs = [step["verb"] for step in self_shuffle_transformation_plan()]
    assert verbs == ["select", "shuffle", "select", "place", "render"]


def test_select_primitives_filter_unusable_objects_without_mutating() -> None:
    clips = [
        {"id": "c1", "duration": 1.0, "usable": True},
        {"id": "c2", "duration": 0.0, "usable": True},
        {"id": "c3", "duration": 1.0, "usable": False},
    ]
    windows = [
        {"id": "w1", "duration": 1.0, "usable": True},
        {"id": "w2", "duration": -1.0, "usable": True},
    ]

    selected_clips = select_dialogue_clips(clips)
    selected_windows = select_speaking_windows(windows)
    selected_clips[0]["id"] = "changed"

    assert [clip["id"] for clip in selected_clips] == ["changed"]
    assert clips[0]["id"] == "c1"
    assert [window["id"] for window in selected_windows] == ["w1"]


def test_remove_repeat_shuffle_stretch_and_compress() -> None:
    mappings = [{"id": "a", "enabled": True}, {"id": "b", "enabled": False}]
    assert [item["id"] for item in remove_disabled_mappings(mappings)] == ["a"]
    assert len(repeat_selection([{"id": "x"}], times=3)) == 3
    assert sorted(item["id"] for item in shuffle_selection([{"id": "a"}, {"id": "b"}], seed=1)) == ["a", "b"]
    assert stretch_duration(2.0, factor=1.5) == 3.0
    assert compress_duration(3.0, factor=2.0) == 1.5
    with pytest.raises(ValueError):
        compress_duration(3.0, factor=0)
