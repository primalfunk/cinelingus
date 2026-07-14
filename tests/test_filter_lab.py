from __future__ import annotations

from pathlib import Path

import pytest

from movie_masher.filter_lab import FilterRecipe, default_filter_registry, load_recipe, save_recipe
from movie_masher.filter_lab.gui_controller import sync_filter_family
from movie_masher.filter_lab.integration import build_strategy_schedule
from movie_masher.filter_lab.plan import plan_from_schedule, write_filter_plan
from movie_masher.filter_lab.strategies import (
    build_bloom_schedule,
    build_chorus_schedule,
    build_contagion_schedule,
    build_doppelganger_schedule,
    build_flashback_schedule,
    build_foreshadow_schedule,
    build_possession_schedule,
    build_spiral_schedule,
    representative_preview_regions,
)
from movie_masher.validation import validate_artifact


def clip(clip_id: str, speaker: str, start: float, duration: float = 2.0) -> dict:
    return {"id": clip_id, "speaker_id": speaker, "movie_timestamp": start, "duration": duration, "path": f"{clip_id}.wav", "transcript": clip_id}


def window(window_id: str, speaker: str, start: float, duration: float = 2.0, scene: str | None = None) -> dict:
    return {"id": window_id, "speaker_id": speaker, "start": start, "duration": duration, "scene_id": scene}


def test_registry_contains_all_families_and_only_real_filters_are_runnable() -> None:
    registry = default_filter_registry()

    assert [family.id for family in registry.families()] == ["translation", "infection", "identity", "memory", "emotion", "time", "experimental"]
    assert {item.id for item in registry.definitions(implemented_only=True)} == {
        "translation.self_shuffle", "translation.echo", "translation.movie_masher", "translation.drift",
        "identity.possession", "identity.doppelganger", "identity.chorus",
        "time.foreshadow", "time.flashback", "time.spiral", "infection.contagion", "experimental.bloom",
    }
    assert registry.get("movie_masher").id == "translation.movie_masher"
    assert registry.get("memory.dream").implemented is False
    with pytest.raises(ValueError, match="in development"):
        registry.validate_stack(["memory.dream"])


def test_family_change_defaults_to_an_implemented_filter(monkeypatch) -> None:
    class Variable:
        def __init__(self, value: str):
            self.value = value

        def get(self) -> str:
            return self.value

        def set(self, value: str) -> None:
            self.value = value

    class Box:
        def configure(self, **kwargs) -> None:
            self.values = kwargs["values"]

    app = type("App", (), {})()
    app.family_var = Variable("Infection")
    app.mode_var = Variable("Mutation")
    app.mode_box = Box()
    monkeypatch.setattr("movie_masher.filter_lab.gui_controller.sync_filter_mode", lambda _app: None)

    sync_filter_family(app)

    assert app.mode_var.get() == "Contagion"
    assert "Mutation" in app.mode_box.values


def test_recipe_round_trip_migrates_legacy_id_and_warns_on_version(tmp_path: Path) -> None:
    recipe = FilterRecipe.create("possession", input_media_roles={"film": "film.mp4"}, parameters={"possessing_speaker": "s1", "possessed_speaker": "s2"})
    path = save_recipe(recipe, tmp_path / "filter_recipe.json")
    validate_artifact("filter_recipe", path, Path.cwd() / "schemas")
    data = recipe.to_dict()
    data["filter_id"] = "possession"
    data["filter_version"] = "0.9.0"
    import json
    path.write_text(json.dumps(data), encoding="utf-8")

    loaded = load_recipe(path)

    assert loaded.recipe.filter_id == "identity.possession"
    assert loaded.migrations
    assert "0.9.0" in loaded.warnings[0]


def test_strategy_dispatcher_returns_the_built_schedule() -> None:
    params = default_filter_registry().get("possession").parameter_defaults | {
        "possessing_speaker": "s1",
        "possessed_speaker": "s2",
        "intensity": "Total",
    }

    schedule = build_strategy_schedule("possession", clips=[clip("a", "s1", 10.0), clip("b", "s1", 20.0)], windows=[window("w", "s2", 100.0)], duration=120.0, parameters=params)

    assert schedule["mutation_id"] == "possession"
    assert schedule["filter_progress_stages"]


def test_possession_is_stable_and_speaker_specific(tmp_path: Path) -> None:
    clips = [clip(f"a{i}", "s1", i * 30.0) for i in range(5)] + [clip("b1", "s2", 10.0)]
    windows = [window(f"w{i}", "s2", 200.0 + i * 10.0) for i in range(4)] + [window("other", "s1", 250.0)]
    params = default_filter_registry().get("possession").parameter_defaults | {"possessing_speaker": "s1", "possessed_speaker": "s2", "intensity": "Total"}

    schedule = build_possession_schedule(clips=clips, windows=windows, duration=300.0, parameters=params, seed=9)
    again = build_possession_schedule(clips=clips, windows=windows, duration=300.0, parameters=params, seed=9)

    assert schedule["mappings"] == again["mappings"]
    assert all(row["source_speaker_id"] == "s1" for row in schedule["mappings"])
    assert all(row["destination_speaker_id"] == "s2" for row in schedule["mappings"])
    assert schedule["filter_validation"]["passed"] is True
    plan = plan_from_schedule(definition=default_filter_registry().get("possession"), schedule=schedule, seed=9)
    plan_path = write_filter_plan(plan, tmp_path / "filter_plan.json")
    validate_artifact("filter_plan", plan_path, Path.cwd() / "schemas")


def test_foreshadow_never_uses_earlier_dialogue() -> None:
    clips = [clip("f1", "s1", 80.0), clip("f2", "s1", 130.0), clip("f3", "s2", 180.0)]
    windows = [window("w1", "s1", 10.0), window("w2", "s2", 40.0), window("w3", "s1", 90.0)]
    params = default_filter_registry().get("foreshadow").parameter_defaults | {"minimum_future_distance": 20.0, "maximum_future_distance": 200.0, "intensity": "Total"}

    schedule = build_foreshadow_schedule(clips=clips, windows=windows, duration=220.0, parameters=params, seed=2)

    assert schedule["mappings"]
    assert all(row["source_movie_timestamp"] > row["destination_timestamp"] + 20.0 for row in schedule["mappings"])
    assert schedule["filter_validation"]["future_only_rule"] is True


def test_contagion_requires_exposure_and_respects_maximum() -> None:
    clips = [clip(f"carrier{i}", "s1", 70.0 + i * 10.0) for i in range(6)] + [clip("other", "s2", 5.0)]
    windows = [
        window("origin", "s1", 5.0, scene="scene1"),
        window("exposure", "s2", 10.0, scene="scene1"),
        window("after1", "s2", 30.0, scene="scene2"),
        window("after2", "s2", 50.0, scene="scene3"),
    ]
    params = default_filter_registry().get("contagion").parameter_defaults | {"initial_carrier": "s1", "maximum_infected_speakers": 2, "intensity": "Total"}

    schedule = build_contagion_schedule(clips=clips, windows=windows, duration=120.0, parameters=params, seed=1)

    infected = schedule["filter_metrics"]["infection_timeline"]
    s2 = next(row for row in infected if row["speaker"] == "s2")
    assert s2["infection_time"] >= 0
    assert all(row["destination_timestamp"] >= row["infection_time"] for row in schedule["mappings"])
    assert schedule["filter_metrics"]["infected_speaker_count"] <= 2


def test_bloom_measurably_increases_and_preview_spans_progression() -> None:
    windows = [window(f"w{i}", "s1" if i % 2 else "s2", 5.0 + i * 10.0) for i in range(20)]
    clips = [clip(f"c{i}", "s2" if i % 2 else "s1", 195.0 - i * 9.0) for i in range(20)]
    params = default_filter_registry().get("bloom").parameter_defaults | {"starting_intensity": 0.2, "ending_intensity": 1.0}

    schedule = build_bloom_schedule(clips=clips, windows=windows, duration=210.0, parameters=params, seed=3)
    preview = representative_preview_regions("bloom", schedule)

    assert schedule["filter_validation"]["aggregate_strength_increases"] is True
    assert schedule["filter_metrics"]["bloom_profile"][-1]["average_transformation_score"] >= schedule["filter_metrics"]["bloom_profile"][0]["average_transformation_score"]
    assert len(preview) == 3
    assert preview[0]["start"] < preview[-1]["start"]


def test_flashback_uses_only_earlier_dialogue() -> None:
    clips = [clip("p1", "s1", 5.0), clip("p2", "s2", 35.0), clip("p3", "s1", 65.0)]
    windows = [window("w1", "s1", 70.0), window("w2", "s2", 110.0), window("w3", "s1", 150.0)]
    params = default_filter_registry().get("flashback").parameter_defaults | {
        "minimum_past_distance": 20.0, "maximum_past_distance": 200.0, "intensity": "Total",
    }

    schedule = build_flashback_schedule(clips=clips, windows=windows, duration=180.0, parameters=params, seed=4)

    assert schedule["mappings"]
    assert all(row["source_movie_timestamp"] < row["destination_timestamp"] - 20.0 for row in schedule["mappings"])
    assert schedule["filter_validation"]["all_sources_are_past_only"] is True


def test_spiral_displacement_never_decreases() -> None:
    clips = [clip(f"c{i}", "s1", float(i * 20)) for i in range(12)]
    windows = [window(f"w{i}", "s1", 90.0 + i * 8.0) for i in range(8)]
    params = default_filter_registry().get("spiral").parameter_defaults | {
        "starting_distance": 10.0, "maximum_distance": 120.0, "direction": "Past only", "intensity": "Total",
    }

    schedule = build_spiral_schedule(clips=clips, windows=windows, duration=220.0, parameters=params, seed=5)
    distances = [row["temporal_displacement"] for row in schedule["mappings"]]

    assert distances
    assert all(right >= left for left, right in zip(distances, distances[1:]))
    assert schedule["filter_validation"]["absolute_displacement_never_decreases"] is True


def test_doppelganger_keeps_one_bidirectional_pair() -> None:
    clips = [clip("a1", "a", 5.0), clip("a2", "a", 15.0), clip("b1", "b", 25.0), clip("b2", "b", 35.0)]
    windows = [window("wa1", "a", 60.0), window("wb1", "b", 70.0), window("wa2", "a", 80.0), window("wb2", "b", 90.0)]
    params = default_filter_registry().get("doppelganger").parameter_defaults | {
        "primary_speaker": "a", "mirror_speaker": "b", "intensity": "Total",
    }

    schedule = build_doppelganger_schedule(clips=clips, windows=windows, duration=120.0, parameters=params, seed=2)

    assert all({row["source_speaker_id"], row["destination_speaker_id"]} == {"a", "b"} for row in schedule["mappings"])
    assert schedule["filter_validation"]["all_mappings_remain_inside_pair"] is True


def test_chorus_uses_one_anchor_for_multiple_speakers() -> None:
    clips = [clip(f"a{i}", "anchor", i * 5.0) for i in range(6)]
    windows = [window("w1", "b", 50.0), window("w2", "c", 60.0), window("w3", "b", 70.0), window("w4", "c", 80.0)]
    params = default_filter_registry().get("chorus").parameter_defaults | {
        "anchor_speaker": "anchor", "maximum_chorus_speakers": 2, "intensity": "Total",
    }

    schedule = build_chorus_schedule(clips=clips, windows=windows, duration=100.0, parameters=params, seed=3)

    assert {row["source_speaker_id"] for row in schedule["mappings"]} == {"anchor"}
    assert {row["destination_speaker_id"] for row in schedule["mappings"]} == {"b", "c"}
    assert schedule["filter_validation"]["all_sources_match_anchor"] is True
