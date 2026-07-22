from __future__ import annotations

from pathlib import Path

import pytest

from cinelingus.filter_lab import FilterRecipe, default_contract_catalog, default_filter_registry, load_recipe, save_recipe
from cinelingus.filter_lab.gui_controller import sync_filter_family
from cinelingus.filter_lab.integration import build_strategy_schedule
from cinelingus.filter_lab.plan import plan_from_schedule, write_filter_plan
from cinelingus.filter_lab.strategies import (
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
from cinelingus.validation import validate_artifact


def clip(clip_id: str, speaker: str, start: float, duration: float = 2.0) -> dict:
    return {"id": clip_id, "speaker_id": speaker, "movie_timestamp": start, "duration": duration, "path": f"{clip_id}.wav", "transcript": clip_id}


def window(window_id: str, speaker: str, start: float, duration: float = 2.0, scene: str | None = None) -> dict:
    return {"id": window_id, "speaker_id": speaker, "start": start, "duration": duration, "scene_id": scene}


def test_registry_contains_all_families_and_only_real_filters_are_runnable() -> None:
    registry = default_filter_registry()

    assert [family.id for family in registry.families()] == ["translation", "infection", "identity", "memory", "emotion", "time", "experimental", "multiworld"]
    assert len(registry.definitions(implemented_only=True)) == 33
    assert {item.id for item in registry.definitions() if not item.implemented} == {
        "multiworld.bleed", "multiworld.chimera", "multiworld.civilization",
        "multiworld.doppelganger", "multiworld.mirror_world",
        "multiworld.parallel_universes", "multiworld.wormhole",
    }
    assert registry.get("translation").id == "multiworld.translation"
    assert registry.get("multiworld.translation").id == "multiworld.translation"
    assert registry.get("Translation").id == "multiworld.translation"
    assert registry.get("Translation").id == "multiworld.translation"
    assert registry.get_in_family("multiworld", "Translation").id == "multiworld.translation"
    assert registry.get_in_family("multiworld", "Translation").id == "multiworld.translation"
    assert registry.get("memory.dream").implemented is True
    assert registry.validate_stack(["memory.dream"]) == []


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
    monkeypatch.setattr("cinelingus.filter_lab.gui_controller.sync_filter_mode", lambda _app: None)

    sync_filter_family(app)

    assert app.mode_var.get() == "Contagion"
    assert "Mutation" in app.mode_box.values


def test_new_catalog_strategies_are_deterministic_and_validate_their_laws(tmp_path: Path) -> None:
    filter_ids = (
        "whisper", "mutation", "dialect", "split_personality", "dream", "recollection", "amnesia",
        "wonder", "regret", "optimist", "paranoia", "exhaustion", "mobius", "venom", "shed_skin", "ouroboros",
    )
    texts = (
        "wow look at the beautiful world", "I am sorry I made a mistake", "we can hope for better together",
        "they are watching someone behind us", "you liar I hate you", "remember the light",
        "why imagine the sky", "I wish I could", "good love is possible", "danger never trust them",
        "kill the threat", "we will know",
    )
    clips = [
        {"id": f"c{i}", "speaker_id": f"s{(i % 3) + 1}", "movie_timestamp": i * 20.0,
         "duration": 2.0 + (i % 4), "path": f"c{i}.wav", "transcript": text}
        for i, text in enumerate(texts)
    ]
    windows = [
        {"id": f"w{i}", "speaker_id": f"s{(i % 3) + 1}", "start": i * 20.0 + 10.0,
         "duration": 2.5 + (i % 3), "transcript": texts[(i + 1) % len(texts)]}
        for i in range(len(texts))
    ]
    parameters = {"intensity": "Total", "minimum_past_distance": 5.0, "identity_stages": 3, "personality_count": 2}

    for filter_id in filter_ids:
        first = build_strategy_schedule(filter_id, clips=clips, windows=windows, duration=250.0, parameters=parameters)
        second = build_strategy_schedule(filter_id, clips=clips, windows=windows, duration=250.0, parameters=parameters)
        assert first["mappings"] == second["mappings"], filter_id
        assert first["filter_validation"]["passed"] is True, filter_id
        assert first["filter_progress_stages"], filter_id
        contract = default_contract_catalog().get(filter_id)
        for invariant in contract.data["hard_invariants"]:
            key = invariant["validator"].split(".", 1)[1]
            assert first["filter_validation"][key] is True, f"{filter_id}: {key}"
        definition = default_filter_registry().get(filter_id)
        plan = plan_from_schedule(definition=definition, schedule=first, seed=1)
        validate_artifact("filter_plan", write_filter_plan(plan, tmp_path / f"{filter_id}_plan.json"), Path.cwd() / "schemas")


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


def test_contagion_uses_full_contact_history_but_only_audio_safe_placement_windows() -> None:
    clips = [clip(f"carrier{i}", "s1", 70.0 + i * 5.0) for i in range(4)] + [clip("other", "s2", 5.0)]
    analysis_windows = [
        window("origin", "s1", 5.0, scene="scene1"),
        window("exposure", "s2", 10.0, scene="scene1"),
        window("unsafe_after", "s2", 30.0, scene="scene2"),
        {
            **window("safe_after", "s2", 50.0, scene="scene3"),
            "montage_moment_id": "moment_safe",
            "montage_audio_eligible": True,
            "montage_placement_eligible": True,
        },
    ]
    placement_windows = [analysis_windows[-1]]
    params = default_filter_registry().get("contagion").parameter_defaults | {
        "initial_carrier": "s1", "maximum_infected_speakers": 2, "intensity": "Trace", "seed": 2,
    }

    schedule = build_strategy_schedule(
        "contagion",
        clips=clips,
        windows=analysis_windows,
        placement_windows=placement_windows,
        duration=120.0,
        parameters=params,
    )

    assert [row["window_id"] for row in schedule["mappings"]] == ["safe_after"]
    assert schedule["mappings"][0]["montage_moment_id"] == "moment_safe"
    assert schedule["mappings"][0]["montage_rescue"] is True
    assert schedule["filter_metrics"]["montage_rescue_used"] is True
    assert schedule["montage_window_eligibility"] == {
        "analysis_window_count": 4,
        "placement_window_count": 1,
        "strategy_used_full_window_context": True,
        "authored_placements_restricted_to_audio_safe_moments": True,
    }
    assert schedule["filter_fallbacks"][0]["action"] == "SELECTED_DETERMINISTIC_AUDIO_SAFE_POST_INFECTION_WINDOW"


def test_standard_strategy_dispatch_restricts_authored_placements_to_montage_windows() -> None:
    params = default_filter_registry().get("possession").parameter_defaults | {
        "possessing_speaker": "s1", "possessed_speaker": "s2", "intensity": "Total",
        "minimum_temporal_separation": 1.0,
    }
    unsafe = window("unsafe", "s2", 50.0)
    safe = {**window("safe", "s2", 70.0), "montage_moment_id": "moment_safe", "montage_audio_eligible": True}

    schedule = build_strategy_schedule(
        "possession",
        clips=[clip("a", "s1", 10.0), clip("b", "s1", 20.0)],
        windows=[unsafe, safe],
        placement_windows=[safe],
        duration=100.0,
        parameters=params,
    )

    assert [row["window_id"] for row in schedule["mappings"]] == ["safe"]
    assert schedule["mappings"][0]["montage_moment_id"] == "moment_safe"
    assert schedule["montage_window_eligibility"]["analysis_window_count"] == 2
    assert schedule["montage_window_eligibility"]["placement_window_count"] == 1


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
