from dataclasses import replace
from pathlib import Path
import wave

import pytest

from cinelingus.config import load_config
from cinelingus.pipeline import Pipeline, _multiworld_identity_quality
from cinelingus.util import read_json, write_json
from cinelingus.filter_lab.multiworld import MULTIWORLD_STAGES, MultiworldPipeline
from cinelingus.filter_lab.acceptance import _provenance_check
from cinelingus.filter_lab.multiworld_strategies import build_multiworld_schedule
from cinelingus.filter_lab.models import FilmInput
from cinelingus.filter_lab.contracts import default_contract_catalog
from cinelingus.filter_lab.gui_controller import current_filter_definition
from cinelingus.filter_lab.presentation import detail_text, film_selector_spec
from cinelingus.filter_lab.registry import default_filter_registry


def test_first_wave_multiworld_catalog_stabilizes_ids_and_cardinality() -> None:
    registry = default_filter_registry()
    definitions = registry.filters_for_family("multiworld")

    assert {row.id for row in definitions} == {
        "multiworld.translation", "multiworld.possession", "multiworld.contagion",
        "multiworld.doppelganger", "multiworld.mirror_world", "multiworld.prophecy",
        "multiworld.echo_chamber", "multiworld.bleed", "multiworld.parallel_universes",
        "multiworld.wormhole", "multiworld.chimera", "multiworld.triangle",
        "multiworld.civilization",
    }
    assert {row.id for row in definitions if row.implemented} == {
        "multiworld.translation", "multiworld.possession", "multiworld.contagion",
        "multiworld.echo_chamber", "multiworld.prophecy",
    }
    assert registry.get("multiworld.chimera").minimum_films == 3
    assert registry.get("multiworld.chimera").maximum_films == 3
    assert registry.get("multiworld.civilization").minimum_films == 5
    assert registry.get("multiworld.civilization").maximum_films is None


def test_gui_selector_spec_is_driven_by_contract_cardinality() -> None:
    registry = default_filter_registry()

    chimera = film_selector_spec(registry.get("multiworld.chimera"))
    civilization = film_selector_spec(registry.get("multiworld.civilization"))
    expanded = film_selector_spec(registry.get("multiworld.civilization"), selected_count=7)

    assert [row["label"] for row in chimera["rows"]] == ["Film A (Anchor)", "Film B", "Film C"]
    assert chimera["can_add"] is False
    assert len(civilization["rows"]) == 5
    assert civilization["can_add"] is True
    assert len(expanded["rows"]) == 7
    assert all(row["removable"] for row in expanded["rows"][5:])
    many = film_selector_spec(registry.get("multiworld.civilization"), selected_count=28)
    assert many["rows"][26]["label"] == "Film AA"
    assert many["rows"][27]["label"] == "Film AB"


def test_app_config_preserves_arbitrary_ordered_films_and_anchor(tmp_path: Path) -> None:
    base = load_config(Path.cwd())
    paths = [tmp_path / f"film_{index}.mp4" for index in range(5)]

    config = base.with_films(paths, anchor_index=2)

    assert config.films == (paths[2], paths[0], paths[1], paths[3], paths[4])
    assert config.destination_video == paths[2]
    assert config.source_dialogue == paths[0]
    assert config.anchor_film == paths[2]
    assert config.with_overrides(source_dialogue=config.destination_video).films == (paths[2],)


def test_multiworld_provenance_requires_cache_roots_and_mapping_hashes_to_match() -> None:
    mappings = [
        {"clip_path": "cache/hash_a/source_dialogue/a.wav", "source_media_hash": "hash_a"},
        {"clip_path": "cache/hash_b/source_dialogue/b.wav", "source_media_hash": "hash_b"},
    ]
    schedule = {"source_media_hashes": ["hash_a", "hash_b"]}

    passed = _provenance_check(schedule, mappings, None)
    failed = _provenance_check(schedule, [*mappings, {"clip_path": "cache/hash_c/source_dialogue/c.wav", "source_media_hash": "hash_c"}], None)

    assert passed["passed"] is True
    assert passed["basis"] == "multiworld_schedule_clip_cache_roots"
    assert failed["passed"] is False


def test_duplicate_filter_names_resolve_inside_the_selected_family() -> None:
    class Variable:
        def __init__(self, value: str) -> None:
            self.value = value

        def get(self) -> str:
            return self.value

    app = type("App", (), {"family_var": Variable("Multiworld"), "mode_var": Variable("Possession")})()
    assert current_filter_definition(app).id == "multiworld.possession"
    app.family_var = Variable("Identity")
    assert current_filter_definition(app).id == "identity.possession"
    assert "This filter is not yet implemented" not in detail_text(default_filter_registry().get("multiworld.possession"))


def test_translation_family_lookup_accepts_new_and_legacy_names() -> None:
    class Variable:
        def __init__(self, value: str) -> None:
            self.value = value

        def get(self) -> str:
            return self.value

    app = type("App", (), {"family_var": Variable("Multiworld"), "mode_var": Variable("Translation")})()
    assert current_filter_definition(app).id == "multiworld.translation"
    app.mode_var = Variable("Translation")
    assert current_filter_definition(app).id == "multiworld.translation"
    app.mode_var = Variable("Transposition")
    assert current_filter_definition(app).id == "multiworld.translation"


def test_general_pipeline_runs_each_stage_in_order_for_any_film_count(tmp_path: Path) -> None:
    definition = default_filter_registry().get("multiworld.chimera")
    runnable = replace(definition, implemented=True)
    pipeline = MultiworldPipeline(runnable, [tmp_path / f"film_{index}.mp4" for index in range(3)], seed=9)

    pipeline.inspect_films(lambda film: {"path": str(film.media_path), "duration": 90.0})
    pipeline.create_shared_timeline()
    pipeline.construct_world_model()
    pipeline.apply_cinematic_law(lambda state: {"law": state.definition.cinematic_law})
    pipeline.generate_replacement_decisions()
    pipeline.review(lambda _state: {"status": "pass"})
    pipeline.render(lambda _state: {"video": "chimera.mp4"})

    assert tuple(pipeline.state.completed_stages) == MULTIWORLD_STAGES
    assert pipeline.state.anchor.label == "Film A"
    assert pipeline.state.world_model["deterministic_seed"] == 9


def test_multiworld_pipeline_rejects_invalid_cardinality_and_unimplemented_law(tmp_path: Path) -> None:
    registry = default_filter_registry()
    with pytest.raises(ValueError, match="at least 3 films"):
        MultiworldPipeline(registry.get("multiworld.chimera"), [tmp_path / "a.mp4", tmp_path / "b.mp4"])
    with pytest.raises(ValueError, match="distinct film paths"):
        MultiworldPipeline(registry.get("multiworld.possession"), [tmp_path / "a.mp4", tmp_path / "a.mp4"])

    pipeline = MultiworldPipeline(registry.get("multiworld.mirror_world"), [tmp_path / "a.mp4", tmp_path / "b.mp4"])
    pipeline.inspect_films(lambda _film: {"duration": 1.0})
    pipeline.create_shared_timeline()
    pipeline.construct_world_model()
    with pytest.raises(NotImplementedError, match="not yet implemented"):
        pipeline.apply_cinematic_law(lambda _state: {})


def _synthetic_world(count: int) -> tuple[tuple[FilmInput, ...], dict[str, dict]]:
    films = tuple(
        FilmInput(id=f"film_{index + 1}", media_path=Path(f"film_{index + 1}.mp4"), label=f"Film {chr(65 + index)}", is_anchor=index == 0)
        for index in range(count)
    )
    artifacts = {}
    for film_index, film in enumerate(films):
        clips = [
            {
                "id": f"clip_{film_index}_{index}", "path": f"cache/hash_{film_index}/source_dialogue/clip_{index}.wav",
                "start": index * 8.0, "duration": 2.0 + index % 3, "speaker_id": f"speaker_{index % 2}",
                "transcript": f"Film {film_index} line {index}",
            }
            for index in range(12)
        ]
        windows = [
            {"id": f"window_{film_index}_{index}", "start": index * 8.0 + 2.0, "duration": 2.5, "speaker_id": f"speaker_{index % 2}"}
            for index in range(12)
        ]
        artifacts[film.id] = {"movie": {"duration": 100.0}, "media_hash": f"hash_{film_index}", "clips": clips, "windows": windows}
    return films, artifacts


@pytest.mark.parametrize(
    ("filter_id", "film_count"),
    [
        ("multiworld.possession", 2),
        ("multiworld.contagion", 3),
        ("multiworld.echo_chamber", 3),
        ("multiworld.prophecy", 2),
    ],
)
def test_multiworld_dialogue_laws_are_deterministic_and_validate_contract_invariants(filter_id: str, film_count: int) -> None:
    films, artifacts = _synthetic_world(film_count)

    first = build_multiworld_schedule(filter_id, films=films, film_artifacts=artifacts, parameters={"intensity": "Total"}, seed=7)
    second = build_multiworld_schedule(filter_id, films=films, film_artifacts=artifacts, parameters={"intensity": "Total"}, seed=7)

    assert first["mappings"] == second["mappings"]
    assert first["acceptance_requirements"]["minimum_dialogue_coverage"] == 0.08
    assert first["filter_validation"]["passed"] is True
    assert all(row["source_film_id"] in {film.id for film in films} for row in first["mappings"])
    definition = default_filter_registry().get(filter_id)
    definition.validate_film_count(film_count)
    contract = default_contract_catalog().get(filter_id)
    for invariant in contract.data["hard_invariants"]:
        key = invariant["validator"].split(".", 1)[1]
        assert first["filter_validation"][key] is True


def test_general_multiworld_runtime_writes_world_artifacts_and_completes_all_stages(monkeypatch, tmp_path: Path) -> None:
    import cinelingus.pipeline as pipeline_module

    media = [tmp_path / "anchor.mp4", tmp_path / "donor.mp4"]
    for index, path in enumerate(media):
        path.write_bytes(f"film-{index}".encode())
    base = load_config(Path.cwd())
    config = replace(
        base,
        cache_dir=tmp_path / "cache",
        output_dir=tmp_path / "output",
        temp_dir=tmp_path / "temp",
    ).with_films(media)
    pipeline = Pipeline(config)

    def fake_analysis(self, media_path: Path, *, force: bool):
        film_index = media.index(media_path)
        media_hash = self.destination.media_hash if film_index == 0 else pipeline.source.media_hash
        clips = [
            {
                "id": f"clip_{film_index}_{index}",
                "path": str(config.cache_dir / media_hash / "source_dialogue" / f"clip_{index}.wav"),
                "start": index * 8.0,
                "duration": 2.0,
                "speaker_id": f"speaker_{index % 2}",
                "transcript": f"line {index}",
            }
            for index in range(8)
        ]
        windows = [
            {"id": f"window_{film_index}_{index}", "start": index * 8.0 + 1.0, "end": index * 8.0 + 3.0, "duration": 2.0, "speaker_id": f"speaker_{index % 2}"}
            for index in range(8)
        ]
        visual = {
            "shots": {
                "shots": [
                    {"id": f"shot_{index}", "start": index * 8.0, "end": index * 8.0 + 4.0, "scene_id": f"scene_{index}"}
                    for index in range(8)
                ]
            }
        }
        return self, {
            "media_hash": media_hash,
            "movie": {"duration": 70.0},
            "clips": clips,
            "windows": windows,
            "source_performances": {"performances": []},
            "destination_performances": {"performances": []},
            "visual": visual,
            "speaker_maps": {
                "source": _speaker_map(8, 2),
                "destination": _speaker_map(8, 2),
            },
        }

    def fake_render(**kwargs):
        kwargs["audio_output"].parent.mkdir(parents=True, exist_ok=True)
        kwargs["audio_output"].write_bytes(b"wav")
        kwargs["video_output"].write_bytes(b"mp4")

    def fake_acceptance(**kwargs):
        write_json(kwargs["output_path"], {"schema_version": "1.0", "status": "pass"})
        return {"status": "pass"}

    def fake_extract_analysis_audio(_media_path: Path, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(output_path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(100)
            handle.writeframes((3000).to_bytes(2, "little", signed=True) * 7000)

    monkeypatch.setattr(Pipeline, "_analyze_multiworld_film", fake_analysis)
    monkeypatch.setattr(pipeline_module, "extract_analysis_audio", fake_extract_analysis_audio)
    monkeypatch.setattr(pipeline_module, "render_mutation_media", fake_render)
    monkeypatch.setattr(pipeline_module, "ffprobe_json", lambda _path: {"streams": [{"codec_type": "video"}, {"codec_type": "audio"}], "format": {"duration": "70.0"}})
    monkeypatch.setattr(pipeline_module, "validate_filter_output", fake_acceptance)
    monkeypatch.setattr(Pipeline, "_write_and_validate", lambda _self, _kind, path, data: write_json(path, data))
    monkeypatch.setattr(pipeline_module, "publish_single_video", lambda **kwargs: kwargs["video"])
    monkeypatch.setattr(pipeline_module, "_rewrite_published_video_references", lambda **_kwargs: None)

    result = pipeline.run_multiworld_filter("multiworld.possession", parameters={"intensity": "Total"})

    schedule = read_json(result["schedule"])
    report = read_json(result["multiworld_report"])
    assert result["video"].exists()
    assert schedule["multiworld"]["completed_stages"] == list(MULTIWORLD_STAGES)
    assert report["completed_stages"] == list(MULTIWORLD_STAGES)
    assert result["analysis_film_inspections"].exists()
    assert result["analysis_shared_timeline"].exists()
    assert result["analysis_world_model"].exists()
    assert result["filter_plan"].exists()
    assert result["montage_plan"].exists()
    assert result["montage_render_acceptance"].exists()
    assert schedule["montage_native"] is True
    assert schedule["full_timeline_native"] is True
    assert schedule["render_duration"] == 70.0


def _speaker_map(item_count: int, speaker_count: int, *, partial: bool = False) -> dict:
    return {
        "actual_backend": "pyannote_partial" if partial else "pyannote",
        "diagnostics": {"speech_item_count": item_count},
        "speaker_segments": [
            {"source_id": f"item_{index}", "speaker_id": f"speaker_{index % speaker_count:03d}"}
            for index in range(item_count)
        ],
    }


def test_identity_quality_warns_on_partial_maps_and_rejects_fragmentation() -> None:
    films, artifacts = _synthetic_world(2)
    for film in films:
        artifacts[film.id]["speaker_maps"] = {
            "source": _speaker_map(20, 4, partial=True),
            "destination": _speaker_map(20, 4, partial=True),
        }

    warning = _multiworld_identity_quality(films, artifacts)
    artifacts[films[0].id]["speaker_maps"]["destination"] = _speaker_map(30, 20, partial=True)
    failed = _multiworld_identity_quality(films, artifacts)

    assert warning["passed"] is True
    assert warning["status"] == "warning"
    assert warning["warnings"]
    assert failed["passed"] is False
    assert failed["films"][0]["fragmented"] is True
