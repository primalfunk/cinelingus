from pathlib import Path

import pytest

from cinelingus.contract_kernel import MediaDescriptor, StreamDescriptor, compile_run_contract
from cinelingus.filter_lab.models import FilmInput
from cinelingus.filter_lab.multiworld_strategies import build_multiworld_schedule
from cinelingus.filter_lab.registry import default_filter_registry
from cinelingus.multi_input_guarantee import (
    MultiInputGuaranteeStatus,
    applicable_multi_input_filter_ids,
    certify_multi_input_schedule,
    write_multi_input_guarantee,
)
from cinelingus.qualification import qualify_schedule


def _world(count: int):
    films = tuple(
        FilmInput(
            id=f"film_{index + 1}",
            media_path=Path(f"film_{index + 1}.mp4"),
            label=f"Film {chr(65 + index)}",
            is_anchor=index == 0,
        )
        for index in range(count)
    )
    artifacts = {}
    media = []
    for film_index, film in enumerate(films):
        media_hash = f"hash_{film_index}"
        clips = [
            {
                "id": f"clip_{film_index}_{index}",
                "path": f"cache/{media_hash}/source_dialogue/clip_{index}.wav",
                "start": index * 8.0,
                "duration": 2.0 + index % 3,
                "speaker_id": f"speaker_{index % 2}",
                "transcript": f"Film {film_index} line {index}",
            }
            for index in range(16)
        ]
        windows = [
            {
                "id": f"window_{film_index}_{index}",
                "start": index * 8.0 + 1.0,
                "duration": 2.5,
                "speaker_id": f"speaker_{index % 2}",
            }
            for index in range(16)
        ]
        artifacts[film.id] = {
            "movie": {"duration": 140.0},
            "media_hash": media_hash,
            "clips": clips,
            "windows": windows,
        }
        media.append(
            MediaDescriptor(
                path=film.media_path,
                media_hash=media_hash,
                format_duration=140.0,
                streams=(
                    StreamDescriptor(0, "video", "h264", 140.0, 0.0),
                    StreamDescriptor(1, "audio", "aac", 140.0, 0.0),
                ),
            )
        )
    return films, artifacts, tuple(media)


def test_applicable_filter_matrix_is_exhaustive_by_declared_arity() -> None:
    assert set(applicable_multi_input_filter_ids(2)) == {
        "multiworld.translation",
        "multiworld.possession",
        "multiworld.contagion",
        "multiworld.echo_chamber",
        "multiworld.prophecy",
    }
    assert set(applicable_multi_input_filter_ids(3)) == {
        "multiworld.contagion",
        "multiworld.echo_chamber",
        "multiworld.triangle",
    }
    assert set(applicable_multi_input_filter_ids(7)) == {
        "multiworld.contagion",
        "multiworld.echo_chamber",
    }
    assert applicable_multi_input_filter_ids(1) == ()


@pytest.mark.parametrize(
    ("filter_id", "film_count"),
    [
        ("multiworld.possession", 2),
        ("multiworld.prophecy", 2),
        ("multiworld.contagion", 2),
        ("multiworld.contagion", 3),
        ("multiworld.contagion", 5),
        ("multiworld.echo_chamber", 2),
        ("multiworld.echo_chamber", 3),
        ("multiworld.echo_chamber", 5),
        ("multiworld.triangle", 3),
    ],
)
def test_every_general_multiworld_filter_reaches_ready_to_render(filter_id: str, film_count: int) -> None:
    films, artifacts, media = _world(film_count)
    definition = default_filter_registry().get(filter_id)
    contract = compile_run_contract(definition=definition, media=media)
    schedule = build_multiworld_schedule(
        filter_id,
        films=films,
        film_artifacts=artifacts,
        parameters={"intensity": "Total"},
        seed=11,
    )
    if definition.requires_speaker_identity:
        schedule["identity_quality"] = {"passed": True}
    qualification = qualify_schedule(schedule, contract)
    guarantee = certify_multi_input_schedule(
        contract=contract,
        schedule=schedule,
        qualification=qualification,
    )
    assert guarantee.status == MultiInputGuaranteeStatus.READY_TO_RENDER
    assert all(guarantee.checks.values())


def test_translation_reaches_ready_to_render_with_donor_provenance(tmp_path: Path) -> None:
    _films, _artifacts, media = _world(2)
    definition = default_filter_registry().get("multiworld.translation")
    contract = compile_run_contract(definition=definition, media=media)
    schedule = {
        "mappings": [
            {
                "window_id": "w1",
                "clip_id": "film_2:c1",
                "source_media_hash": "hash_1",
                "enabled": True,
            }
        ]
    }
    qualification = qualify_schedule(schedule, contract)
    guarantee = certify_multi_input_schedule(
        contract=contract,
        schedule=schedule,
        qualification=qualification,
    )
    assert guarantee.status == MultiInputGuaranteeStatus.READY_TO_RENDER
    path = write_multi_input_guarantee(
        guarantee,
        tmp_path / "multi_input_guarantee.json",
        Path(__file__).parents[1] / "schemas",
    )
    assert path.exists()


def test_missing_required_donor_is_rejected_before_render() -> None:
    _films, _artifacts, media = _world(3)
    definition = default_filter_registry().get("multiworld.contagion")
    contract = compile_run_contract(definition=definition, media=media)
    schedule = {
        "mappings": [
            {
                "window_id": "w1",
                "clip_id": "film_2:c1",
                "source_media_hash": "hash_1",
                "enabled": True,
            }
        ]
    }
    qualification = qualify_schedule(schedule, contract)
    guarantee = certify_multi_input_schedule(
        contract=contract,
        schedule=schedule,
        qualification=qualification,
    )
    assert guarantee.status == MultiInputGuaranteeStatus.REJECTED
    assert guarantee.checks["all_required_inputs_contribute"] is False
