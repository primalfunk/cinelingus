from pathlib import Path

from cinelingus.contract_kernel import MediaDescriptor, StreamDescriptor, compile_run_contract
from cinelingus.contract_runtime import activate_run_contract
from cinelingus.filter_lab.acceptance import validate_schedule_quality
from cinelingus.filter_lab.registry import default_filter_registry
from cinelingus.montage import build_full_timeline_plan


LAWS = {
    "visual": "COMPLETE_ANCHOR_TIMELINE_FROM_ZERO",
    "temporal": "ANCHOR_CHRONOLOGY_PRESERVED",
    "dialogue": "MULTIWORLD_FILTER_CONTRACT",
    "requested_audio": "MULTIWORLD_FILTER_CONTRACT_AUDIO_LAW",
    "actual_audio_method": "CONTINUOUS_SOURCE_SOUNDTRACK_BED",
}


def _contract(filter_id: str):
    media = tuple(
        MediaDescriptor(
            path=Path(f"film_{index}.mp4"),
            media_hash=f"hash_{index}",
            format_duration=100.0,
            streams=(
                StreamDescriptor(0, "video", "h264", 100.0, 0.0),
                StreamDescriptor(1, "audio", "aac", 100.0, 0.0),
            ),
        )
        for index in range(2)
    )
    return compile_run_contract(
        definition=default_filter_registry().get(filter_id),
        media=media,
    )


def _plan(contract, schedule):
    return build_full_timeline_plan(
        filter_id=contract.filter_id,
        filter_contract_version=contract.filter_version,
        anchor_source_id="film_1",
        anchor_media_hash="hash_0",
        anchor_duration=100.0,
        supporting_audio_durations=[100.0],
        random_seed=1,
        governing_relationship="test",
        laws=LAWS,
        schedule=schedule,
    )


def test_sparse_possession_is_judged_by_law_not_generic_density() -> None:
    contract = _contract("multiworld.possession")
    schedule = {
        "render_duration": 100.0,
        "identity_quality": {"passed": True},
        "acceptance_requirements": {
            "minimum_dialogue_coverage": 0.08,
            "timeline_bucket_count": 4,
            "minimum_occupied_timeline_buckets": 3,
            "minimum_unique_source_ratio": 0.8,
            "maximum_source_reuse": 2,
        },
        "mappings": [{
            "window_id": "w1",
            "clip_id": "c1",
            "source_media_hash": "hash_1",
            "destination_timestamp": 2.0,
            "planned_render_duration": 1.0,
            "enabled": True,
        }],
    }
    with activate_run_contract(contract):
        _plan(contract, schedule)
    quality = validate_schedule_quality(schedule)
    assert all(quality["checks"].values())
    assert schedule["acceptance_requirements"]["minimum_dialogue_coverage"] == 0.0
    assert schedule["acceptance_requirements"]["minimum_occupied_timeline_buckets"] == 1


def test_repeated_multiworld_sources_are_qualified_before_legacy_acceptance() -> None:
    contract = _contract("multiworld.contagion")
    schedule = {
        "render_duration": 100.0,
        "mappings": [
            {"window_id": "w1", "clip_id": "c1", "source_media_hash": "hash_1", "destination_timestamp": 2.0, "planned_render_duration": 1.0, "enabled": True},
            {"window_id": "w2", "clip_id": "c1", "source_media_hash": "hash_1", "destination_timestamp": 60.0, "planned_render_duration": 1.0, "enabled": True},
        ],
    }
    with activate_run_contract(contract):
        plan = _plan(contract, schedule)
    assert [row["enabled"] for row in schedule["mappings"]] == [True, False]
    assert plan["provenance"]["schedule_qualification"]["status"] == "DEGRADED"
    assert all(validate_schedule_quality(schedule)["checks"].values())
