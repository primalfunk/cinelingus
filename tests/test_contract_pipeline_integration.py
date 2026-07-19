from pathlib import Path

from cinelingus.contract_kernel import MediaDescriptor, StreamDescriptor, compile_run_contract
from cinelingus.contract_runtime import activate_run_contract, active_schedule_qualification
from cinelingus.filter_lab.registry import default_filter_registry
from cinelingus.montage import build_full_timeline_plan
from cinelingus.qualification import QualificationStatus


LAWS = {
    "visual": "COMPLETE_SOURCE_TIMELINE_FROM_ZERO",
    "temporal": "DESTINATION_CHRONOLOGY_PRESERVED",
    "dialogue": "FILTER_CONTRACT",
    "requested_audio": "FILTER_CONTRACT_AUDIO_LAW",
    "actual_audio_method": "CONTINUOUS_SOURCE_SOUNDTRACK_BED",
}


def _contract(filter_id: str = "time.foreshadow"):
    media = MediaDescriptor(
        path=Path("pokemon.mp4"),
        media_hash="pokemon-hash",
        format_duration=1341.887,
        streams=(
            StreamDescriptor(0, "video", "h264", 1341.826522, 0.0),
            StreamDescriptor(1, "audio", "aac", 1341.887, 0.0),
        ),
    )
    return compile_run_contract(
        definition=default_filter_registry().get(filter_id),
        media=(media,),
    )


def _plan(schedule: dict, contract) -> dict:
    return build_full_timeline_plan(
        filter_id=contract.filter_id,
        filter_contract_version=contract.filter_version,
        anchor_source_id="film_1",
        anchor_media_hash="pokemon-hash",
        anchor_duration=1341.887,
        supporting_audio_durations=[1341.887],
        random_seed=7,
        governing_relationship="test",
        laws=LAWS,
        schedule=schedule,
    )


def test_canonical_stream_clock_repairs_foreshadow_duration_mismatch() -> None:
    contract = _contract()
    schedule = {"mappings": [{"window_id": "w1", "clip_id": "c1", "enabled": True}]}
    with activate_run_contract(contract):
        plan = _plan(schedule, contract)
    assert contract.timeline.duration == 1341.827
    assert plan["actual_duration"] == 1341.827
    assert plan["provenance"]["run_contract_id"] == contract.contract_id


def test_candidate_exhaustion_degrades_to_unmodified_windows_without_repetition() -> None:
    contract = _contract("emotion.regret")
    schedule = {
        "mappings": [
            {"window_id": "w1", "clip_id": "only-clip", "enabled": True},
            {"window_id": "w2", "clip_id": "only-clip", "enabled": True},
            {"window_id": "w3", "clip_id": "only-clip", "enabled": True},
        ]
    }
    with activate_run_contract(contract):
        plan = _plan(schedule, contract)
        qualification = active_schedule_qualification()
        assert qualification is not None
        assert qualification.status == QualificationStatus.DEGRADED
        assert qualification.measurements["disabled_mapping_count"] == 2
    assert [row["enabled"] for row in schedule["mappings"]] == [True, False, False]
    assert plan["provenance"]["schedule_qualification"]["status"] == "DEGRADED"


def test_echo_is_the_only_filter_authorized_to_repeat() -> None:
    contract = _contract("translation.echo")
    schedule = {
        "mappings": [
            {"window_id": "w1", "clip_id": "echo-clip", "enabled": True},
            {"window_id": "w2", "clip_id": "echo-clip", "enabled": True},
        ]
    }
    with activate_run_contract(contract):
        build_full_timeline_plan(
            filter_id=contract.filter_id,
            filter_contract_version=contract.filter_version,
            anchor_source_id="film_1",
            anchor_media_hash="pokemon-hash",
            anchor_duration=1341.887,
            supporting_audio_durations=[1341.887],
            random_seed=7,
            governing_relationship="test",
            laws=LAWS,
            schedule=schedule,
            repetition_authorized=True,
            repetition_authorization_basis="FILTER_CONTRACT:translation.echo",
        )
        qualification = active_schedule_qualification()
        assert qualification is not None
        assert qualification.status == QualificationStatus.READY
    assert all(row["enabled"] for row in schedule["mappings"])
