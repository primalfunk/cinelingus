from pathlib import Path

from cinelingus.contract_kernel import MediaDescriptor, StreamDescriptor, compile_run_contract
from cinelingus.filter_lab.registry import default_filter_registry
from cinelingus.multi_input_guarantee import MultiInputGuaranteeStatus, certify_multi_input_schedule
from cinelingus.qualification import qualify_schedule


def test_translation_schedule_level_donor_hash_satisfies_guarantee() -> None:
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
    contract = compile_run_contract(
        definition=default_filter_registry().get("multiworld.translation"),
        media=media,
    )
    schedule = {
        "source_media_hash": "hash_1",
        "destination_media_hash": "hash_0",
        "mappings": [
            {"window_id": "w1", "clip_id": "c1", "enabled": True},
            {"window_id": "w2", "clip_id": "c2", "enabled": True},
        ],
    }
    qualification = qualify_schedule(schedule, contract)
    guarantee = certify_multi_input_schedule(
        contract=contract,
        schedule=schedule,
        qualification=qualification,
    )
    assert guarantee.status == MultiInputGuaranteeStatus.READY_TO_RENDER
    assert guarantee.observed_contributor_hashes == ("hash_1",)
