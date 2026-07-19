from pathlib import Path

from cinelingus.contract_kernel import MediaDescriptor, StreamDescriptor, compile_run_contract
from cinelingus.filter_lab.registry import default_filter_registry
from cinelingus.qualification import QualificationStatus, qualify_schedule


def _contract():
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
        definition=default_filter_registry().get("multiworld.echo_chamber"),
        media=media,
    )


def test_echo_groups_remain_atomic_after_repetition_qualification() -> None:
    schedule = {
        "render_duration": 100.0,
        "mappings": [
            {"window_id": "w1", "echo_group_id": "g1", "clip_id": "a", "source_media_hash": "hash_0", "enabled": True},
            {"window_id": "w1", "echo_group_id": "g1", "clip_id": "b", "source_media_hash": "hash_1", "enabled": True},
            {"window_id": "w2", "echo_group_id": "g2", "clip_id": "a", "source_media_hash": "hash_0", "enabled": True},
            {"window_id": "w2", "echo_group_id": "g2", "clip_id": "c", "source_media_hash": "hash_1", "enabled": True},
        ],
    }
    qualification = qualify_schedule(schedule, _contract())
    enabled = [row for row in schedule["mappings"] if row.get("enabled", True)]
    assert {row["echo_group_id"] for row in enabled} == {"g1"}
    assert {row["source_media_hash"] for row in enabled} == {"hash_0", "hash_1"}
    assert qualification.status == QualificationStatus.DEGRADED
    assert qualification.measurements["incomplete_echo_groups_removed"] == 1
