from pathlib import Path

from cinelingus.contract_kernel import MediaDescriptor, compile_run_contract
from cinelingus.filter_lab.registry import default_filter_registry
from cinelingus.qualification import QualificationStatus, qualify_schedule, write_schedule_qualification


def _contract(filter_id: str = "emotion.regret"):
    media = MediaDescriptor.from_probe(
        path=Path("film.mp4"), media_hash="hash",
        probe={"format": {"duration": "60"}, "streams": [
            {"codec_type": "video", "duration": "60"},
            {"codec_type": "audio", "duration": "60"},
        ]},
    )
    return compile_run_contract(definition=default_filter_registry().get(filter_id), media=[media])


def test_qualification_leaves_duplicate_windows_unmodified(tmp_path: Path) -> None:
    schedule = {"mappings": [
        {"window_id": "w1", "clip_id": "c1", "enabled": True, "destination_timestamp": 0.0},
        {"window_id": "w2", "clip_id": "c1", "enabled": True, "destination_timestamp": 10.0},
        {"window_id": "w3", "clip_id": "c2", "enabled": True, "destination_timestamp": 20.0},
    ], "filter_validation": {"proxy_is_disclosed": True}}

    result = qualify_schedule(schedule, _contract())
    path = tmp_path / "schedule_qualification.json"
    write_schedule_qualification(result, path, Path.cwd() / "schemas")

    assert result.status == QualificationStatus.DEGRADED
    assert result.measurements["disabled_mapping_count"] == 1
    assert [row["window_id"] for row in schedule["mappings"] if row.get("enabled", True)] == ["w1", "w3"]
    assert schedule["mappings"][1]["qualification_disabled_reason"]
    assert path.exists()


def test_qualification_is_ready_for_unique_foreshadow_schedule() -> None:
    schedule = {"mappings": [
        {"window_id": "w1", "clip_id": "c1", "enabled": True},
        {"window_id": "w2", "clip_id": "c2", "enabled": True},
    ], "filter_validation": {"future_only_rule": True}}

    result = qualify_schedule(schedule, _contract("time.foreshadow"))

    assert result.status == QualificationStatus.READY
    assert result.measurements["disabled_mapping_count"] == 0


def test_qualification_blocks_failed_filter_invariant() -> None:
    schedule = {"mappings": [{"window_id": "w1", "clip_id": "c1"}], "filter_validation": {"law": False}}

    result = qualify_schedule(schedule, _contract("memory.dream"))

    assert result.status == QualificationStatus.BLOCKED
