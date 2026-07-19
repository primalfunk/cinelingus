import json
from pathlib import Path

import pytest

from cinelingus.run_guard import (
    FilterExecutionMismatch,
    RunInProgressError,
    exclusive_output_run,
    verify_filter_execution,
)


def test_output_run_lock_rejects_a_second_active_run(tmp_path: Path) -> None:
    output = tmp_path / "output"

    with exclusive_output_run(output, "contagion"):
        with pytest.raises(RunInProgressError, match="already using this output directory"):
            with exclusive_output_run(output, "translation"):
                pass


def test_output_run_lock_recovers_a_stale_process_lock(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    lock = output / ".cinelingus-active-run.json"
    lock.write_text(json.dumps({"run_id": "stale", "pid": 99999999, "filter_id": "multiworld.contagion"}))

    with exclusive_output_run(output, "contagion") as lease:
        assert lease.filter_id == "infection.contagion"
        assert lock.exists()

    assert not lock.exists()


def test_filter_execution_receipt_requires_matching_fresh_identity(tmp_path: Path) -> None:
    output = tmp_path / "output"
    video = output / "result.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"video")

    with exclusive_output_run(output, "multiworld.contagion") as lease:
        evidence = output / "filter_acceptance.json"
        evidence.write_text(json.dumps({"filter_id": "multiworld.translation", "status": "pass"}))
        with pytest.raises(FilterExecutionMismatch, match="Requested filter multiworld.contagion"):
            verify_filter_execution(
                lease,
                requested_filter_id="multiworld.contagion",
                evidence_paths=[evidence],
                output=video,
            )

        receipt = next((output / "run_receipts").glob("*.json"))
        assert json.loads(receipt.read_text())["status"] == "fail"


def test_filter_execution_receipt_accepts_matching_current_run_evidence(tmp_path: Path) -> None:
    output = tmp_path / "output"
    video = output / "result.mp4"

    with exclusive_output_run(output, "multiworld.contagion") as lease:
        video.write_bytes(b"video")
        evidence = output / "filter_acceptance.json"
        evidence.write_text(json.dumps({"filter_id": "multiworld.contagion", "status": "pass"}))
        receipt = verify_filter_execution(
            lease,
            requested_filter_id="multiworld.contagion",
            evidence_paths=[evidence],
            output=video,
        )

        document = json.loads(receipt.read_text())
        assert document["status"] == "pass"
        assert document["executed_filter_ids"] == ["multiworld.contagion"]
