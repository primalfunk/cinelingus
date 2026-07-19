from pathlib import Path

from cinelingus.certification import CertificationState, certify_filter_run, write_filter_certification
from cinelingus.contract_kernel import MediaDescriptor, RunContract, StreamDescriptor, compile_run_contract
from cinelingus.filter_lab.registry import default_filter_registry
from cinelingus.qualification import QualificationStatus, ScheduleQualification


def _contract() -> RunContract:
    media = MediaDescriptor(
        path=Path("film.mp4"),
        media_hash="film-hash",
        format_duration=120.0,
        streams=(
            StreamDescriptor(0, "video", "h264", 120.0, 0.0),
            StreamDescriptor(1, "audio", "aac", 120.0, 0.0),
        ),
    )
    return compile_run_contract(
        definition=default_filter_registry().get("time.foreshadow"),
        media=(media,),
    )


def _qualification(contract: RunContract, status: QualificationStatus) -> ScheduleQualification:
    return ScheduleQualification(
        status=status,
        contract_id=contract.contract_id,
        filter_id=contract.filter_id,
        checks={"ok": True},
        measurements={},
        actions=(),
        reasons=("one placement was safely omitted",) if status == QualificationStatus.DEGRADED else (),
        schedule_signature="schedule-hash",
        creation_timestamp="2026-07-18T00:00:00+00:00",
    )


def test_certification_requires_all_three_evidence_layers() -> None:
    contract = _contract()
    record = certify_filter_run(
        contract=contract,
        qualification=_qualification(contract, QualificationStatus.READY),
        filter_acceptance={"status": "pass"},
        render_acceptance={"acceptance_status": "PASS"},
    )
    assert record.state == CertificationState.CERTIFIED
    assert all(record.checks.values())


def test_safe_candidate_exhaustion_is_degraded_not_failed() -> None:
    contract = _contract()
    record = certify_filter_run(
        contract=contract,
        qualification=_qualification(contract, QualificationStatus.DEGRADED),
        filter_acceptance={"status": "pass"},
        render_acceptance={"acceptance_status": "PASS"},
    )
    assert record.state == CertificationState.DEGRADED


def test_missing_render_evidence_remains_experimental() -> None:
    contract = _contract()
    record = certify_filter_run(
        contract=contract,
        qualification=_qualification(contract, QualificationStatus.READY),
        filter_acceptance={"status": "pass"},
        render_acceptance=None,
    )
    assert record.state == CertificationState.EXPERIMENTAL


def test_execution_error_is_blocked_and_schema_valid(tmp_path: Path) -> None:
    contract = _contract()
    record = certify_filter_run(
        contract=contract,
        qualification=None,
        filter_acceptance=None,
        render_acceptance=None,
        execution_error="planner failed",
    )
    assert record.state == CertificationState.BLOCKED
    path = write_filter_certification(
        record,
        tmp_path / "filter_certification.json",
        Path(__file__).parents[1] / "schemas",
    )
    assert path.exists()
