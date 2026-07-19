from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from .contract_kernel import RunContract
from .qualification import QualificationStatus, ScheduleQualification
from .util import read_json, utc_now, write_json
from .validation import _validate_object


CERTIFICATION_VERSION = "filter_certification_v1"


class CertificationState(str, Enum):
    CERTIFIED = "CERTIFIED"
    EXPERIMENTAL = "EXPERIMENTAL"
    DEGRADED = "DEGRADED"
    BLOCKED = "BLOCKED"
    UNAVAILABLE = "UNAVAILABLE"


@dataclass(frozen=True)
class FilterCertification:
    state: CertificationState
    contract_id: str
    filter_id: str
    checks: dict[str, bool]
    evidence: dict[str, Any]
    reasons: tuple[str, ...]
    creation_timestamp: str
    certification_version: str = CERTIFICATION_VERSION
    schema_version: str = "1.0"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "certification_version": self.certification_version,
            "creation_timestamp": self.creation_timestamp,
            "contract_id": self.contract_id,
            "filter_id": self.filter_id,
            "state": self.state.value,
            "checks": dict(self.checks),
            "evidence": dict(self.evidence),
            "reasons": list(self.reasons),
        }


def certify_filter_run(
    *,
    contract: RunContract,
    qualification: ScheduleQualification | None,
    filter_acceptance: dict[str, Any] | None,
    render_acceptance: dict[str, Any] | None,
    execution_error: str | None = None,
) -> FilterCertification:
    """Derive a filter's availability state solely from run evidence."""
    qualification_ready = qualification is not None and qualification.status in {
        QualificationStatus.READY,
        QualificationStatus.DEGRADED,
    }
    filter_pass = bool(filter_acceptance and filter_acceptance.get("status") == "pass")
    render_pass = bool(render_acceptance and render_acceptance.get("acceptance_status") == "PASS")
    checks = {
        "execution_completed": execution_error is None,
        "schedule_qualified": qualification_ready,
        "filter_acceptance_passed": filter_pass,
        "render_acceptance_passed": render_pass,
        "contract_identity_consistent": bool(
            qualification is not None and qualification.contract_id == contract.contract_id
        ),
    }
    reasons: list[str] = []
    if execution_error:
        reasons.append(execution_error)
    if qualification is None:
        reasons.append("No schedule qualification evidence was produced.")
    elif qualification.status == QualificationStatus.BLOCKED:
        reasons.extend(qualification.reasons or ("Schedule qualification was blocked.",))
    elif qualification.status == QualificationStatus.DEGRADED:
        reasons.extend(qualification.reasons)
    if filter_acceptance is None:
        reasons.append("No filter acceptance evidence was produced.")
    elif not filter_pass:
        reasons.append("Filter acceptance failed.")
    if render_acceptance is None:
        reasons.append("No final render acceptance evidence was produced.")
    elif not render_pass:
        reasons.append("Final stream or timing acceptance failed.")

    if (
        execution_error
        or qualification is not None and qualification.status == QualificationStatus.BLOCKED
        or filter_acceptance is not None and not filter_pass
        or render_acceptance is not None and not render_pass
    ):
        state = CertificationState.BLOCKED
    elif all(checks.values()):
        state = (
            CertificationState.DEGRADED
            if qualification is not None and qualification.status == QualificationStatus.DEGRADED
            else CertificationState.CERTIFIED
        )
    elif execution_error is None:
        state = CertificationState.EXPERIMENTAL
    else:
        state = CertificationState.BLOCKED

    return FilterCertification(
        state=state,
        contract_id=contract.contract_id,
        filter_id=contract.filter_id,
        checks=checks,
        evidence={
            "schedule_qualification_status": qualification.status.value if qualification else None,
            "schedule_signature": qualification.schedule_signature if qualification else None,
            "filter_acceptance_status": filter_acceptance.get("status") if filter_acceptance else None,
            "render_acceptance_status": render_acceptance.get("acceptance_status") if render_acceptance else None,
            "canonical_output_duration": contract.timeline.duration,
        },
        reasons=tuple(dict.fromkeys(reasons)),
        creation_timestamp=utc_now(),
    )


def write_filter_certification(
    certification: FilterCertification,
    output_path: Path,
    schemas_dir: Path,
) -> Path:
    data = certification.to_dict()
    _validate_object(data, read_json(schemas_dir / "filter_certification.schema.json"), str(output_path))
    write_json(output_path, data)
    return output_path


