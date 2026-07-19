from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from .contract_kernel import RunContract
from .util import read_json, stable_hash, utc_now, write_json
from .validation import _validate_object


QUALIFICATION_VERSION = "schedule_qualification_v1"


class QualificationStatus(str, Enum):
    READY = "READY"
    DEGRADED = "DEGRADED"
    BLOCKED = "BLOCKED"


@dataclass(frozen=True)
class ScheduleQualification:
    status: QualificationStatus
    contract_id: str
    filter_id: str
    checks: dict[str, bool]
    measurements: dict[str, Any]
    actions: tuple[dict[str, Any], ...]
    reasons: tuple[str, ...]
    schedule_signature: str
    creation_timestamp: str
    qualification_version: str = QUALIFICATION_VERSION
    schema_version: str = "1.0"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "qualification_version": self.qualification_version,
            "creation_timestamp": self.creation_timestamp,
            "contract_id": self.contract_id,
            "filter_id": self.filter_id,
            "status": self.status.value,
            "checks": dict(self.checks),
            "measurements": dict(self.measurements),
            "actions": list(self.actions),
            "reasons": list(self.reasons),
            "schedule_signature": self.schedule_signature,
        }


def qualify_schedule(schedule: dict[str, Any], contract: RunContract) -> ScheduleQualification:
    """Qualify and safely degrade a schedule against the compiled run contract."""
    mappings = list(schedule.get("mappings") or [])
    enabled = [row for row in mappings if row.get("enabled", True)]
    seen: set[str] = set()
    duplicate_counts: dict[str, int] = {}
    qualified_mappings: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    reasons: list[str] = []
    repetition_forbidden = contract.repetition_policy.get("policy") == "forbidden"

    for mapping in mappings:
        row = dict(mapping)
        clip_id = str(row.get("clip_id") or row.get("clip_path") or "")
        if row.get("enabled", True) and repetition_forbidden and clip_id and clip_id in seen:
            duplicate_counts[clip_id] = duplicate_counts.get(clip_id, 1) + 1
            row["enabled"] = False
            row["qualification_disabled_reason"] = "candidate_pool_exhausted_without_repetition_authorization"
            actions.append({
                "action": "LEAVE_WINDOW_UNMODIFIED",
                "window_id": str(row.get("window_id") or ""),
                "clip_id": clip_id,
                "reason": row["qualification_disabled_reason"],
            })
        elif row.get("enabled", True) and clip_id:
            seen.add(clip_id)
        qualified_mappings.append(row)

    incomplete_echo_groups: list[str] = []
    if contract.filter_id == "multiworld.echo_chamber":
        required_hashes = {row.media_hash for row in contract.media}
        groups: dict[str, list[dict[str, Any]]] = {}
        for row in qualified_mappings:
            if not row.get("enabled", True):
                continue
            group_id = str(row.get("echo_group_id") or "")
            if group_id:
                groups.setdefault(group_id, []).append(row)
        for group_id, group in groups.items():
            observed_hashes = {
                str(row.get("source_media_hash"))
                for row in group
                if row.get("source_media_hash")
            }
            if observed_hashes == required_hashes:
                continue
            incomplete_echo_groups.append(group_id)
            for row in group:
                row["enabled"] = False
                row["qualification_disabled_reason"] = "echo_group_incomplete_after_repetition_qualification"
                actions.append({
                    "action": "LEAVE_ECHO_GROUP_UNMODIFIED",
                    "window_id": str(row.get("window_id") or ""),
                    "clip_id": str(row.get("clip_id") or row.get("clip_path") or ""),
                    "echo_group_id": group_id,
                    "reason": row["qualification_disabled_reason"],
                })

    schedule["mappings"] = qualified_mappings
    qualified_enabled = [row for row in qualified_mappings if row.get("enabled", True)]
    disabled_count = len(enabled) - len(qualified_enabled)
    filter_validation = dict(schedule.get("filter_validation") or {})
    declared_validation_passed = all(value is not False for value in filter_validation.values())
    checks = {
        "at_least_one_mapping": bool(qualified_enabled),
        "repetition_policy_satisfied": not repetition_forbidden or len({str(row.get("clip_id") or row.get("clip_path") or "") for row in qualified_enabled}) == len(qualified_enabled),
        "declared_filter_validation_passed": declared_validation_passed,
        "complete_input_scope": contract.input_scope == "complete_media_files",
        "canonical_extent_positive": contract.timeline.duration > 0,
        "echo_groups_remain_complete": not incomplete_echo_groups or bool(qualified_enabled),
    }
    if not checks["at_least_one_mapping"]:
        reasons.append("The filter produced no qualified placements.")
    if not checks["declared_filter_validation_passed"]:
        reasons.append("The filter's own schedule validation reported a failed invariant.")
    if disabled_count:
        reasons.append(
            f"{disabled_count} placements were left unmodified because the candidate pool was exhausted and repetition is forbidden."
        )

    if incomplete_echo_groups:
        reasons.append(
            f"{len(incomplete_echo_groups)} echo groups were left unmodified because repetition qualification removed one or more required film layers."
        )

    if not all(checks.values()):
        status = QualificationStatus.BLOCKED
    elif disabled_count:
        status = QualificationStatus.DEGRADED
    else:
        status = QualificationStatus.READY
    signature_payload = {
        "contract_id": contract.contract_id,
        "filter_id": contract.filter_id,
        "enabled": [
            {
                "window_id": row.get("window_id"),
                "clip_id": row.get("clip_id"),
                "destination_timestamp": row.get("destination_timestamp"),
            }
            for row in qualified_enabled
        ],
        "actions": actions,
    }
    qualification = ScheduleQualification(
        status=status,
        contract_id=contract.contract_id,
        filter_id=contract.filter_id,
        checks=checks,
        measurements={
            "proposed_mapping_count": len(enabled),
            "qualified_mapping_count": len(qualified_enabled),
            "disabled_mapping_count": disabled_count,
            "unique_source_count": len(seen),
            "duplicate_source_counts": duplicate_counts,
            "incomplete_echo_groups_removed": len(incomplete_echo_groups),
            "output_duration": contract.timeline.duration,
        },
        actions=tuple(actions),
        reasons=tuple(reasons),
        schedule_signature=stable_hash(signature_payload),
        creation_timestamp=utc_now(),
    )
    schedule["run_contract_id"] = contract.contract_id
    schedule["canonical_output_duration"] = contract.timeline.duration
    schedule["schedule_qualification"] = qualification.to_dict()
    return qualification


def write_schedule_qualification(
    qualification: ScheduleQualification,
    output_path: Path,
    schemas_dir: Path,
) -> Path:
    data = qualification.to_dict()
    _validate_object(data, read_json(schemas_dir / "schedule_qualification.schema.json"), str(output_path))
    write_json(output_path, data)
    return output_path
