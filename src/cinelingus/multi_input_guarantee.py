from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from .contract_kernel import RunContract
from .filter_lab.registry import default_filter_registry
from .qualification import QualificationStatus, ScheduleQualification
from .util import read_json, utc_now, write_json
from .validation import _validate_object


GUARANTEE_VERSION = "multi_input_guarantee_v1"


class MultiInputGuaranteeStatus(str, Enum):
    READY_TO_RENDER = "READY_TO_RENDER"
    REJECTED = "REJECTED"


@dataclass(frozen=True)
class MultiInputGuarantee:
    status: MultiInputGuaranteeStatus
    contract_id: str
    filter_id: str
    film_count: int
    applicable_filter_ids: tuple[str, ...]
    checks: dict[str, bool]
    expected_contributor_hashes: tuple[str, ...]
    observed_contributor_hashes: tuple[str, ...]
    reasons: tuple[str, ...]
    creation_timestamp: str
    guarantee_version: str = GUARANTEE_VERSION
    schema_version: str = "1.0"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "guarantee_version": self.guarantee_version,
            "creation_timestamp": self.creation_timestamp,
            "contract_id": self.contract_id,
            "filter_id": self.filter_id,
            "film_count": self.film_count,
            "status": self.status.value,
            "applicable_filter_ids": list(self.applicable_filter_ids),
            "checks": dict(self.checks),
            "expected_contributor_hashes": list(self.expected_contributor_hashes),
            "observed_contributor_hashes": list(self.observed_contributor_hashes),
            "reasons": list(self.reasons),
        }


def applicable_multi_input_filter_ids(film_count: int) -> tuple[str, ...]:
    """Return every implemented filter whose declared arity accepts this world."""
    if film_count < 2:
        return ()
    rows = []
    for definition in default_filter_registry().definitions(implemented_only=True):
        if definition.minimum_films < 2:
            continue
        if film_count < definition.minimum_films:
            continue
        if definition.maximum_films is not None and film_count > definition.maximum_films:
            continue
        rows.append(definition.id)
    return tuple(sorted(rows))


def certify_multi_input_schedule(
    *,
    contract: RunContract,
    schedule: dict[str, Any],
    qualification: ScheduleQualification,
) -> MultiInputGuarantee:
    """Gate rendering after analysis, when material capabilities are knowable."""
    registry = default_filter_registry()
    definition = registry.get(contract.filter_id)
    film_count = len(contract.media)
    applicable = applicable_multi_input_filter_ids(film_count)
    enabled = [row for row in schedule.get("mappings", []) if row.get("enabled", True)]
    known_hashes = {row.media_hash for row in contract.media}
    donor_hashes = {row.media_hash for row in contract.media[1:]}
    expected_hashes = known_hashes if contract.filter_id == "multiworld.echo_chamber" else donor_hashes
    observed_hashes = {
        str(row.get("source_media_hash"))
        for row in enabled
        if row.get("source_media_hash")
    }
    if schedule.get("source_media_hash"):
        observed_hashes.add(str(schedule["source_media_hash"]))
    observed_hashes.update(str(value) for value in schedule.get("source_media_hashes", []) if value)
    identity_quality = schedule.get("identity_quality")
    identity_sufficient = (
        not definition.requires_speaker_identity
        or isinstance(identity_quality, dict) and identity_quality.get("passed") is True
    )
    checks = {
        "two_or_more_complete_inputs": film_count >= 2 and contract.input_scope == "complete_media_files",
        "filter_is_implemented": definition.implemented,
        "filter_is_applicable_to_arity": contract.filter_id in applicable,
        "every_input_has_video": all(bool(row.video_streams) for row in contract.media),
        "every_input_has_audio": all(bool(row.audio_streams) for row in contract.media),
        "canonical_extent_positive": contract.timeline.duration > 0,
        "schedule_qualification_not_blocked": qualification.status != QualificationStatus.BLOCKED,
        "qualified_placement_exists": bool(enabled),
        "all_required_inputs_contribute": expected_hashes <= observed_hashes,
        "no_unknown_input_contributes": observed_hashes <= known_hashes,
        "speaker_identity_capability_sufficient": identity_sufficient,
    }
    reasons = tuple(_reason_for(name) for name, passed in checks.items() if not passed)
    status = (
        MultiInputGuaranteeStatus.READY_TO_RENDER
        if all(checks.values())
        else MultiInputGuaranteeStatus.REJECTED
    )
    result = MultiInputGuarantee(
        status=status,
        contract_id=contract.contract_id,
        filter_id=contract.filter_id,
        film_count=film_count,
        applicable_filter_ids=applicable,
        checks=checks,
        expected_contributor_hashes=tuple(sorted(expected_hashes)),
        observed_contributor_hashes=tuple(sorted(observed_hashes)),
        reasons=reasons,
        creation_timestamp=utc_now(),
    )
    schedule["multi_input_guarantee"] = result.to_dict()
    return result


def write_multi_input_guarantee(
    guarantee: MultiInputGuarantee,
    output_path: Path,
    schemas_dir: Path,
) -> Path:
    data = guarantee.to_dict()
    _validate_object(data, read_json(schemas_dir / "multi_input_guarantee.schema.json"), str(output_path))
    write_json(output_path, data)
    return output_path


def _reason_for(check: str) -> str:
    return {
        "two_or_more_complete_inputs": "The experiment does not contain two or more complete inputs.",
        "filter_is_implemented": "The selected filter is not executable.",
        "filter_is_applicable_to_arity": "The selected filter does not accept this number of films.",
        "every_input_has_video": "At least one input has no usable video stream.",
        "every_input_has_audio": "At least one input has no usable audio stream.",
        "canonical_extent_positive": "The required streams do not support a positive output extent.",
        "schedule_qualification_not_blocked": "The authored schedule failed contract qualification.",
        "qualified_placement_exists": "Analysis produced no renderable cross-film placement.",
        "all_required_inputs_contribute": "One or more required films contributes no qualified material.",
        "no_unknown_input_contributes": "The schedule refers to material outside the selected inputs.",
        "speaker_identity_capability_sufficient": "Direct speaker evidence is insufficient for this filter.",
    }[check]


