from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

from cinelingus.util import utc_now

from .contracts import FilterContractCatalog, default_contract_catalog
from .registry import FilterRegistry, default_filter_registry


COMPATIBILITY_COMPILER_VERSION = "combination_compatibility_v1"
BLOOM_FILTER_ID = "experimental.bloom"


class CombinationStatus(str, Enum):
    CERTIFIED = "CERTIFIED"
    COMPATIBLE_UNPROVEN = "COMPATIBLE_UNPROVEN"
    REQUIRES_REANALYSIS = "REQUIRES_REANALYSIS"
    INCOMPATIBLE = "INCOMPATIBLE"
    UNAVAILABLE = "UNAVAILABLE"


@dataclass(frozen=True)
class CombinationDecision:
    predecessor_filter_id: str
    successor_filter_id: str
    status: CombinationStatus
    executable: bool
    checks: dict[str, bool]
    shared_relationship_domains: tuple[str, ...]
    reasons: tuple[str, ...]
    evidence: dict[str, Any]
    compiler_version: str = COMPATIBILITY_COMPILER_VERSION

    @property
    def pair_id(self) -> str:
        return f"{self.predecessor_filter_id}->{self.successor_filter_id}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "pair_id": self.pair_id,
            "predecessor_filter_id": self.predecessor_filter_id,
            "successor_filter_id": self.successor_filter_id,
            "status": self.status.value,
            "executable": self.executable,
            "checks": dict(self.checks),
            "shared_relationship_domains": list(self.shared_relationship_domains),
            "reasons": list(self.reasons),
            "evidence": dict(self.evidence),
            "compiler_version": self.compiler_version,
        }

    def recipe_record(self) -> dict[str, str]:
        return {
            "filters": f"{self.predecessor_filter_id} -> {self.successor_filter_id}",
            "decision": self.status.value,
            "compiler_version": self.compiler_version,
        }


def compile_ordered_combination(
    predecessor_filter_id: str,
    successor_filter_id: str,
    *,
    registry: FilterRegistry | None = None,
    catalog: FilterContractCatalog | None = None,
    certification_evidence: Mapping[str, Mapping[str, Any]] | None = None,
) -> CombinationDecision:
    """Compile one ordered pair from declared capabilities and Procedure evidence."""
    registry = registry or default_filter_registry()
    catalog = catalog or default_contract_catalog()
    predecessor = registry.get(predecessor_filter_id)
    successor = registry.get(successor_filter_id)
    predecessor_contract = catalog.get(predecessor.id, registry).data
    successor_contract = catalog.get(successor.id, registry).data
    procedure = dict(successor_contract.get("procedure_behavior") or {})
    shared_domains = tuple(sorted(
        {item.value for item in predecessor.changes_dimensions}
        & {item.value for item in successor.reads_dimensions}
    ))
    explicit_incompatibility = (
        successor.id in predecessor.incompatible_filters
        or predecessor.id in successor.incompatible_filters
    )
    order_allowed = predecessor.id != BLOOM_FILTER_ID and successor.id == BLOOM_FILTER_ID
    one_primary_shape = successor.id == BLOOM_FILTER_ID and predecessor.id != BLOOM_FILTER_ID
    predecessor_state_declared = "filter_plan" in predecessor.intermediate_products
    receives_transformed = procedure.get("receives_transformed_specimen") is True
    multi_step_validated = procedure.get("support_status") == "multi_step_validated"
    pair_id = f"{predecessor.id}->{successor.id}"
    evidence = dict((certification_evidence or {}).get(pair_id) or {})
    evidence_passed = evidence.get("status") == "PASS"
    checks = {
        "distinct_filters": predecessor.id != successor.id,
        "predecessor_implemented": predecessor.implemented,
        "successor_implemented": successor.implemented,
        "no_explicit_incompatibility": not explicit_incompatibility,
        "supported_order": order_allowed,
        "one_primary_plus_bloom_shape": one_primary_shape,
        "predecessor_normalized_state_declared": predecessor_state_declared,
        "successor_receives_transformed_specimen": receives_transformed,
        "successor_multi_step_validated": multi_step_validated,
        "certification_evidence_passed": evidence_passed,
    }

    reasons: list[str] = []
    if not predecessor.implemented or not successor.implemented:
        status = CombinationStatus.UNAVAILABLE
        reasons.append("One or both filters are not implemented.")
    elif predecessor.id == successor.id:
        status = CombinationStatus.INCOMPATIBLE
        reasons.append("Repeated application of the same filter has no validated Procedure contract.")
    elif explicit_incompatibility:
        status = CombinationStatus.INCOMPATIBLE
        reasons.append("The registry explicitly declares these filters incompatible.")
    elif predecessor.id == BLOOM_FILTER_ID:
        status = CombinationStatus.INCOMPATIBLE
        reasons.append("Bloom is a successor progression modifier and cannot precede a primary filter.")
    elif successor.id != BLOOM_FILTER_ID:
        if shared_domains and not receives_transformed:
            status = CombinationStatus.REQUIRES_REANALYSIS
            reasons.append(
                "The successor reads state changed by the predecessor but does not consume a transformed specimen."
            )
        else:
            status = CombinationStatus.INCOMPATIBLE
            reasons.append("The current runtime permits only one primary filter followed by Bloom.")
    elif not predecessor_state_declared:
        status = CombinationStatus.INCOMPATIBLE
        reasons.append("The predecessor does not expose normalized state for a later operator.")
    elif not receives_transformed or not multi_step_validated:
        status = CombinationStatus.COMPATIBLE_UNPROVEN
        reasons.append("The ordered shape is supported, but Bloom has not validated transformed-specimen execution.")
    elif not evidence_passed:
        status = CombinationStatus.COMPATIBLE_UNPROVEN
        reasons.append("The Procedure contract is eligible but no passing combination certification was supplied.")
    else:
        status = CombinationStatus.CERTIFIED

    return CombinationDecision(
        predecessor_filter_id=predecessor.id,
        successor_filter_id=successor.id,
        status=status,
        executable=status == CombinationStatus.CERTIFIED,
        checks=checks,
        shared_relationship_domains=shared_domains,
        reasons=tuple(reasons),
        evidence={
            "predecessor_contract_version": predecessor_contract.get("contract_version"),
            "successor_contract_version": successor_contract.get("contract_version"),
            "successor_procedure_support": procedure.get("support_status"),
            "successor_receives_transformed_specimen": receives_transformed,
            "certification": evidence,
        },
    )


def compile_compatibility_matrix(
    *,
    registry: FilterRegistry | None = None,
    catalog: FilterContractCatalog | None = None,
    certification_evidence: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    registry = registry or default_filter_registry()
    catalog = catalog or default_contract_catalog()
    definitions = registry.definitions()
    decisions = [
        compile_ordered_combination(
            first.id,
            second.id,
            registry=registry,
            catalog=catalog,
            certification_evidence=certification_evidence,
        )
        for first in definitions
        for second in definitions
        if first.id != second.id
    ]
    rows = [decision.to_dict() for decision in decisions]
    counts = {
        status.value: sum(decision.status == status for decision in decisions)
        for status in CombinationStatus
    }
    signature_payload = {
        "compiler_version": COMPATIBILITY_COMPILER_VERSION,
        "filter_ids": sorted(definition.id for definition in definitions),
        "decisions": rows,
    }
    signature = hashlib.sha256(
        json.dumps(signature_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": "1.0",
        "compiler_version": COMPATIBILITY_COMPILER_VERSION,
        "creation_timestamp": utc_now(),
        "matrix_signature": signature,
        "filter_count": len(definitions),
        "ordered_pair_count": len(decisions),
        "status_counts": counts,
        "executable_pair_ids": [decision.pair_id for decision in decisions if decision.executable],
        "decisions": rows,
    }
