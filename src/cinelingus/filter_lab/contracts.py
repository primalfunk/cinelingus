from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from cinelingus.util import read_json
from cinelingus.validation import ValidationError, validate_artifact

from .models import FilterDefinition
from .registry import FilterRegistry, default_filter_registry
from .strategies import has_strategy


UNAVAILABLE_CLASS_A_ARTIFACTS = {
    "semantic_features", "semantic_embeddings", "emotional_features",
    "voice_conversion", "dsp_features", "procedure_state",
}
PLACEHOLDER_MARKERS = ("planned strategy", "placeholder", "tbd", "to be defined")


@dataclass(frozen=True)
class FilterContract:
    path: Path
    data: dict[str, Any]

    @property
    def filter_id(self) -> str:
        return str(self.data["filter_id"])

    @property
    def status(self) -> str:
        return str(self.data["status"])

    @property
    def capabilities(self) -> dict[str, Any]:
        return dict(self.data["capabilities"])

    @property
    def multiworld(self) -> dict[str, Any] | None:
        value = self.data.get("multiworld_contract")
        return dict(value) if isinstance(value, dict) else None

    def to_dict(self) -> dict[str, Any]:
        return dict(self.data)


class FilterContractCatalog:
    def __init__(self, contracts: Iterable[FilterContract]) -> None:
        self._by_id: dict[str, FilterContract] = {}
        for contract in contracts:
            if contract.filter_id in self._by_id:
                first = self._by_id[contract.filter_id].path
                raise ValidationError(f"Duplicate contracts for {contract.filter_id}: {first} and {contract.path}")
            self._by_id[contract.filter_id] = contract

    def get(self, filter_id_or_alias: str, registry: FilterRegistry | None = None) -> FilterContract:
        resolved = (registry or default_filter_registry()).get(filter_id_or_alias).id
        try:
            return self._by_id[resolved]
        except KeyError as exc:
            raise ValidationError(f"No filter contract exists for {resolved}.") from exc

    def contracts(self) -> tuple[FilterContract, ...]:
        return tuple(sorted(self._by_id.values(), key=lambda row: row.filter_id))

    def validate_registry_parity(self, registry: FilterRegistry | None = None) -> None:
        registry = registry or default_filter_registry()
        definitions = {row.id: row for row in registry.definitions()}
        contract_ids = set(self._by_id)
        missing = sorted(set(definitions) - contract_ids)
        extra = sorted(contract_ids - set(definitions))
        if missing or extra:
            details = []
            if missing:
                details.append(f"missing contracts: {', '.join(missing)}")
            if extra:
                details.append(f"orphan contracts: {', '.join(extra)}")
            raise ValidationError("; ".join(details))
        for filter_id, definition in definitions.items():
            _validate_contract_against_definition(self._by_id[filter_id], definition)

    def to_runtime_catalog(self) -> dict[str, dict[str, Any]]:
        return {contract.filter_id: contract.to_dict() for contract in self.contracts()}

    def render_markdown(self) -> str:
        lines = [
            "# Cinelingus Filter Contract Catalog",
            "",
            "Generated from the machine-valid contracts in filter_contracts/.",
            "",
            "This is the engineering contract view. Stable filter IDs, internal families, execution modes, and artifact names remain unchanged for compatibility. Public names, disciplines, operating modes, availability, capability tiers, and single-operator Procedure status are defined in [the public apparatus catalog](architecture/public_apparatus_catalog.md). Where the names differ, the public catalog controls presentation and this document controls executable behavior.",
            "",
            "| Filter | Family | Films | Cinematic law | Status | Execution | Contract proposition |",
            "|---|---|---:|---|---:|---|---|",
        ]
        for contract in self.contracts():
            row = contract.data
            proposition = str(row["creative_proposition"]).replace("|", "\\|")
            multiworld = row.get("multiworld_contract") or {}
            maximum = multiworld.get("maximum_films")
            film_range = "1" if not multiworld else (
                str(multiworld["minimum_films"]) if maximum == multiworld["minimum_films"]
                else f"{multiworld['minimum_films']}+" if maximum is None
                else f"{multiworld['minimum_films']}-{maximum}"
            )
            lines.append(
                f"| {row['filter_name']} | {row['family_id']} | {film_range} | "
                f"{multiworld.get('cinematic_law', definition_law(row))} | {row['status']} | "
                f"{row['capabilities']['execution_mode']} | {proposition} |"
            )
        lines.extend([
            "",
            "## Procedure status",
            "",
            "Procedure Behaviour entries are architectural commitments. No contract in this catalog claims multi-step runtime support.",
            "",
        ])
        return "\n".join(lines)


def load_contract_catalog(
    contracts_dir: Path,
    schemas_dir: Path,
    *,
    registry: FilterRegistry | None = None,
) -> FilterContractCatalog:
    if not contracts_dir.exists():
        raise ValidationError(f"Filter contract directory does not exist: {contracts_dir}")
    paths = sorted(contracts_dir.glob("*/*.json"))
    if not paths:
        raise ValidationError(f"No filter contracts found beneath {contracts_dir}")
    contracts = []
    for path in paths:
        validate_artifact("filter_contract", path, schemas_dir)
        contracts.append(FilterContract(path=path, data=read_json(path)))
    catalog = FilterContractCatalog(contracts)
    catalog.validate_registry_parity(registry)
    return catalog


@lru_cache(maxsize=1)
def default_contract_catalog() -> FilterContractCatalog:
    root = Path(__file__).resolve().parents[3]
    return load_contract_catalog(root / "filter_contracts", root / "schemas")


def _validate_contract_against_definition(contract: FilterContract, definition: FilterDefinition) -> None:
    data = contract.data
    expected = {
        "filter_name": definition.name,
        "family_id": definition.family_id,
        "filter_version": definition.version,
        "implementation_class": definition.implementation_class,
    }
    for field, value in expected.items():
        if data[field] != value:
            raise ValidationError(
                f"{contract.path}: {field} is {data[field]!r}; registry declares {value!r}."
            )
    reads = [row.value for row in definition.reads_dimensions]
    changes = [row.value for row in definition.changes_dimensions]
    if data["relationship_domains"]["reads"] != reads:
        raise ValidationError(f"{contract.path}: relationship reads do not match the registry.")
    if data["relationship_domains"]["changes"] != changes:
        raise ValidationError(f"{contract.path}: relationship changes do not match the registry.")
    if set(data["specimen_inputs"]) != set(definition.required_inputs):
        raise ValidationError(f"{contract.path}: specimen inputs do not match the registry.")
    if set(data["required_analysis"]) != set(definition.required_artifacts):
        raise ValidationError(f"{contract.path}: required analysis does not match the registry.")
    capabilities = data["capabilities"]
    capability_expectations = {
        "implemented": definition.implemented,
        "implementation_key": definition.implementation_key,
        "execution_mode": definition.execution_mode,
        "sparse_schedule": definition.sparse_schedule,
        "requires_speaker_identity": definition.requires_speaker_identity,
        "supports_preview": definition.supports_preview,
        "supports_full_length": "full_length" in definition.supported_output_forms,
        "requires_output_acceptance": definition.requires_output_acceptance,
    }
    for field, value in capability_expectations.items():
        if capabilities[field] != value:
            raise ValidationError(
                f"{contract.path}: capability {field} is {capabilities[field]!r}; registry declares {value!r}."
            )
    if definition.implemented and contract.status != "accepted":
        raise ValidationError(f"{contract.path}: implemented filters must have accepted contracts.")
    if contract.status == "accepted":
        serialized = repr(data).lower()
        marker = next((item for item in PLACEHOLDER_MARKERS if item in serialized), None)
        if marker:
            raise ValidationError(f"{contract.path}: accepted contract contains placeholder language '{marker}'.")
        empty_validators = [row["id"] for row in data["hard_invariants"] if not row["validator"].strip()]
        if empty_validators:
            raise ValidationError(f"{contract.path}: accepted invariants need validators: {', '.join(empty_validators)}")
    if definition.implementation_class == "A":
        unavailable = sorted(set(data["required_analysis"]) & UNAVAILABLE_CLASS_A_ARTIFACTS)
        if unavailable:
            raise ValidationError(f"{contract.path}: Class A requests unavailable artifacts: {', '.join(unavailable)}")
    if definition.execution_mode == "scheduling_strategy" and not has_strategy(str(definition.implementation_key)):
        raise ValidationError(f"{contract.path}: declared scheduling strategy is not registered.")
    if definition.is_multiworld:
        multiworld = data.get("multiworld_contract") or {}
        expected_multiworld = {
            "cinematic_law": definition.cinematic_law,
            "minimum_films": definition.minimum_films,
            "maximum_films": definition.maximum_films,
            "anchor_behavior": definition.anchor_behavior,
            "affected_elements": list(definition.affected_elements),
            "quality_requirements": list(definition.quality_requirements),
            "deterministic_seed_support": definition.deterministic_seed_support,
        }
        for field, value in expected_multiworld.items():
            if multiworld.get(field) != value:
                raise ValidationError(f"{contract.path}: multiworld {field} does not match the registry.")
        interface = multiworld.get("interface") or {}
        interface_expectations = {
            "inputs": list(definition.required_inputs),
            "outputs": list(definition.output_artifacts),
            "affected_artifacts": list(definition.affected_artifacts),
            "intermediate_products": list(definition.intermediate_products),
        }
        for field, value in interface_expectations.items():
            if interface.get(field) != value:
                raise ValidationError(f"{contract.path}: multiworld interface {field} does not match the registry.")


def definition_law(row: dict[str, Any]) -> str:
    return str(row.get("creative_proposition", "Internal Transformation")).split(".", 1)[0]
