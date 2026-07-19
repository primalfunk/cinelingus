from copy import deepcopy

from cinelingus.filter_lab.combination import CombinationStatus, compile_ordered_combination
from cinelingus.filter_lab.contracts import FilterContract, FilterContractCatalog, default_contract_catalog


PAIR_ID = "multiworld.translation->experimental.bloom"


def test_passing_evidence_cannot_override_a_single_step_contract() -> None:
    decision = compile_ordered_combination(
        "multiworld.translation",
        "experimental.bloom",
        certification_evidence={PAIR_ID: {"status": "PASS", "artifact": "fake.json"}},
    )
    assert decision.status == CombinationStatus.COMPATIBLE_UNPROVEN
    assert decision.executable is False


def test_multi_step_contract_plus_passing_evidence_is_required_for_certification() -> None:
    contracts = []
    for contract in default_contract_catalog().contracts():
        data = deepcopy(contract.data)
        if contract.filter_id == "experimental.bloom":
            data["procedure_behavior"] = {
                **data["procedure_behavior"],
                "support_status": "multi_step_validated",
                "receives_transformed_specimen": True,
            }
        contracts.append(FilterContract(path=contract.path, data=data))
    catalog = FilterContractCatalog(contracts)
    decision = compile_ordered_combination(
        "multiworld.translation",
        "experimental.bloom",
        catalog=catalog,
        certification_evidence={PAIR_ID: {"status": "PASS", "artifact": "procedure_certification.json"}},
    )
    assert decision.status == CombinationStatus.CERTIFIED
    assert decision.executable is True
    assert decision.recipe_record()["decision"] == "CERTIFIED"
