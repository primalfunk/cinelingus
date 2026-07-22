from __future__ import annotations

from dataclasses import replace

import pytest

from cinelingus.filter_lab.gui_controller import current_filter_definition, sync_filter_family
from cinelingus.filter_lab.public_catalog import (
    ApparatusStatus,
    OperatingMode,
    PublicApparatusCatalog,
    default_public_apparatus_catalog,
)
from cinelingus.filter_lab.registry import default_filter_registry


class Variable:
    def __init__(self, value: str) -> None:
        self.value = value

    def get(self) -> str:
        return self.value

    def set(self, value: str) -> None:
        self.value = value


class ChoiceBox:
    def __init__(self) -> None:
        self.values: list[str] = []

    def configure(self, *, values) -> None:
        self.values = list(values)


def test_catalog_has_exact_registry_parity_and_six_public_disciplines() -> None:
    registry = default_filter_registry()
    catalog = default_public_apparatus_catalog()

    assert {entry.internal_id for entry in catalog.entries()} == {item.id for item in registry.definitions()}
    assert [item.name for item in catalog.disciplines()] == [
        "Chronomancy Engine", "Contagion Laboratory", "Memory Palace",
        "Mask Workshop", "Alchemical Engine", "Lexicon",
    ]
    assert "Multiworld" not in {item.name for item in catalog.disciplines()}


def test_public_renames_resolve_without_changing_internal_ids() -> None:
    catalog = default_public_apparatus_catalog()

    assert catalog.resolve("Premonition", operating_mode="solitary").internal_id == "time.foreshadow"
    assert catalog.resolve("Foreshadow", operating_mode="solitary").internal_id == "time.foreshadow"
    assert catalog.resolve("Echoes", operating_mode="solitary").internal_id == "time.flashback"
    assert catalog.resolve("Flashback", operating_mode="solitary").internal_id == "time.flashback"
    assert catalog.resolve("Transposition", operating_mode="multiworld").internal_id == "multiworld.translation"
    assert catalog.resolve("Translation/Transposition").internal_id == "multiworld.translation"


def test_echo_and_echoes_remain_distinct_apparatuses() -> None:
    catalog = default_public_apparatus_catalog()

    echo = catalog.resolve("Echo", operating_mode="solitary", discipline="alchemy")
    echoes = catalog.resolve("Echoes", operating_mode="solitary", discipline="chronomancy")

    assert echo.internal_id == "translation.echo"
    assert echoes.internal_id == "time.flashback"


def test_duplicate_names_require_public_context() -> None:
    catalog = default_public_apparatus_catalog()

    with pytest.raises(ValueError, match="ambiguous"):
        catalog.resolve("Possession")
    assert catalog.resolve("Possession", operating_mode=OperatingMode.SOLITARY).internal_id == "identity.possession"
    assert catalog.resolve("Possession", operating_mode=OperatingMode.MULTIWORLD).internal_id == "multiworld.possession"


def test_dormant_entries_are_not_primary_or_invokable() -> None:
    catalog = default_public_apparatus_catalog()
    dormant = catalog.get("multiworld.wormhole")

    assert dormant.status == ApparatusStatus.DORMANT
    assert dormant.visible_in_primary_catalog is False
    assert dormant not in catalog.entries(primary_only=True)
    with pytest.raises(ValueError, match="dormant"):
        dormant.require_invokable()


def test_catalog_rejects_status_that_disagrees_with_engineering_truth() -> None:
    catalog = default_public_apparatus_catalog()
    entries = list(catalog.entries())
    index = next(index for index, entry in enumerate(entries) if entry.internal_id == "multiworld.wormhole")
    entries[index] = replace(entries[index], status=ApparatusStatus.AVAILABLE)

    with pytest.raises(ValueError, match="Availability mismatch"):
        PublicApparatusCatalog(entries)


def test_film_count_and_truthful_proxy_disclosures() -> None:
    catalog = default_public_apparatus_catalog()

    catalog.get("multiworld.chimera").validate_film_count(3)
    with pytest.raises(ValueError, match="at least 3 films"):
        catalog.get("multiworld.chimera").validate_film_count(2)
    assert "lexical proxy" in catalog.get("emotion.wonder").public_description
    assert "not semantic understanding" in catalog.get("memory.dream").public_description
    assert "does not convert accents" in catalog.get("infection.dialect").public_description


def test_gui_population_uses_public_mode_and_discipline(monkeypatch) -> None:
    app = type("App", (), {})()
    app.operating_mode_var = Variable("Several Films")
    app.family_var = Variable("Mask Workshop")
    app.mode_var = Variable("Possession")
    app.mode_box = ChoiceBox()
    monkeypatch.setattr("cinelingus.filter_lab.gui_controller.sync_filter_mode", lambda _app: None)

    sync_filter_family(app)

    assert app.mode_box.values == ["Possession"]
    assert current_filter_definition(app).id == "multiworld.possession"
