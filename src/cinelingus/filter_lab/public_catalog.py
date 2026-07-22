from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache
from typing import Iterable

from .models import FilterDefinition
from .registry import FilterRegistry, default_filter_registry


class OperatingMode(StrEnum):
    SOLITARY = "solitary"
    MULTIWORLD = "multiworld"


class ApparatusStatus(StrEnum):
    AVAILABLE = "available"
    EXPERIMENTAL = "experimental"
    DORMANT = "dormant"
    BLOCKED = "blocked"
    HIDDEN = "hidden"


class CapabilityTier(StrEnum):
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"
    F = "F"


class ProcedureEligibility(StrEnum):
    SINGLE_ONLY = "single_only"


@dataclass(frozen=True)
class PublicDiscipline:
    id: str
    name: str
    motto: str
    display_order: int


@dataclass(frozen=True)
class PublicApparatusEntry:
    internal_id: str
    public_name: str
    discipline: str
    operating_mode: OperatingMode
    status: ApparatusStatus
    visible_in_primary_catalog: bool
    minimum_films: int
    maximum_films: int | None
    short_law: str
    public_description: str
    compatibility_aliases: tuple[str, ...]
    capability_tags: tuple[str, ...]
    minimum_capability_tier: CapabilityTier
    unavailable_reason: str | None
    display_order: int
    procedure_eligibility: ProcedureEligibility = ProcedureEligibility.SINGLE_ONLY

    @property
    def invokable(self) -> bool:
        return self.status in {ApparatusStatus.AVAILABLE, ApparatusStatus.EXPERIMENTAL}

    def validate_film_count(self, count: int) -> None:
        if count < self.minimum_films:
            raise ValueError(f"{self.public_name} requires at least {self.minimum_films} films; received {count}.")
        if self.maximum_films is not None and count > self.maximum_films:
            raise ValueError(f"{self.public_name} accepts at most {self.maximum_films} films; received {count}.")

    def require_invokable(self) -> None:
        if not self.invokable:
            reason = self.unavailable_reason or "This apparatus is not available for invocation."
            raise ValueError(f"{self.public_name} is {self.status.value}. {reason}")


DISCIPLINES = (
    PublicDiscipline("chronomancy", "Chronomancy Engine", "Time is wrong.", 0),
    PublicDiscipline("contagion", "Contagion Laboratory", "Films behave like diseases.", 1),
    PublicDiscipline("memory", "Memory Palace", "Films remember incorrectly.", 2),
    PublicDiscipline("mask", "Mask Workshop", "Identity is fluid.", 3),
    PublicDiscipline("alchemy", "Alchemical Engine", "The substance of cinema changes.", 4),
    PublicDiscipline("lexicon", "Lexicon", "Words reshape reality.", 5),
)


_PUBLIC_PLACEMENT = {
    "translation.drift": ("Drift", "chronomancy"),
    "time.foreshadow": ("Premonition", "chronomancy"),
    "time.flashback": ("Echoes", "chronomancy"),
    "time.mobius": ("Möbius", "chronomancy"),
    "time.spiral": ("Spiral", "chronomancy"),
    "experimental.ouroboros": ("Ouroboros", "chronomancy"),
    "infection.contagion": ("Contagion", "contagion"),
    "infection.dialect": ("Dialect", "contagion"),
    "infection.mutation": ("Mutation", "contagion"),
    "infection.whisper": ("Whisper", "contagion"),
    "memory.amnesia": ("Amnesia", "memory"),
    "memory.dream": ("Dream", "memory"),
    "memory.recollection": ("Recollection", "memory"),
    "identity.chorus": ("Chorus", "mask"),
    "identity.doppelganger": ("Doppelgänger", "mask"),
    "identity.possession": ("Possession", "mask"),
    "identity.split_personality": ("Split Personality", "mask"),
    "translation.echo": ("Echo", "alchemy"),
    "translation.self_shuffle": ("Self Shuffle", "alchemy"),
    "experimental.bloom": ("Bloom", "alchemy"),
    "experimental.shed_skin": ("Shed Skin", "alchemy"),
    "experimental.venom": ("Venom", "alchemy"),
    "emotion.exhaustion": ("Exhaustion", "alchemy"),
    "emotion.optimist": ("Optimist", "lexicon"),
    "emotion.paranoia": ("Paranoia", "lexicon"),
    "emotion.regret": ("Regret", "lexicon"),
    "emotion.wonder": ("Wonder", "lexicon"),
    "multiworld.translation": ("Transposition", "alchemy"),
    "multiworld.possession": ("Possession", "mask"),
    "multiworld.contagion": ("Contagion", "contagion"),
    "multiworld.echo_chamber": ("Echo Chamber", "alchemy"),
    "multiworld.prophecy": ("Prophecy", "chronomancy"),
    "multiworld.bleed": ("Bleed", "alchemy"),
    "multiworld.chimera": ("Chimera", "alchemy"),
    "multiworld.civilization": ("Civilization", "memory"),
    "multiworld.doppelganger": ("Doppelgänger", "mask"),
    "multiworld.mirror_world": ("Mirror World", "mask"),
    "multiworld.parallel_universes": ("Parallel Universes", "chronomancy"),
    "multiworld.triangle": ("Triangle", "alchemy"),
    "multiworld.wormhole": ("Wormhole", "chronomancy"),
}

_DISPLAY_ORDER = {internal_id: index for index, internal_id in enumerate(_PUBLIC_PLACEMENT)}
_DISPLAY_ORDER.update({
    "infection.contagion": 6,
    "infection.mutation": 7,
    "infection.whisper": 8,
    "infection.dialect": 9,
    "identity.possession": 13,
    "identity.doppelganger": 14,
    "identity.chorus": 15,
    "identity.split_personality": 16,
    "emotion.wonder": 23,
    "emotion.regret": 24,
    "emotion.optimist": 25,
    "emotion.paranoia": 26,
})

_EXPLICIT_ALIASES = {
    "time.foreshadow": ("Foreshadow",),
    "time.flashback": ("Flashback",),
    "multiworld.translation": ("Translation", "Translation/Transposition"),
}

_DORMANT_TIERS = {
    "multiworld.bleed": CapabilityTier.D,
    "multiworld.chimera": CapabilityTier.E,
    "multiworld.civilization": CapabilityTier.F,
    "multiworld.doppelganger": CapabilityTier.D,
    "multiworld.mirror_world": CapabilityTier.D,
    "multiworld.parallel_universes": CapabilityTier.C,
    "multiworld.triangle": CapabilityTier.B,
    "multiworld.wormhole": CapabilityTier.C,
}

_TIER_B = {
    "translation.drift", "translation.echo", "infection.whisper", "emotion.exhaustion",
    "experimental.bloom", "experimental.shed_skin", "multiworld.echo_chamber",
}


def _status(definition: FilterDefinition) -> ApparatusStatus:
    if not definition.implemented:
        return ApparatusStatus.DORMANT
    return ApparatusStatus.EXPERIMENTAL if definition.experimental else ApparatusStatus.AVAILABLE


def _capability_tier(definition: FilterDefinition) -> CapabilityTier:
    if definition.id in _DORMANT_TIERS:
        return _DORMANT_TIERS[definition.id]
    return CapabilityTier.B if definition.id in _TIER_B else CapabilityTier.A


def _capability_tags(definition: FilterDefinition) -> tuple[str, ...]:
    tags = list(dict.fromkeys(definition.affected_elements))
    if definition.minimum_films > 1:
        tags.append("cross_film_relationship")
    if definition.requires_speaker_identity:
        tags.append("speaker_identity")
    if definition.sparse_schedule:
        tags.append("dialogue_selection")
    if definition.id in _TIER_B or definition.id == "multiworld.bleed":
        tags.append("audio_transformation")
    return tuple(dict.fromkeys(tags))


def _description(definition: FilterDefinition) -> str:
    text = definition.creative_description
    if definition.family_id == "emotion" and definition.id != "emotion.exhaustion":
        return f"{text} Selection follows a disclosed lexical proxy, not inferred emotion."
    if definition.id == "memory.dream":
        return f"{text} Associations use token overlap and temporal distance, not semantic understanding."
    if definition.id == "infection.dialect":
        return f"{text} This changes dialogue cadence selection; it does not convert accents."
    return text


def _aliases(definition: FilterDefinition, public_name: str) -> tuple[str, ...]:
    values = [*_EXPLICIT_ALIASES.get(definition.id, ()), *definition.legacy_aliases]
    if definition.name != public_name:
        values.append(definition.name)
    return tuple(dict.fromkeys(value for value in values if value != public_name))


class PublicApparatusCatalog:
    def __init__(self, entries: Iterable[PublicApparatusEntry], registry: FilterRegistry | None = None) -> None:
        self.registry = registry or default_filter_registry()
        self._entries = tuple(entries)
        self._by_internal_id = {entry.internal_id: entry for entry in self._entries}
        self._disciplines = {discipline.id: discipline for discipline in DISCIPLINES}
        self._validate()

    def _validate(self) -> None:
        if len(self._by_internal_id) != len(self._entries):
            raise ValueError("Every internal operator must have exactly one public apparatus entry.")
        expected = {definition.id for definition in self.registry.definitions()}
        actual = set(self._by_internal_id)
        if expected != actual:
            raise ValueError(f"Public apparatus parity mismatch; missing={sorted(expected - actual)}, extra={sorted(actual - expected)}")
        for entry in self._entries:
            definition = self.registry.get(entry.internal_id)
            if entry.discipline not in self._disciplines:
                raise ValueError(f"Unknown public discipline '{entry.discipline}' for {entry.internal_id}.")
            expected_mode = OperatingMode.MULTIWORLD if definition.minimum_films > 1 else OperatingMode.SOLITARY
            if entry.operating_mode != expected_mode:
                raise ValueError(f"Operating mode mismatch for {entry.internal_id}.")
            if (entry.minimum_films, entry.maximum_films) != (definition.minimum_films, definition.maximum_films):
                raise ValueError(f"Film-count mismatch for {entry.internal_id}.")
            if entry.invokable and not definition.implemented:
                raise ValueError(f"Availability mismatch for {entry.internal_id}.")
            if definition.implemented and entry.status in {ApparatusStatus.DORMANT, ApparatusStatus.BLOCKED}:
                raise ValueError(f"Availability mismatch for {entry.internal_id}.")
            if entry.status in {ApparatusStatus.DORMANT, ApparatusStatus.BLOCKED} and definition.implementation_key is not None:
                raise ValueError(f"Dormant apparatus {entry.internal_id} must not have an implementation key.")

    def disciplines(self) -> tuple[PublicDiscipline, ...]:
        return DISCIPLINES

    def discipline(self, discipline_id_or_name: str) -> PublicDiscipline:
        key = discipline_id_or_name.casefold()
        for discipline in DISCIPLINES:
            if key in {discipline.id.casefold(), discipline.name.casefold()}:
                return discipline
        raise ValueError(f"Unknown apparatus discipline '{discipline_id_or_name}'.")

    def entries(self, *, operating_mode: OperatingMode | str | None = None, discipline: str | None = None,
                primary_only: bool = False, invokable_only: bool = False) -> tuple[PublicApparatusEntry, ...]:
        mode = OperatingMode(operating_mode) if operating_mode is not None else None
        discipline_id = self.discipline(discipline).id if discipline is not None else None
        rows = (
            entry for entry in self._entries
            if (mode is None or entry.operating_mode == mode)
            and (discipline_id is None or entry.discipline == discipline_id)
            and (not primary_only or entry.visible_in_primary_catalog)
            and (not invokable_only or entry.invokable)
        )
        discipline_order = {item.id: item.display_order for item in DISCIPLINES}
        return tuple(sorted(rows, key=lambda item: (discipline_order[item.discipline], item.display_order, item.public_name)))

    def get(self, internal_id: str) -> PublicApparatusEntry:
        try:
            return self._by_internal_id[internal_id]
        except KeyError as exc:
            raise ValueError(f"Unknown internal apparatus id '{internal_id}'.") from exc

    def resolve(self, name_or_id: str, *, operating_mode: OperatingMode | str | None = None,
                discipline: str | None = None) -> PublicApparatusEntry:
        if name_or_id in self._by_internal_id:
            candidates = (self._by_internal_id[name_or_id],)
        else:
            key = name_or_id.casefold()
            candidates = tuple(
                entry for entry in self._entries
                if key == entry.public_name.casefold()
                or key in {alias.casefold() for alias in entry.compatibility_aliases}
            )
        if operating_mode is not None:
            mode = OperatingMode(operating_mode)
            candidates = tuple(entry for entry in candidates if entry.operating_mode == mode)
        if discipline is not None:
            discipline_id = self.discipline(discipline).id
            candidates = tuple(entry for entry in candidates if entry.discipline == discipline_id)
        if not candidates:
            raise ValueError(f"Unknown public apparatus '{name_or_id}'.")
        if len(candidates) > 1:
            raise ValueError(f"Public apparatus '{name_or_id}' is ambiguous; specify operating mode and discipline.")
        return candidates[0]


def _build_entries(registry: FilterRegistry) -> tuple[PublicApparatusEntry, ...]:
    entries = []
    for definition in registry.definitions():
        public_name, discipline = _PUBLIC_PLACEMENT[definition.id]
        mode = OperatingMode.MULTIWORLD if definition.minimum_films > 1 else OperatingMode.SOLITARY
        status = _status(definition)
        tier = _capability_tier(definition)
        entries.append(PublicApparatusEntry(
            internal_id=definition.id,
            public_name=public_name,
            discipline=discipline,
            operating_mode=mode,
            status=status,
            visible_in_primary_catalog=definition.implemented,
            minimum_films=definition.minimum_films,
            maximum_films=definition.maximum_films,
            short_law=definition.creative_description,
            public_description=_description(definition),
            compatibility_aliases=_aliases(definition, public_name),
            capability_tags=_capability_tags(definition),
            minimum_capability_tier=tier,
            unavailable_reason=(f"Requires capability tier {tier.value} machinery not yet available through this apparatus." if not definition.implemented else None),
            display_order=_DISPLAY_ORDER[definition.id],
        ))
    return tuple(entries)


@lru_cache(maxsize=1)
def default_public_apparatus_catalog() -> PublicApparatusCatalog:
    registry = default_filter_registry()
    return PublicApparatusCatalog(_build_entries(registry), registry)
