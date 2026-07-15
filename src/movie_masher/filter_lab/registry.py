from __future__ import annotations

from functools import lru_cache

from .models import FilterDefinition, FilterFamilyDefinition, FilterParameter, RelationshipDimension as D


INTENSITY = FilterParameter(
    "intensity", "Intensity", "choice", "Moderate",
    "How often or strongly the filter intervenes.",
    ("Trace", "Light", "Moderate", "Strong", "Total"),
)
PROGRESSION = FilterParameter(
    "progression", "Progression", "choice", "Constant",
    "How transformation intensity changes through the output.",
    ("Constant", "Increasing", "Decreasing", "Pulsing", "Scene-based"),
)
PERFORMANCE = FilterParameter(
    "performance_preservation", "Performance preservation", "choice", "Similar performance",
    "How closely replacement delivery should match the destination performance.",
    ("Exact rhythm preferred", "Similar performance", "Duration only", "Free transformation"),
)
SEMANTIC = FilterParameter(
    "semantic_relationship", "Semantic relationship", "choice", "Similar",
    "How source and destination meanings should relate.",
    ("Similar", "Complementary", "Contradictory", "Unrelated"),
)
ALLOW_REUSE = FilterParameter("allow_line_reuse", "Allow line reuse", "boolean", False, "Permit a source line to be used more than once.", advanced=True)

PLANNED_IMPLEMENTATION_CLASSES = {
    "infection.whisper": "C",
    "infection.mutation": "F",
    "infection.dialect": "C",
    "identity.split_personality": "A",
    "memory.dream": "B",
    "memory.recollection": "B",
    "memory.amnesia": "A",
    "emotion.wonder": "B",
    "emotion.regret": "B",
    "emotion.optimist": "B",
    "emotion.paranoia": "B",
    "emotion.exhaustion": "C",
    "time.mobius": "A",
    "experimental.venom": "B",
    "experimental.shed_skin": "D",
    "experimental.ouroboros": "E",
}


class FilterRegistry:
    def __init__(self, families: tuple[FilterFamilyDefinition, ...], definitions: tuple[FilterDefinition, ...]) -> None:
        self._families = {family.id: family for family in families}
        self._definitions = {definition.id: definition for definition in definitions}
        self._aliases: dict[str, str] = {}
        name_counts: dict[str, int] = {}
        for definition in definitions:
            key = self._key(definition.name)
            name_counts[key] = name_counts.get(key, 0) + 1
        for definition in definitions:
            aliases = [definition.id, *definition.legacy_aliases]
            if name_counts[self._key(definition.name)] == 1:
                aliases.append(definition.name)
            for alias in aliases:
                key = self._key(alias)
                existing = self._aliases.get(key)
                if existing is not None and existing != definition.id:
                    raise ValueError(f"Filter alias '{alias}' belongs to both {existing} and {definition.id}.")
                self._aliases[key] = definition.id
            if definition.family_id not in self._families:
                raise ValueError(f"Unknown family '{definition.family_id}' for {definition.id}.")

    @staticmethod
    def _key(value: str) -> str:
        return value.strip().lower().replace("-", "_").replace(" ", "_")

    def resolve_id(self, filter_id_or_alias: str) -> tuple[str, str | None]:
        key = self._key(filter_id_or_alias)
        try:
            resolved = self._aliases[key]
        except KeyError as exc:
            raise ValueError(f"Unknown filter '{filter_id_or_alias}'.") from exc
        migration = None if filter_id_or_alias == resolved else f"Migrated legacy filter identifier '{filter_id_or_alias}' to '{resolved}'."
        return resolved, migration

    def get(self, filter_id_or_alias: str) -> FilterDefinition:
        resolved, _migration = self.resolve_id(filter_id_or_alias)
        return self._definitions[resolved]

    def get_in_family(self, family_id: str, filter_name_or_id: str) -> FilterDefinition:
        key = self._key(filter_name_or_id)
        for definition in self.filters_for_family(family_id):
            if key in {self._key(definition.id), self._key(definition.name)}:
                return definition
        raise ValueError(f"Unknown filter '{filter_name_or_id}' in family '{family_id}'.")

    def families(self) -> tuple[FilterFamilyDefinition, ...]:
        return tuple(sorted(self._families.values(), key=lambda item: item.order))

    def family(self, family_id: str) -> FilterFamilyDefinition:
        return self._families[family_id]

    def definitions(self, *, implemented_only: bool = False) -> tuple[FilterDefinition, ...]:
        rows = tuple(self._definitions.values())
        if implemented_only:
            rows = tuple(item for item in rows if item.implemented)
        family_order = {item.id: item.order for item in self.families()}
        return tuple(sorted(rows, key=lambda item: (family_order[item.family_id], item.name)))

    def filters_for_family(self, family_id: str, *, implemented_only: bool = False) -> tuple[FilterDefinition, ...]:
        return tuple(item for item in self.definitions(implemented_only=implemented_only) if item.family_id == family_id)

    def validate_stack(self, filter_ids: list[str] | tuple[str, ...]) -> list[dict[str, str]]:
        if not filter_ids:
            raise ValueError("A filter recipe must contain at least one filter.")
        definitions = [self.get(item) for item in filter_ids]
        decisions: list[dict[str, str]] = []
        primary_count = sum(1 for item in definitions if item.id != "experimental.bloom")
        if primary_count > 1:
            raise ValueError("This release supports one primary filter; Bloom may be added only as a progression modifier.")
        for index, current in enumerate(definitions):
            if not current.implemented:
                raise ValueError(f"{current.name} is in development and cannot be run.")
            for other in definitions[index + 1:]:
                if other.id in current.incompatible_filters or current.id in other.incompatible_filters:
                    raise ValueError(f"{current.name} is incompatible with {other.name}.")
                if other.id == "experimental.bloom":
                    decisions.append({"filters": f"{current.id} -> {other.id}", "decision": "allowed_progression_modifier"})
        return decisions


def _definition(
    filter_id: str,
    name: str,
    family: str,
    summary: str,
    creative: str,
    operational: str,
    reads: tuple[D, ...],
    changes: tuple[D, ...],
    inputs: tuple[str, ...],
    artifacts: tuple[str, ...],
    *,
    parameters: tuple[FilterParameter, ...] = (),
    implemented: bool = False,
    experimental: bool = False,
    implementation_key: str | None = None,
    implementation_class: str | None = None,
    execution_mode: str = "unavailable",
    sparse_schedule: bool = False,
    requires_speaker_identity: bool = False,
    aliases: tuple[str, ...] = (),
    preview: bool = False,
    preserves: dict[str, str] | None = None,
    limitations: tuple[str, ...] = (),
    minimum_films: int = 1,
    maximum_films: int | None = 1,
    anchor_behavior: str = "anchor_timeline",
    cinematic_law: str = "Internal Transformation",
    affected_elements: tuple[str, ...] | None = None,
    quality_requirements: tuple[str, ...] = ("Contract invariants pass",),
    deterministic_seed_support: bool = True,
    output_artifacts: tuple[str, ...] = ("rendered_film",),
    affected_artifacts: tuple[str, ...] | None = None,
    intermediate_products: tuple[str, ...] = ("filter_plan",),
    supported_output_forms: tuple[str, ...] | None = None,
) -> FilterDefinition:
    return FilterDefinition(
        id=filter_id,
        name=name,
        family_id=family,
        summary=summary,
        creative_description=creative,
        operational_description=operational,
        reads_dimensions=reads,
        changes_dimensions=changes,
        preserves=preserves or {},
        required_inputs=inputs,
        required_artifacts=artifacts,
        supported_output_forms=(supported_output_forms or ("preview", "best_short", "full_length")) if implemented else (),
        parameters=parameters,
        implemented=implemented,
        experimental=experimental,
        version="1.0.0",
        implementation_key=implementation_key,
        implementation_class=implementation_class or PLANNED_IMPLEMENTATION_CLASSES.get(filter_id, "F"),
        execution_mode=execution_mode,
        sparse_schedule=sparse_schedule,
        requires_speaker_identity=requires_speaker_identity,
        requires_output_acceptance=implemented,
        legacy_aliases=aliases,
        supports_preview=preview,
        supports_stacking=filter_id == "experimental.bloom",
        known_limitations=limitations,
        minimum_films=minimum_films,
        maximum_films=maximum_films,
        anchor_behavior=anchor_behavior,
        cinematic_law=cinematic_law,
        affected_elements=affected_elements or tuple(item.value for item in changes),
        quality_requirements=quality_requirements,
        deterministic_seed_support=deterministic_seed_support,
        output_artifacts=output_artifacts,
        affected_artifacts=affected_artifacts or artifacts,
        intermediate_products=intermediate_products,
    )


def _multiworld_stub(
    key: str,
    name: str,
    law: str,
    summary: str,
    *,
    minimum_films: int = 2,
    maximum_films: int | None = 2,
    affected_elements: tuple[str, ...] = ("dialogue",),
) -> FilterDefinition:
    dimensions = tuple(D(item) for item in affected_elements if item in {row.value for row in D})
    if not dimensions:
        dimensions = (D.DIALOGUE,)
    return _definition(
        f"multiworld.{key}", name, "multiworld", summary, summary,
        "This filter is not yet implemented.", dimensions, dimensions, ("films",),
        ("film_inspections", "shared_timeline", "world_model"),
        minimum_films=minimum_films,
        maximum_films=maximum_films,
        cinematic_law=law,
        affected_elements=affected_elements,
        quality_requirements=("Cinematic-law invariants pass", "Output provenance names every contributing film"),
        output_artifacts=("rendered_film",),
        affected_artifacts=("shared_timeline", "world_model", "replacement_decisions"),
        intermediate_products=("film_inspections", "shared_timeline", "world_model", "replacement_decisions", "filter_plan"),
    )


def _catalog_definition(
    filter_id: str,
    name: str,
    family: str,
    summary: str,
    creative: str,
    operational: str,
    reads: tuple[D, ...],
    changes: tuple[D, ...],
    artifacts: tuple[str, ...],
    *,
    parameters: tuple[FilterParameter, ...] = (INTENSITY,),
    requires_speaker_identity: bool = False,
    experimental: bool = False,
    preserves: dict[str, str] | None = None,
    limitations: tuple[str, ...] = (),
) -> FilterDefinition:
    key = filter_id.split(".", 1)[1]
    return _definition(
        filter_id, name, family, summary, creative, operational,
        reads, changes, ("film",), artifacts,
        parameters=parameters,
        implemented=True,
        experimental=experimental,
        implementation_key=key,
        implementation_class=PLANNED_IMPLEMENTATION_CLASSES[filter_id],
        execution_mode="scheduling_strategy",
        sparse_schedule=True,
        requires_speaker_identity=requires_speaker_identity,
        aliases=(key,),
        preview=True,
        preserves=preserves,
        limitations=limitations,
    )


def _multiworld_dialogue_definition(
    key: str,
    name: str,
    law: str,
    summary: str,
    operational: str,
    *,
    minimum_films: int = 2,
    maximum_films: int | None = 2,
    affected_elements: tuple[str, ...] = ("dialogue",),
    parameters: tuple[FilterParameter, ...] = (INTENSITY,),
    requires_speaker_identity: bool = False,
    preserves: dict[str, str] | None = None,
) -> FilterDefinition:
    dimensions = tuple(D(item) for item in affected_elements if item in {row.value for row in D})
    return _definition(
        f"multiworld.{key}", name, "multiworld", summary, summary, operational,
        dimensions or (D.DIALOGUE,), dimensions or (D.DIALOGUE,), ("films",),
        ("film_inspections", "shared_timeline", "world_model"),
        parameters=parameters,
        implemented=True,
        implementation_key=f"multiworld_{key}",
        implementation_class="F",
        execution_mode="transformation",
        sparse_schedule=True,
        requires_speaker_identity=requires_speaker_identity,
        preview=False,
        preserves=preserves,
        minimum_films=minimum_films,
        maximum_films=maximum_films,
        cinematic_law=law,
        affected_elements=affected_elements,
        quality_requirements=("Cinematic-law invariants pass", "Output provenance names every contributing film"),
        output_artifacts=("rendered_film",),
        affected_artifacts=("shared_timeline", "world_model", "replacement_decisions"),
        intermediate_products=("film_inspections", "shared_timeline", "world_model", "replacement_decisions", "filter_plan"),
        supported_output_forms=("full_length",),
    )


FAMILIES = (
    FilterFamilyDefinition("translation", "Translation", "Move or substitute dialogue while preserving the destination structure.", 0),
    FilterFamilyDefinition("infection", "Infection", "Allow dialogue traits or identities to spread through contact.", 1),
    FilterFamilyDefinition("identity", "Identity", "Manipulate the relationship between performers and recurring voices.", 2),
    FilterFamilyDefinition("memory", "Memory", "Make a film remember, misremember, repeat, or forget its dialogue.", 3),
    FilterFamilyDefinition("emotion", "Emotion", "Redirect dialogue according to emotional features.", 4),
    FilterFamilyDefinition("time", "Time", "Reorganize dialogue through narrative and chronological relationships.", 5),
    FilterFamilyDefinition("experimental", "Experimental", "Strongly authored procedures combining relationship dimensions.", 6),
    FilterFamilyDefinition("multiworld", "Multiworld", "Apply one cinematic law across two or more complete films.", 7),
)


DEFINITIONS = (
    _definition(
        "translation.self_shuffle", "Self Shuffle", "translation",
        "One film speaks its own lines in different moments.",
        "The film exchanges memories with itself.",
        "Dialogue is reassigned to different speaking windows while original-line placement is prohibited.",
        (D.DIALOGUE, D.PERFORMANCE, D.IDENTITY, D.TIME), (D.DIALOGUE, D.TIME), ("film",),
        ("dialogue_events", "performances", "speakers", "scenes"),
        parameters=(FilterParameter("seed", "Seed", "integer", 1, "Makes the shuffle reproducible.", minimum=0, advanced=True), ALLOW_REUSE, PERFORMANCE),
        implemented=True, implementation_key="self_shuffle", implementation_class="A", execution_mode="transformation", aliases=("self_shuffle", "mutation_self_shuffle", "Self-Shuffle"), preview=True,
        preserves={"performance": "Preserved where possible", "identity": "Stable where speaker material permits"},
    ),
    _definition(
        "translation.echo", "Echo", "translation",
        "Earlier dialogue returns after a fixed delay.",
        "The film hears itself again.",
        "Selected lines repeat later over the original film at a configurable delay.",
        (D.DIALOGUE, D.TIME), (D.DIALOGUE, D.TIME), ("film",), ("dialogue_events",),
        parameters=(
            FilterParameter("delay_seconds", "Delay", "float", 7.0, "Seconds before a selected line returns.", minimum=0.1, maximum=600.0),
            FilterParameter("repeat_frequency", "Repeat frequency", "integer", 1, "Use every nth eligible line.", minimum=1, advanced=True),
            FilterParameter("max_repeats", "Maximum repeats", "integer", 90, "Maximum echoed lines.", minimum=0, advanced=True),
            FilterParameter("duck_original_at_echoes", "Duck original speech", "boolean", True, "Make the returning line audible over the source soundtrack."),
        ),
        implemented=True, implementation_key="echo", implementation_class="A", execution_mode="transformation", aliases=("echo",), preview=True,
        preserves={"identity": "Preserved", "performance": "Preserved", "time": "Delayed copy"},
    ),
    _definition(
        "multiworld.movie_masher", "Movie Masher", "multiworld",
        "An anchor film speaks with dialogue from another film.",
        "One film becomes the voice haunting another.",
        "Source performances fill destination speaking windows using duration, rhythm, scene, and speaker compatibility.",
        (D.DIALOGUE, D.PERFORMANCE, D.IDENTITY, D.TIME), (D.DIALOGUE, D.IDENTITY, D.PERFORMANCE),
        ("films",),
        ("dialogue_events", "performances", "speakers", "scenes", "shots"),
        parameters=(INTENSITY, PERFORMANCE), implemented=True, implementation_key="movie_masher", implementation_class="A", execution_mode="transformation", aliases=("movie_masher", "translation.movie_masher", "Transposition"), preview=True,
        preserves={"time": "Anchor chronology preserved", "performance": "Matched where possible"},
        minimum_films=2, maximum_films=2, cinematic_law="Dialogue Translation",
        affected_elements=("dialogue", "identity", "performance"),
        quality_requirements=("Dialogue coverage passes", "Silence ratio passes", "Audio provenance passes", "Contract invariants pass"),
        output_artifacts=("replacement_dialogue", "rendered_film"),
        affected_artifacts=("dialogue_events", "performances", "speakers", "scenes", "shots", "shared_timeline"),
        intermediate_products=("film_inspections", "shared_timeline", "world_model", "replacement_decisions", "filter_plan"),
    ),
    _definition(
        "translation.drift", "Drift", "translation",
        "Dialogue slides increasingly away from its original timing.",
        "The voice loses pace with the body.",
        "Each line receives a progressively larger positive time offset across the film.",
        (D.DIALOGUE, D.TIME), (D.TIME,), ("film",), ("dialogue_events",),
        parameters=(
            FilterParameter("starting_offset", "Starting offset", "float", 1.0, "Initial dialogue delay in seconds.", minimum=0.0, maximum=600.0),
            FilterParameter("maximum_offset", "Maximum offset", "float", 18.0, "Late-film dialogue delay in seconds.", minimum=0.0, maximum=600.0),
            FilterParameter("preserve_original_soundtrack", "Preserve soundtrack", "boolean", True, "Keep non-dialogue source audio beneath the drift."),
        ),
        implemented=True, implementation_key="drift", implementation_class="A", execution_mode="transformation", aliases=("drift",), preview=True,
        preserves={"dialogue": "Original lines", "identity": "Original voices", "performance": "Original delivery"},
    ),
    _definition(
        "identity.possession", "Possession", "identity",
        "One recurring voice takes residence inside another speaker.",
        "One voice takes residence inside another performer.",
        "Cinelingus consistently maps lines from a possessing speaker onto windows belonging to a possessed speaker.",
        (D.DIALOGUE, D.PERFORMANCE, D.IDENTITY, D.TIME), (D.DIALOGUE, D.IDENTITY), ("film",),
        ("dialogue_events", "performances", "speakers", "scenes"),
        parameters=(
            FilterParameter("possessing_speaker", "Possessing speaker", "speaker", "auto", "Speaker whose dialogue identity spreads."),
            FilterParameter("possessed_speaker", "Possessed speaker", "speaker", "auto", "Speaker whose speaking windows are replaced."),
            INTENSITY,
            FilterParameter("identity_stability", "Identity stability", "choice", "Strict", "How consistently the chosen identity mapping is maintained.", ("Strict", "Mostly stable", "Fluid", "Chaotic")),
            FilterParameter("minimum_temporal_separation", "Minimum temporal separation", "float", 20.0, "Minimum seconds between source and destination moments.", minimum=0.0, advanced=True),
            PERFORMANCE, ALLOW_REUSE,
            FilterParameter("replace_scope", "Replacement scope", "choice", "All appearances", "Replace every eligible appearance or a representative subset.", ("All appearances", "Selected scenes")),
        ),
        implemented=True, implementation_key="possession", implementation_class="A", execution_mode="scheduling_strategy", sparse_schedule=True, requires_speaker_identity=True, aliases=("possession",), preview=True,
        preserves={"time": "Destination chronology preserved", "performance": "Duration matched where possible"},
        limitations=("Visible performer identity is approximated through diarized speaker windows.",),
    ),
    _definition(
        "time.foreshadow", "Foreshadow", "time",
        "Earlier scenes speak with dialogue from later scenes.",
        "The film knows what it has not yet lived.",
        "Every replacement source begins later than its destination by the configured minimum distance.",
        (D.DIALOGUE, D.PERFORMANCE, D.TIME), (D.DIALOGUE, D.TIME), ("film",),
        ("dialogue_events", "performances", "scenes"),
        parameters=(
            FilterParameter("minimum_future_distance", "Minimum future distance", "float", 30.0, "Minimum seconds the source line must lie in the future.", minimum=0.0),
            FilterParameter("maximum_future_distance", "Maximum future distance", "float", 900.0, "Maximum future reach in seconds.", minimum=0.1, advanced=True),
            INTENSITY, PERFORMANCE, SEMANTIC, ALLOW_REUSE,
            FilterParameter("final_act_policy", "Final-act policy", "choice", "Gradually reduce", "How to handle windows with too little future dialogue.", ("Gradually reduce", "Stop at cutoff", "Explicit wraparound")),
            PROGRESSION,
        ),
        implemented=True, implementation_key="foreshadow", implementation_class="A", execution_mode="scheduling_strategy", sparse_schedule=True, aliases=("foreshadow",), preview=True,
        preserves={"identity": "Preserved where possible", "performance": "Duration matched", "time": "Destination chronology preserved"},
    ),
    _definition(
        "infection.contagion", "Contagion", "infection",
        "A dialogue identity spreads through measured speaker contact.",
        "A voice passes from speaker to speaker through conversation.",
        "A deterministic contact graph controls exposure, infection time, and progressive source-identity replacement.",
        (D.DIALOGUE, D.PERFORMANCE, D.IDENTITY, D.TIME), (D.DIALOGUE, D.IDENTITY, D.TIME), ("film",),
        ("dialogue_events", "performances", "speakers", "scenes", "speaker_graph"),
        parameters=(
            FilterParameter("initial_carrier", "Initial carrier", "speaker", "auto", "Speaker from whom infection begins."),
            FilterParameter("spread_speed", "Spread speed", "choice", "Moderate", "How quickly valid contact produces infection.", ("Slow", "Moderate", "Fast")),
            FilterParameter("contact_threshold", "Contact threshold", "float", 1.0, "Minimum measured contact needed for exposure.", minimum=0.1, advanced=True),
            FilterParameter("maximum_infected_speakers", "Maximum infected speakers", "integer", 4, "Limits the spread.", minimum=1),
            INTENSITY,
            FilterParameter("recovery_allowed", "Allow recovery", "boolean", False, "Permit identity to return after infection.", advanced=True),
            PROGRESSION,
            FilterParameter("source_pool_policy", "Source dialogue pool", "choice", "Initial carrier", "Use only the carrier or all currently infected speakers.", ("Initial carrier", "Combined infected pool")),
        ),
        implemented=True, implementation_key="contagion", implementation_class="A", execution_mode="scheduling_strategy", sparse_schedule=True, requires_speaker_identity=True, aliases=("contagion",), preview=True,
        preserves={"performance": "Duration matched where possible", "time": "Exposure precedes infection"},
    ),
    _definition(
        "experimental.bloom", "Bloom", "experimental",
        "Transformation begins almost invisibly and grows progressively stranger.",
        "The film opens slowly into its transformed state.",
        "A nonlinear progression curve increases replacement frequency and loosens time, identity, semantic, and performance constraints.",
        (D.DIALOGUE, D.PERFORMANCE, D.IDENTITY, D.TIME), (D.DIALOGUE, D.PERFORMANCE, D.IDENTITY, D.TIME), ("film",),
        ("dialogue_events", "performances", "speakers", "scenes"),
        parameters=(
            FilterParameter("starting_intensity", "Starting intensity", "float", 0.05, "Transformation strength at the beginning.", minimum=0.0, maximum=1.0),
            FilterParameter("ending_intensity", "Ending intensity", "float", 0.95, "Transformation strength near the end.", minimum=0.0, maximum=1.0),
            FilterParameter("curve_shape", "Curve shape", "choice", "Gentle nonlinear", "How the transformation grows.", ("Gentle nonlinear", "Linear", "Late surge", "Early surge")),
            FilterParameter("bloom_dimensions", "Dimensions allowed to bloom", "choice", "Dialogue + Time + mild Identity", "Relationships allowed to loosen progressively.", ("Dialogue + Time", "Dialogue + Time + mild Identity", "All four dimensions")),
            FilterParameter("maximum_identity_instability", "Maximum identity instability", "float", 0.4, "Late-output tolerance for voice mismatch.", minimum=0.0, maximum=1.0, advanced=True),
            FilterParameter("maximum_temporal_distance", "Maximum temporal distance", "float", 1200.0, "Maximum source/destination separation.", minimum=0.0, advanced=True),
            FilterParameter("preserve_ending_coherence", "Preserve ending coherence", "boolean", True, "Avoid total randomness in the final moments."),
        ),
        implemented=True, experimental=True, implementation_key="bloom", implementation_class="A", execution_mode="scheduling_strategy", sparse_schedule=True, aliases=("bloom",), preview=True,
        preserves={"progression": "Measurably increasing", "ending": "Authored coherence when enabled"},
    ),
    _catalog_definition(
        "infection.whisper", "Whisper", "infection", "A stable carrier voice spreads through quiet, band-limited replacements.",
        "A voice enters at the edge of hearing.",
        "Dialogue from one recurring carrier fills selected windows with explicit gain, high-pass, and low-pass treatment.",
        (D.DIALOGUE, D.IDENTITY, D.PERFORMANCE), (D.DIALOGUE, D.PERFORMANCE), ("dialogue_events", "speakers"),
        parameters=(FilterParameter("carrier_speaker", "Carrier speaker", "speaker", "auto", "Recurring voice whose quiet trait spreads."), INTENSITY),
        requires_speaker_identity=True,
        preserves={"time": "Destination chronology preserved", "identity": "One stable carrier"},
    ),
    _catalog_definition(
        "infection.mutation", "Mutation", "infection", "Dialogue replacement mismatch grows measurably through the output.",
        "Contamination alters what it carries.",
        "Duration and speaker-identity mismatch are combined into a bounded mutation magnitude that never decreases.",
        (D.DIALOGUE, D.IDENTITY, D.PERFORMANCE, D.TIME), (D.DIALOGUE, D.IDENTITY, D.PERFORMANCE), ("dialogue_events", "speakers"),
        preserves={"time": "Destination chronology preserved", "progression": "Mutation magnitude is monotonic"},
    ),
    _catalog_definition(
        "infection.dialect", "Dialect", "infection", "One recurring carrier's cadence spreads through selected dialogue windows.",
        "The cast acquires a common tongue.",
        "Carrier lines are duration-fitted to destination windows with bounded time stretching and a stable source identity.",
        (D.PERFORMANCE, D.IDENTITY, D.DIALOGUE), (D.PERFORMANCE, D.DIALOGUE), ("performances", "speakers"),
        parameters=(FilterParameter("carrier_speaker", "Carrier speaker", "speaker", "auto", "Recurring voice whose cadence spreads."), INTENSITY),
        requires_speaker_identity=True,
        preserves={"time": "Destination chronology preserved", "identity": "One stable carrier"},
    ),
    _definition(
        "identity.doppelganger", "Doppelgänger", "identity",
        "Two recurring speakers exchange dialogue identities through one stable mirrored pair.",
        "Each of two bodies becomes the other's recurring double.",
        "Dialogue from either selected speaker fills eligible windows belonging to the other; the pair never changes during a run.",
        (D.IDENTITY, D.DIALOGUE, D.PERFORMANCE, D.TIME), (D.IDENTITY, D.DIALOGUE), ("film",),
        ("dialogue_events", "performances", "speakers", "scenes"),
        parameters=(
            FilterParameter("primary_speaker", "Primary speaker", "speaker", "auto", "First member of the mirrored pair."),
            FilterParameter("mirror_speaker", "Mirror speaker", "speaker", "auto", "Second member of the mirrored pair."),
            INTENSITY, PERFORMANCE, ALLOW_REUSE,
        ),
        implemented=True, implementation_key="doppelganger", implementation_class="A", execution_mode="scheduling_strategy",
        sparse_schedule=True, requires_speaker_identity=True, aliases=("doppelganger", "doppelgänger"), preview=True,
        preserves={"time": "Destination chronology preserved", "identity": "One stable bidirectional pair"},
    ),
    _definition(
        "identity.chorus", "Chorus", "identity",
        "Several recurring speakers converge on one stable dialogue identity.",
        "Many bodies speak with one recurring voice.",
        "Dialogue from one anchor speaker fills eligible windows belonging to a bounded set of other speakers.",
        (D.IDENTITY, D.DIALOGUE, D.PERFORMANCE, D.TIME), (D.IDENTITY, D.DIALOGUE), ("film",),
        ("dialogue_events", "performances", "speakers", "scenes"),
        parameters=(
            FilterParameter("anchor_speaker", "Anchor speaker", "speaker", "auto", "Speaker whose dialogue identity the chorus adopts."),
            FilterParameter("maximum_chorus_speakers", "Maximum chorus speakers", "integer", 4, "Maximum non-anchor speakers transformed.", minimum=1),
            INTENSITY, PERFORMANCE, ALLOW_REUSE,
        ),
        implemented=True, implementation_key="chorus", implementation_class="A", execution_mode="scheduling_strategy",
        sparse_schedule=True, requires_speaker_identity=True, aliases=("chorus",), preview=True,
        preserves={"time": "Destination chronology preserved", "identity": "One stable anchor identity"},
    ),
    _catalog_definition(
        "identity.split_personality", "Split Personality", "identity", "One recurring speaker is partitioned across two or more stable donor identities.",
        "A voice becomes multiple occupants.",
        "Eligible windows for one anchor speaker receive donor voices in a stable round-robin partition.",
        (D.IDENTITY, D.DIALOGUE, D.PERFORMANCE), (D.IDENTITY, D.DIALOGUE), ("speakers", "dialogue_events"),
        parameters=(FilterParameter("anchor_speaker", "Anchor speaker", "speaker", "auto", "Speaker whose dialogue identity divides."), FilterParameter("personality_count", "Personality count", "integer", 2, "Number of stable donor identities.", minimum=2), INTENSITY),
        requires_speaker_identity=True,
        preserves={"time": "Destination chronology preserved", "partition": "Donor identity is stable per partition"},
    ),
    _catalog_definition(
        "memory.dream", "Dream", "memory", "Dialogue returns through a disclosed lexical-association and temporal-drift proxy.",
        "The film dreams its own speech.",
        "Token overlap plus normalized temporal distance ranks non-identical source lines; no semantic embedding is claimed.",
        (D.DIALOGUE, D.TIME), (D.DIALOGUE, D.TIME), ("dialogue_events",),
        preserves={"time": "Destination chronology preserved", "provenance": "Association proxy is emitted per mapping"},
        limitations=("Associative meaning is approximated by tokens and temporal distance, not semantic embeddings.",),
    ),
    _catalog_definition(
        "memory.recollection", "Recollection", "memory", "Past dialogue resurfaces only after a measurable minimum distance.",
        "The film remembers aloud.",
        "Every source line precedes its destination, preferring the same speaker when a viable earlier line exists.",
        (D.DIALOGUE, D.TIME, D.IDENTITY), (D.DIALOGUE, D.TIME), ("dialogue_events",),
        parameters=(FilterParameter("minimum_past_distance", "Minimum past distance", "float", 15.0, "Seconds by which a memory must precede its destination.", minimum=0.0), INTENSITY),
        preserves={"time": "Destination chronology preserved", "identity": "Preserved where earlier material exists"},
    ),
    _catalog_definition(
        "memory.amnesia", "Amnesia", "memory", "The eligible dialogue memory pool shrinks irreversibly through the output.",
        "The film forgets how it spoke.",
        "Each successive selected window draws from a prefix of the source pool that never grows.",
        (D.DIALOGUE, D.IDENTITY, D.TIME), (D.DIALOGUE, D.IDENTITY), ("dialogue_events",),
        preserves={"time": "Destination chronology preserved", "progression": "Forgotten sources never return"},
    ),
    *tuple(_catalog_definition(
        f"emotion.{key}", name, "emotion",
        f"Dialogue is redirected using a disclosed deterministic {key} lexical proxy.",
        f"The film speaks through {key}.",
        f"Transcript tokens are scored against the versioned {key} term list; no emotional embedding is claimed.",
        (D.DIALOGUE, D.PERFORMANCE), (D.DIALOGUE,), ("dialogue_events", "performances"),
        limitations=(f"{name} is a lexical proxy, not an inferred emotional state.",),
        preserves={"time": "Destination chronology preserved", "provenance": "Proxy name, terms, and score are emitted"},
    ) for key, name in (("wonder", "Wonder"), ("regret", "Regret"), ("optimist", "Optimist"), ("paranoia", "Paranoia"))),
    _catalog_definition(
        "emotion.exhaustion", "Exhaustion", "emotion", "Slow-delivery dialogue is progressively stretched, darkened, and attenuated.",
        "The film speaks through exhaustion.",
        "Seconds per word ranks delivery; bounded time stretch, low-pass filtering, and gain reduction grow through selected windows.",
        (D.DIALOGUE, D.PERFORMANCE), (D.DIALOGUE, D.PERFORMANCE), ("dialogue_events", "performances"),
        preserves={"time": "Destination chronology preserved", "provenance": "Performance proxy and audio controls are emitted"},
        limitations=("Exhaustion is a seconds-per-word performance proxy, not an inferred emotional state.",),
    ),
    _definition(
        "time.flashback", "Flashback", "time",
        "Later scenes speak only with dialogue from sufficiently earlier moments.",
        "The past interrupts the present.",
        "Every replacement source precedes its destination by the configured minimum temporal distance.",
        (D.DIALOGUE, D.TIME, D.PERFORMANCE), (D.DIALOGUE, D.TIME), ("film",), ("dialogue_events", "scenes"),
        parameters=(
            FilterParameter("minimum_past_distance", "Minimum past distance", "float", 30.0, "Minimum seconds the source line must lie in the past.", minimum=0.0),
            FilterParameter("maximum_past_distance", "Maximum past distance", "float", 900.0, "Maximum reach into the past.", minimum=0.1, advanced=True),
            INTENSITY, PERFORMANCE, ALLOW_REUSE, PROGRESSION,
        ),
        implemented=True, implementation_key="flashback", implementation_class="A", execution_mode="scheduling_strategy",
        sparse_schedule=True, aliases=("flashback",), preview=True,
        preserves={"identity": "Preserved where possible", "performance": "Duration matched", "time": "Destination chronology preserved"},
    ),
    _definition(
        "time.spiral", "Spiral", "time",
        "Dialogue revisits moments at measurably increasing temporal distance.",
        "The film circles while moving forward.",
        "Selected replacements alternate around the present while absolute source distance grows monotonically.",
        (D.DIALOGUE, D.TIME, D.PERFORMANCE), (D.DIALOGUE, D.TIME), ("film",), ("dialogue_events", "scenes"),
        parameters=(
            FilterParameter("starting_distance", "Starting distance", "float", 10.0, "Initial target temporal displacement.", minimum=0.1),
            FilterParameter("maximum_distance", "Maximum distance", "float", 600.0, "Final target temporal displacement.", minimum=0.2),
            FilterParameter("direction", "Direction", "choice", "Alternating", "Whether the spiral reaches into past, future, or both.", ("Alternating", "Past only", "Future only")),
            INTENSITY, PERFORMANCE, ALLOW_REUSE,
        ),
        implemented=True, implementation_key="spiral", implementation_class="A", execution_mode="scheduling_strategy",
        sparse_schedule=True, aliases=("spiral",), preview=True,
        preserves={"identity": "Preserved where possible", "performance": "Duration matched", "progression": "Absolute displacement never decreases"},
    ),
    _catalog_definition(
        "time.mobius", "Möbius", "time", "Beginning and ending dialogue occupy paired positions on opposite sides of one fold.",
        "The film discovers its other side.",
        "Each selected destination is paired with source dialogue nearest its normalized opposite position.",
        (D.DIALOGUE, D.TIME), (D.DIALOGUE, D.TIME), ("dialogue_events", "scenes"),
        preserves={"time": "Destination chronology preserved", "fold": "Normalized source and destination positions sum to approximately one"},
    ),
    _catalog_definition(
        "experimental.venom", "Venom", "experimental", "Lexical hostility grows monotonically through the output.",
        "Meaning turns against its scene.",
        "A disclosed hostility term score orders source dialogue from lower to higher pressure.",
        (D.DIALOGUE, D.PERFORMANCE, D.TIME), (D.DIALOGUE, D.PERFORMANCE), ("dialogue_events", "performances", "speakers"),
        experimental=True,
        preserves={"time": "Destination chronology preserved", "progression": "Hostility proxy never decreases"},
        limitations=("Hostility is a lexical proxy, not a semantic or emotional embedding.",),
    ),
    _catalog_definition(
        "experimental.shed_skin", "Shed Skin", "experimental", "Dialogue identity changes in ordered temporal stages and never reverts.",
        "The film leaves voices behind.",
        "The output is divided into stages, each assigned one stable source speaker identity.",
        (D.DIALOGUE, D.PERFORMANCE, D.IDENTITY, D.TIME), (D.IDENTITY, D.TIME), ("dialogue_events", "performances", "speakers"),
        parameters=(FilterParameter("identity_stages", "Identity stages", "integer", 3, "Maximum ordered identity stages.", minimum=2), INTENSITY),
        requires_speaker_identity=True,
        experimental=True,
        preserves={"progression": "Identity stage never decreases", "stage_identity": "One stable source identity per stage"},
    ),
    _catalog_definition(
        "experimental.ouroboros", "Ouroboros", "experimental", "Dialogue follows one closed ring whose ending material feeds the opening.",
        "The film consumes its own conclusion.",
        "A stable circular source offset is applied to every selected destination window.",
        (D.DIALOGUE, D.PERFORMANCE, D.IDENTITY, D.TIME), (D.DIALOGUE, D.TIME), ("dialogue_events", "performances", "speakers"),
        experimental=True,
        preserves={"time": "Destination chronology preserved", "ring": "Circular offset is stable"},
    ),
    _multiworld_dialogue_definition(
        "possession", "Possession", "Identity Exchange",
        "One film's stable donor identity takes residence inside one recurring anchor-film speaker.",
        "Select one recurring speaker per film and place only the donor identity into eligible windows belonging to the stable anchor host.",
        affected_elements=("dialogue", "identity"),
        requires_speaker_identity=True,
        preserves={"time": "Anchor chronology is preserved", "identity": "One stable cross-film pair"},
    ),
    _multiworld_dialogue_definition(
        "contagion", "Contagion", "Narrative Infection",
        "Donor films infect the anchor timeline in ordered, non-reverting phases.",
        "Partition selected anchor windows into deterministic donor-film phases; every donor contributes and phase order never reverses.",
        maximum_films=None,
        affected_elements=("dialogue", "identity", "scene_order"),
        preserves={"time": "Anchor chronology is preserved", "progression": "Donor phases never revert"},
    ),
    _multiworld_stub("doppelganger", "Doppelgänger", "Identity Exchange", "Two films discover recurring doubles in one another.", affected_elements=("dialogue", "identity", "shot_selection")),
    _multiworld_stub("mirror_world", "Mirror World", "Reality Collision", "An anchor film is reflected through a second cinematic reality.", affected_elements=("scene_order", "identity", "shot_selection")),
    _multiworld_dialogue_definition(
        "prophecy", "Prophecy", "Temporal Exchange",
        "Later normalized dialogue from one film predicts earlier positions in the anchor film.",
        "A donor line is eligible only when its normalized source position leads the anchor destination by the configured minimum.",
        affected_elements=("dialogue", "time", "scene_order"),
        parameters=(FilterParameter("minimum_normalized_lead", "Minimum prophecy lead", "float", 0.15, "Minimum normalized timeline lead between prophecy and destination.", minimum=0.01, maximum=0.8), INTENSITY),
        preserves={"time": "Anchor chronology is preserved", "provenance": "Normalized positions and lead are emitted"},
    ),
    _multiworld_dialogue_definition(
        "echo_chamber", "Echo Chamber", "Dialogue Translation",
        "Every selected anchor window becomes a staggered, attenuated echo group containing every film.",
        "Add one provenance-bearing dialogue layer per film with nondecreasing delay and nonincreasing gain.",
        maximum_films=None,
        affected_elements=("dialogue", "soundscape"),
        parameters=(FilterParameter("layer_delay_seconds", "Layer delay", "float", 0.18, "Delay added between film layers.", minimum=0.04, maximum=2.0), FilterParameter("layer_attenuation_db", "Layer attenuation", "float", -4.0, "Gain change applied to each successive layer.", minimum=-24.0, maximum=-0.1), INTENSITY),
        preserves={"time": "Anchor chronology is preserved", "provenance": "Every echo layer names its source film"},
    ),
    _multiworld_stub("bleed", "Bleed", "Reality Collision", "Images, sound, and narrative details leak between worlds.", affected_elements=("dialogue", "music", "soundscape", "shot_selection")),
    _multiworld_stub("parallel_universes", "Parallel Universes", "Temporal Exchange", "Equivalent moments from multiple films coexist on a shared timeline.", maximum_films=None, affected_elements=("time", "scene_order", "shot_selection")),
    _multiworld_stub("wormhole", "Wormhole", "Temporal Exchange", "Selected moments cross between films through deterministic temporal portals.", maximum_films=None, affected_elements=("time", "scene_order", "shot_selection")),
    _multiworld_stub("chimera", "Chimera", "Genre Mutation", "Three films combine into one hybrid cinematic organism.", minimum_films=3, maximum_films=3, affected_elements=("dialogue", "identity", "music", "soundscape", "genre")),
    _multiworld_stub("triangle", "Triangle", "Narrative Infection", "Three films exchange pressure through a closed narrative relationship.", minimum_films=3, maximum_films=3, affected_elements=("dialogue", "identity", "scene_order")),
    _multiworld_stub("civilization", "Civilization", "Reality Collision", "Five or more films form a persistent shared cinematic society.", minimum_films=5, maximum_films=None, affected_elements=("dialogue", "identity", "scene_order", "music", "soundscape", "shot_selection", "genre")),
)


@lru_cache(maxsize=1)
def default_filter_registry() -> FilterRegistry:
    return FilterRegistry(FAMILIES, DEFINITIONS)
