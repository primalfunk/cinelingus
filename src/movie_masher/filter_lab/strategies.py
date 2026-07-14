from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Callable

from movie_masher import __version__
from movie_masher.util import utc_now


INTENSITY_RATIOS = {"Trace": 0.15, "Light": 0.3, "Moderate": 0.55, "Strong": 0.8, "Total": 1.0}


@dataclass(frozen=True)
class StrategySpec:
    builder: Callable[..., dict[str, Any]]
    progress_stages: tuple[str, str, str, str]


STRATEGY_SPECS: dict[str, StrategySpec] = {}


def scheduling_strategy(filter_id: str, progress_stages: tuple[str, str, str, str]):
    def register(builder: Callable[..., dict[str, Any]]):
        if filter_id in STRATEGY_SPECS:
            raise ValueError(f"Duplicate scheduling strategy for '{filter_id}'.")
        STRATEGY_SPECS[filter_id] = StrategySpec(builder=builder, progress_stages=progress_stages)
        return builder
    return register


def get_strategy_spec(filter_id: str) -> StrategySpec:
    try:
        return STRATEGY_SPECS[filter_id]
    except KeyError as exc:
        raise ValueError(f"No Filter Laboratory strategy is registered for '{filter_id}'.") from exc


def has_strategy(filter_id: str) -> bool:
    return filter_id in STRATEGY_SPECS


@scheduling_strategy("possession", ("identifying viable speakers", "constructing speaker dialogue pools", "mapping possessing speaker to possessed speaker", "validating identity consistency"))
def build_possession_schedule(
    *, clips: list[dict[str, Any]], windows: list[dict[str, Any]], duration: float,
    parameters: dict[str, Any], seed: int = 1,
) -> dict[str, Any]:
    rng = random.Random(seed)
    clip_pools = _speaker_pools(clips)
    window_pools = _speaker_pools(windows)
    viable_sources = {key: rows for key, rows in clip_pools.items() if len(rows) >= 2}
    viable_destinations = {key: rows for key, rows in window_pools.items() if len(rows) >= 1}
    if len(set(viable_sources) | set(viable_destinations)) < 2:
        raise ValueError("Possession requires at least two viable diarized speakers.")
    possessing = str(parameters.get("possessing_speaker", "auto"))
    possessed = str(parameters.get("possessed_speaker", "auto"))
    if possessing == "auto":
        possessing = max(viable_sources, key=lambda key: (_pool_duration(viable_sources[key]), len(viable_sources[key]), key))
    if possessing not in viable_sources:
        raise ValueError(f"Possessing speaker '{possessing}' has insufficient dialogue.")
    if possessed == "auto":
        choices = {key: rows for key, rows in viable_destinations.items() if key != possessing}
        if not choices:
            raise ValueError("Possession could not find a distinct possessed speaker.")
        possessed = max(choices, key=lambda key: (_pool_duration(choices[key]), len(choices[key]), key))
    if possessed == possessing:
        raise ValueError("Possessing and possessed speakers must be different.")
    if possessed not in viable_destinations:
        raise ValueError(f"Possessed speaker '{possessed}' has no viable destination windows.")

    separation = float(parameters.get("minimum_temporal_separation", 20.0))
    ratio = INTENSITY_RATIOS.get(str(parameters.get("intensity", "Moderate")), 0.55)
    allow_reuse = bool(parameters.get("allow_line_reuse", False))
    source_pool = list(viable_sources[possessing])
    destinations = sorted(viable_destinations[possessed], key=_start)
    target_count = max(1, round(len(destinations) * ratio))
    selected_windows = _evenly_select(destinations, target_count)
    used: set[str] = set()
    mappings: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    duration_fits: list[float] = []
    for window in selected_windows:
        candidates = [
            clip for clip in source_pool
            if abs(_start(clip) - _start(window)) >= separation and (allow_reuse or _id(clip) not in used)
        ]
        if not candidates:
            rejected.append(_rejection(window, "no_possessing_line_with_temporal_separation"))
            continue
        rng.shuffle(candidates)
        candidates.sort(key=lambda clip: (_duration_fit(_duration(clip), _duration(window)), _duration(clip)), reverse=True)
        chosen = candidates[0]
        used.add(_id(chosen))
        fit = _duration_fit(_duration(chosen), _duration(window))
        duration_fits.append(fit)
        mapping = _mapping(chosen, window, "possession", "stable_speaker_identity_reassignment")
        mapping.update({
            "possessing_speaker": possessing, "possessed_speaker": possessed,
            "identity_consistent": True, "duration_fit": round(fit, 4),
        })
        mappings.append(mapping)
    if not mappings:
        raise ValueError("Possession found no viable speaker-specific replacements.")
    reuse_count = len(mappings) - len({_id_from_mapping(item) for item in mappings})
    metrics = {
        "possessing_speaker": possessing, "possessed_speaker": possessed,
        "replaced_windows": len(mappings), "eligible_windows": len(destinations),
        "source_line_reuse": reuse_count, "identity_consistency_rate": 1.0,
        "mean_duration_fit": round(sum(duration_fits) / len(duration_fits), 4),
        "auto_selection": {
            "possessing": parameters.get("possessing_speaker", "auto") == "auto",
            "possessed": parameters.get("possessed_speaker", "auto") == "auto",
            "reason": "largest viable source pool and largest distinct destination presence",
        },
    }
    summary = (
        f"Possession replaced {len(mappings)} dialogue windows belonging to {possessed} with lines spoken by "
        f"{possessing}. Identity mapping remained stable; {reuse_count} source lines were reused."
    )
    return _schedule("possession", duration, mappings, rejected, metrics, {
        "passed": True, "source_and_destination_speakers_distinct": possessing != possessed,
        "all_source_lines_match_possessing_speaker": all(item.get("source_speaker_id") == possessing for item in mappings),
        "all_destination_windows_match_possessed_speaker": all(item.get("destination_speaker_id") == possessed for item in mappings),
        "source_identity_is_stable": all(item.get("source_speaker_id") == possessing for item in mappings),
        "destination_identity_is_stable": all(item.get("destination_speaker_id") == possessed for item in mappings),
    }, summary)


@scheduling_strategy("doppelganger", ("identifying the mirrored pair", "building reciprocal dialogue pools", "exchanging speaker identities", "validating pair stability"))
def build_doppelganger_schedule(
    *, clips: list[dict[str, Any]], windows: list[dict[str, Any]], duration: float,
    parameters: dict[str, Any], seed: int = 1,
) -> dict[str, Any]:
    rng = random.Random(seed)
    clip_pools = _speaker_pools(clips)
    window_pools = _speaker_pools(windows)
    viable = {
        speaker for speaker in set(clip_pools) & set(window_pools)
        if clip_pools[speaker] and window_pools[speaker]
    }
    if len(viable) < 2:
        raise ValueError("Doppelgänger requires at least two viable diarized speakers with dialogue and destination windows.")
    ranked = sorted(viable, key=lambda speaker: (_pool_duration(clip_pools[speaker]) + _pool_duration(window_pools[speaker]), speaker), reverse=True)
    primary = str(parameters.get("primary_speaker", "auto"))
    mirror = str(parameters.get("mirror_speaker", "auto"))
    if primary == "auto":
        primary = ranked[0]
    if primary not in viable:
        raise ValueError(f"Primary speaker '{primary}' is not viable.")
    if mirror == "auto":
        mirror = next((speaker for speaker in ranked if speaker != primary), "")
    if mirror not in viable or mirror == primary:
        raise ValueError("Doppelgänger requires two distinct viable speakers.")
    ratio = INTENSITY_RATIOS.get(str(parameters.get("intensity", "Moderate")), 0.55)
    allow_reuse = bool(parameters.get("allow_line_reuse", False))
    pair = (primary, mirror)
    destinations = sorted(window_pools[primary] + window_pools[mirror], key=_start)
    selected = _evenly_select(destinations, max(1, round(len(destinations) * ratio)))
    used: set[str] = set()
    mappings: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for window in selected:
        destination_speaker = _speaker(window)
        source_speaker = mirror if destination_speaker == primary else primary
        candidates = [row for row in clip_pools[source_speaker] if allow_reuse or _id(row) not in used]
        if not candidates:
            rejected.append(_rejection(window, "mirrored_speaker_pool_exhausted"))
            continue
        rng.shuffle(candidates)
        candidates.sort(key=lambda row: _duration_fit(_duration(row), _duration(window)), reverse=True)
        chosen = candidates[0]
        used.add(_id(chosen))
        mapping = _mapping(chosen, window, "doppelganger", "stable_bidirectional_identity_mirror")
        mapping.update({
            "primary_speaker": primary, "mirror_speaker": mirror,
            "mirrored_direction": f"{source_speaker}->{destination_speaker}",
            "identity_pair_stable": True,
        })
        mappings.append(mapping)
    if not mappings:
        raise ValueError("Doppelgänger found no viable reciprocal replacements.")
    valid = all(
        {row.get("source_speaker_id"), row.get("destination_speaker_id")} == set(pair)
        and row.get("source_speaker_id") != row.get("destination_speaker_id")
        for row in mappings
    )
    if not valid:
        raise ValueError("Doppelgänger validation failed: a mapping escaped the selected mirrored pair.")
    metrics = {
        "primary_speaker": primary, "mirror_speaker": mirror, "transformed_windows": len(mappings),
        "eligible_windows": len(destinations), "source_line_reuse": len(mappings) - len({_id_from_mapping(row) for row in mappings}),
    }
    return _schedule("doppelganger", duration, mappings, rejected, metrics, {
        "passed": True, "pair_is_distinct": primary != mirror, "source_and_destination_are_distinct": primary != mirror,
        "all_mappings_remain_inside_pair": valid,
    }, f"Doppelgänger exchanged dialogue identities between {primary} and {mirror} across {len(mappings)} windows without changing the pair.")


@scheduling_strategy("chorus", ("selecting the anchor identity", "ranking chorus speakers", "mapping anchor dialogue across bodies", "validating anchor consistency"))
def build_chorus_schedule(
    *, clips: list[dict[str, Any]], windows: list[dict[str, Any]], duration: float,
    parameters: dict[str, Any], seed: int = 1,
) -> dict[str, Any]:
    rng = random.Random(seed)
    clip_pools = _speaker_pools(clips)
    window_pools = _speaker_pools(windows)
    viable_anchors = {speaker: rows for speaker, rows in clip_pools.items() if rows}
    if not viable_anchors or len(window_pools) < 2:
        raise ValueError("Chorus requires an anchor dialogue pool and at least two viable diarized speakers.")
    anchor = str(parameters.get("anchor_speaker", "auto"))
    if anchor == "auto":
        anchor = max(viable_anchors, key=lambda speaker: (_pool_duration(viable_anchors[speaker]), len(viable_anchors[speaker]), speaker))
    if anchor not in viable_anchors:
        raise ValueError(f"Chorus anchor speaker '{anchor}' has no viable dialogue.")
    maximum = max(1, int(parameters.get("maximum_chorus_speakers", 4)))
    ranked_targets = sorted(
        (speaker for speaker in window_pools if speaker != anchor),
        key=lambda speaker: (_pool_duration(window_pools[speaker]), len(window_pools[speaker]), speaker),
        reverse=True,
    )[:maximum]
    if not ranked_targets:
        raise ValueError("Chorus could not find any non-anchor speakers.")
    ratio = INTENSITY_RATIOS.get(str(parameters.get("intensity", "Moderate")), 0.55)
    allow_reuse = bool(parameters.get("allow_line_reuse", False))
    destinations = sorted([row for speaker in ranked_targets for row in window_pools[speaker]], key=_start)
    selected = _evenly_select(destinations, max(1, round(len(destinations) * ratio)))
    used: set[str] = set()
    mappings: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for window in selected:
        candidates = [row for row in viable_anchors[anchor] if allow_reuse or _id(row) not in used]
        if not candidates:
            rejected.append(_rejection(window, "anchor_dialogue_pool_exhausted"))
            continue
        rng.shuffle(candidates)
        candidates.sort(key=lambda row: _duration_fit(_duration(row), _duration(window)), reverse=True)
        chosen = candidates[0]
        used.add(_id(chosen))
        mapping = _mapping(chosen, window, "chorus", "single_anchor_identity_across_speakers")
        mapping.update({"anchor_speaker": anchor, "chorus_speaker": _speaker(window), "anchor_identity_consistent": True})
        mappings.append(mapping)
    if not mappings:
        raise ValueError("Chorus found no viable anchor replacements.")
    consistent = all(row.get("source_speaker_id") == anchor and row.get("destination_speaker_id") != anchor for row in mappings)
    if not consistent:
        raise ValueError("Chorus validation failed: every mapping must use the anchor source identity.")
    transformed_speakers = sorted({str(row.get("destination_speaker_id")) for row in mappings})
    metrics = {
        "anchor_speaker": anchor, "chorus_speakers": transformed_speakers,
        "chorus_speaker_count": len(transformed_speakers), "maximum_chorus_speakers": maximum,
        "transformed_windows": len(mappings), "eligible_windows": len(destinations),
    }
    return _schedule("chorus", duration, mappings, rejected, metrics, {
        "passed": True, "all_sources_match_anchor": consistent, "maximum_chorus_speakers_respected": len(transformed_speakers) <= maximum,
    }, f"Chorus used {anchor} as one stable dialogue identity across {len(transformed_speakers)} other speakers and {len(mappings)} windows.")


@scheduling_strategy("foreshadow", ("searching future dialogue", "measuring temporal displacement", "protecting final-act windows", "validating the future-only rule"))
def build_foreshadow_schedule(
    *, clips: list[dict[str, Any]], windows: list[dict[str, Any]], duration: float,
    parameters: dict[str, Any], seed: int = 1,
) -> dict[str, Any]:
    rng = random.Random(seed)
    minimum = float(parameters.get("minimum_future_distance", 30.0))
    maximum = float(parameters.get("maximum_future_distance", max(duration, minimum + 0.1)))
    if maximum <= minimum:
        raise ValueError("Maximum future distance must be greater than minimum future distance.")
    ratio = INTENSITY_RATIOS.get(str(parameters.get("intensity", "Moderate")), 0.55)
    allow_reuse = bool(parameters.get("allow_line_reuse", False))
    policy = str(parameters.get("final_act_policy", "Gradually reduce"))
    eligible_windows = sorted([item for item in windows if _duration(item) > 0], key=_start)
    if policy == "Stop at cutoff":
        eligible_windows = [item for item in eligible_windows if _start(item) <= max(0.0, duration - minimum)]
    selected_windows = _evenly_select(eligible_windows, max(1, round(len(eligible_windows) * ratio)))
    used: set[str] = set()
    mappings: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    displacements: list[float] = []
    for window in selected_windows:
        lower = _start(window) + minimum
        upper = _start(window) + maximum
        candidates = [clip for clip in clips if lower < _start(clip) <= upper and (allow_reuse or _id(clip) not in used)]
        wrapped = False
        if not candidates and policy == "Explicit wraparound":
            candidates = [clip for clip in clips if _start(clip) < _start(window) and (allow_reuse or _id(clip) not in used)]
            wrapped = bool(candidates)
        if not candidates:
            rejected.append(_rejection(window, "no_future_dialogue_within_configured_distance"))
            continue
        rng.shuffle(candidates)
        candidates.sort(key=lambda clip: (_duration_fit(_duration(clip), _duration(window)), _start(clip)), reverse=True)
        chosen = candidates[0]
        used.add(_id(chosen))
        displacement = _start(chosen) - _start(window)
        displacements.append(displacement)
        mapping = _mapping(chosen, window, "foreshadow", "future_dialogue_only")
        mapping.update({"future_displacement": round(displacement, 3), "minimum_future_distance": minimum, "explicit_wraparound": wrapped})
        mappings.append(mapping)
    if not mappings:
        raise ValueError("Foreshadow found no viable future dialogue for the selected windows.")
    violations = [item for item in mappings if not item.get("explicit_wraparound") and float(item["future_displacement"]) <= minimum]
    if violations:
        raise ValueError(f"Foreshadow validation failed: {len(violations)} mappings violate the future-only rule.")
    metrics = {
        "average_temporal_displacement": round(sum(displacements) / len(displacements), 3),
        "minimum_temporal_displacement": round(min(displacements), 3),
        "maximum_temporal_displacement": round(max(displacements), 3),
        "eligible_windows": len(eligible_windows), "transformed_windows": len(mappings),
        "eligible_percentage_transformed": round(100 * len(mappings) / max(1, len(eligible_windows)), 2),
        "final_act_policy": policy, "fallback_count": sum(1 for item in mappings if item.get("explicit_wraparound")),
    }
    summary = (
        f"Foreshadow transformed {len(mappings)} windows using dialogue an average of "
        f"{metrics['average_temporal_displacement']:.1f} seconds later in the film. Final-act policy: {policy}."
    )
    return _schedule("foreshadow", duration, mappings, rejected, metrics, {
        "passed": True, "future_only_rule": not violations,
        "non_wraparound_sources_are_future_only": not violations,
        "wraparound_is_explicit": all(bool(item.get("explicit_wraparound")) == (float(item["future_displacement"]) < 0) for item in mappings),
        "explicit_wraparound_count": metrics["fallback_count"],
    }, summary)


@scheduling_strategy("flashback", ("searching earlier dialogue", "measuring past displacement", "protecting early-film windows", "validating the past-only rule"))
def build_flashback_schedule(
    *, clips: list[dict[str, Any]], windows: list[dict[str, Any]], duration: float,
    parameters: dict[str, Any], seed: int = 1,
) -> dict[str, Any]:
    rng = random.Random(seed)
    minimum = float(parameters.get("minimum_past_distance", 30.0))
    maximum = float(parameters.get("maximum_past_distance", max(duration, minimum + 0.1)))
    if maximum <= minimum:
        raise ValueError("Maximum past distance must be greater than minimum past distance.")
    ratio = INTENSITY_RATIOS.get(str(parameters.get("intensity", "Moderate")), 0.55)
    allow_reuse = bool(parameters.get("allow_line_reuse", False))
    eligible_windows = sorted([row for row in windows if _duration(row) > 0 and _start(row) > minimum], key=_start)
    selected = _evenly_select(eligible_windows, max(1, round(len(eligible_windows) * ratio)))
    used: set[str] = set()
    mappings: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    displacements: list[float] = []
    for window in selected:
        lower = _start(window) - maximum
        upper = _start(window) - minimum
        candidates = [row for row in clips if lower <= _start(row) < upper and (allow_reuse or _id(row) not in used)]
        if not candidates:
            rejected.append(_rejection(window, "no_past_dialogue_within_configured_distance"))
            continue
        rng.shuffle(candidates)
        candidates.sort(
            key=lambda row: (
                _duration_fit(_duration(row), _duration(window)),
                int(bool(_speaker(row) and _speaker(row) == _speaker(window))),
                _start(row),
            ),
            reverse=True,
        )
        chosen = candidates[0]
        used.add(_id(chosen))
        displacement = _start(window) - _start(chosen)
        displacements.append(displacement)
        mapping = _mapping(chosen, window, "flashback", "past_dialogue_only")
        mapping.update({"past_displacement": round(displacement, 3), "minimum_past_distance": minimum})
        mappings.append(mapping)
    if not mappings:
        raise ValueError("Flashback found no viable earlier dialogue for the selected windows.")
    valid = all(float(row["past_displacement"]) > minimum for row in mappings)
    if not valid:
        raise ValueError("Flashback validation failed: a mapping violates the past-only rule.")
    metrics = {
        "average_temporal_displacement": round(sum(displacements) / len(displacements), 3),
        "minimum_temporal_displacement": round(min(displacements), 3),
        "maximum_temporal_displacement": round(max(displacements), 3),
        "eligible_windows": len(eligible_windows), "transformed_windows": len(mappings),
    }
    return _schedule("flashback", duration, mappings, rejected, metrics, {
        "passed": True, "past_only_rule": valid, "all_sources_are_past_only": valid,
        "maximum_past_distance_is_respected": all(float(row["past_displacement"]) <= maximum for row in mappings),
    }, f"Flashback transformed {len(mappings)} later windows with dialogue averaging {metrics['average_temporal_displacement']:.1f} seconds earlier in the film.")


@scheduling_strategy("spiral", ("calculating expanding temporal targets", "alternating around the present", "selecting non-decreasing displacements", "validating spiral growth"))
def build_spiral_schedule(
    *, clips: list[dict[str, Any]], windows: list[dict[str, Any]], duration: float,
    parameters: dict[str, Any], seed: int = 1,
) -> dict[str, Any]:
    rng = random.Random(seed)
    starting = float(parameters.get("starting_distance", 10.0))
    maximum = float(parameters.get("maximum_distance", min(600.0, max(duration, starting + 0.1))))
    if maximum <= starting:
        raise ValueError("Spiral maximum distance must be greater than starting distance.")
    direction = str(parameters.get("direction", "Alternating"))
    ratio = INTENSITY_RATIOS.get(str(parameters.get("intensity", "Moderate")), 0.55)
    allow_reuse = bool(parameters.get("allow_line_reuse", False))
    eligible = sorted([row for row in windows if _duration(row) > 0], key=_start)
    selected = _evenly_select(eligible, max(1, round(len(eligible) * ratio)))
    used: set[str] = set()
    mappings: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    previous_distance = 0.0
    for index, window in enumerate(selected):
        position = index / max(1, len(selected) - 1)
        target = starting + (maximum - starting) * position
        desired_direction = direction
        if direction == "Alternating":
            desired_direction = "Past only" if index % 2 == 0 else "Future only"
        candidates = []
        for clip in clips:
            if not allow_reuse and _id(clip) in used:
                continue
            signed = _start(clip) - _start(window)
            distance = abs(signed)
            if distance + 1e-6 < max(starting, previous_distance) or distance > maximum:
                continue
            if desired_direction == "Past only" and signed >= 0:
                continue
            if desired_direction == "Future only" and signed <= 0:
                continue
            candidates.append(clip)
        if not candidates:
            rejected.append(_rejection(window, f"no_{desired_direction.lower().replace(' ', '_')}_candidate_preserves_spiral_growth"))
            continue
        rng.shuffle(candidates)
        candidates.sort(
            key=lambda row: (
                -abs(abs(_start(row) - _start(window)) - target),
                _duration_fit(_duration(row), _duration(window)),
            ),
            reverse=True,
        )
        chosen = candidates[0]
        used.add(_id(chosen))
        signed = _start(chosen) - _start(window)
        displacement = abs(signed)
        previous_distance = displacement
        mapping = _mapping(chosen, window, "spiral", "monotonically_expanding_temporal_revisit")
        mapping.update({
            "spiral_index": len(mappings), "spiral_target_distance": round(target, 3),
            "temporal_displacement": round(displacement, 3),
            "temporal_direction": "future" if signed > 0 else "past",
            "progression_value": round(position, 5),
        })
        mappings.append(mapping)
    if not mappings:
        raise ValueError("Spiral found no viable sequence of expanding temporal replacements.")
    distances = [float(row["temporal_displacement"]) for row in mappings]
    increasing = all(right + 1e-6 >= left for left, right in zip(distances, distances[1:]))
    if not increasing:
        raise ValueError("Spiral validation failed: absolute temporal displacement decreased.")
    metrics = {
        "direction": direction, "transformed_windows": len(mappings), "eligible_windows": len(eligible),
        "starting_displacement": round(distances[0], 3), "ending_displacement": round(distances[-1], 3),
        "maximum_configured_distance": maximum,
    }
    return _schedule("spiral", duration, mappings, rejected, metrics, {
        "passed": True, "absolute_displacement_never_decreases": increasing,
        "direction_policy_is_respected": all(
            direction == "Alternating"
            or (direction == "Past only" and row["temporal_direction"] == "past")
            or (direction == "Future only" and row["temporal_direction"] == "future")
            for row in mappings
        ),
    }, f"Spiral revisited {len(mappings)} moments while absolute temporal displacement grew from {distances[0]:.1f} to {distances[-1]:.1f} seconds.")


@scheduling_strategy("contagion", ("building speaker contact graph", "simulating exposure", "propagating infection", "generating infection timeline"))
def build_contagion_schedule(
    *, clips: list[dict[str, Any]], windows: list[dict[str, Any]], duration: float,
    parameters: dict[str, Any], seed: int = 1,
) -> dict[str, Any]:
    rng = random.Random(seed)
    clip_pools = _speaker_pools(clips)
    speakers = sorted(set(clip_pools) | set(_speaker_pools(windows)))
    if len(speakers) < 2:
        raise ValueError("Contagion requires at least two diarized speakers.")
    carrier = str(parameters.get("initial_carrier", "auto"))
    if carrier == "auto":
        carrier = max(clip_pools, key=lambda key: (_pool_duration(clip_pools[key]), len(clip_pools[key]), key))
    if carrier not in clip_pools:
        raise ValueError(f"Initial carrier '{carrier}' has no dialogue pool.")
    max_infected = min(len(speakers), max(1, int(parameters.get("maximum_infected_speakers", 4))))
    threshold = float(parameters.get("contact_threshold", 1.0))
    speed = {"Slow": 1.5, "Moderate": 1.0, "Fast": 0.5}.get(str(parameters.get("spread_speed", "Moderate")), 1.0)
    threshold *= speed
    scenes = _group_scenes(windows)
    graph = _speaker_contact_graph(scenes)
    infected: dict[str, dict[str, Any]] = {carrier: {"infection_time": 0.0, "infection_scene": "origin", "infecting_speaker": None, "confidence": 1.0}}
    exposure: defaultdict[tuple[str, str], float] = defaultdict(float)
    timeline: list[dict[str, Any]] = [{"speaker": carrier, "state": "fully_infected", "exposure_scene": None, "infection_scene": "origin", "infection_time": 0.0, "infecting_speaker": None, "infection_strength": 1.0}]
    for scene_id, scene_rows in scenes:
        scene_speakers = sorted({_speaker(item) for item in scene_rows if _speaker(item)})
        active_infected = [item for item in scene_speakers if item in infected and infected[item]["infection_time"] <= _start(scene_rows[0])]
        for source in active_infected:
            for target in scene_speakers:
                if target == source or target in infected or len(infected) >= max_infected:
                    continue
                exposure[(source, target)] += max(1.0, graph.get(source, {}).get(target, 1.0) / max(1, len(scenes)))
                if exposure[(source, target)] >= threshold:
                    infection_time = max(_start(scene_rows[0]), infected[source]["infection_time"])
                    confidence = min(1.0, exposure[(source, target)] / max(threshold, 0.001))
                    infected[target] = {"infection_time": infection_time, "infection_scene": scene_id, "infecting_speaker": source, "confidence": confidence}
                    timeline.append({
                        "speaker": target, "state": "fully_infected", "exposure_scene": scene_id,
                        "infection_scene": scene_id, "infection_time": round(infection_time, 3),
                        "infecting_speaker": source, "infection_strength": round(confidence, 4),
                    })
    if len(infected) < 2:
        raise ValueError("Contagion found no valid speaker exposure strong enough to spread infection.")

    ratio = INTENSITY_RATIOS.get(str(parameters.get("intensity", "Moderate")), 0.55)
    pool_policy = str(parameters.get("source_pool_policy", "Initial carrier"))
    source_pool = list(clip_pools[carrier])
    used: set[str] = set()
    mappings: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    transformed_by_speaker: Counter[str] = Counter()
    for index, window in enumerate(sorted(windows, key=_start)):
        target = _speaker(window)
        state = infected.get(target)
        if target == carrier or state is None or _start(window) < float(state["infection_time"]):
            continue
        progress = min(1.0, max(0.0, _start(window) / max(duration, 0.001)))
        strength = min(1.0, ratio * (0.35 + 0.65 * progress))
        if rng.random() > strength:
            continue
        if pool_policy == "Combined infected pool":
            allowed = [speaker for speaker, row in infected.items() if float(row["infection_time"]) <= _start(window)]
            candidates = [clip for speaker in allowed for clip in clip_pools.get(speaker, []) if _speaker(clip) != target]
        else:
            candidates = list(source_pool)
        candidates = [clip for clip in candidates if _id(clip) not in used]
        if not candidates:
            rejected.append(_rejection(window, "infected_pool_exhausted"))
            continue
        rng.shuffle(candidates)
        candidates.sort(key=lambda clip: _duration_fit(_duration(clip), _duration(window)), reverse=True)
        chosen = candidates[0]
        used.add(_id(chosen))
        mapping = _mapping(chosen, window, "contagion", "speaker_contact_infection")
        mapping.update({
            "infection_time": round(float(state["infection_time"]), 3), "infection_source": state["infecting_speaker"],
            "infection_strength": round(strength, 4), "progression_value": round(strength, 4),
        })
        mappings.append(mapping)
        transformed_by_speaker[target] += 1
    if not mappings:
        raise ValueError("Contagion spread successfully but produced no viable post-infection dialogue mappings.")
    for row in timeline:
        row["transformed_lines"] = transformed_by_speaker[row["speaker"]]
    premature = [item for item in mappings if float(item["destination_timestamp"]) < float(item["infection_time"])]
    if premature:
        raise ValueError("Contagion validation failed: a speaker was transformed before infection.")
    metrics = {
        "initial_carrier": carrier, "infected_speaker_count": len(infected), "maximum_infected_speakers": max_infected,
        "transformed_lines": len(mappings), "infection_timeline": timeline, "speaker_graph": graph,
    }
    summary = f"Contagion began with {carrier}, spread through measured contact to {len(infected) - 1} other speakers, and transformed {len(mappings)} post-infection lines."
    schedule = _schedule("contagion", duration, mappings, rejected, metrics, {
        "passed": True, "no_premature_infection": not premature, "no_mapping_precedes_infection": not premature,
        "maximum_infected_respected": len(infected) <= max_infected,
        "maximum_infected_speakers_is_respected": len(infected) <= max_infected,
    }, summary)
    schedule["infection_timeline"] = timeline
    schedule["speaker_graph"] = graph
    return schedule


@scheduling_strategy("bloom", ("calculating progression curve", "adjusting position-dependent candidate weights", "measuring transformation growth", "validating bloom progression"))
def build_bloom_schedule(
    *, clips: list[dict[str, Any]], windows: list[dict[str, Any]], duration: float,
    parameters: dict[str, Any], seed: int = 1,
) -> dict[str, Any]:
    rng = random.Random(seed)
    start_strength = float(parameters.get("starting_intensity", 0.05))
    end_strength = float(parameters.get("ending_intensity", 0.95))
    if end_strength < start_strength:
        raise ValueError("Bloom ending intensity must not be lower than starting intensity.")
    curve_name = str(parameters.get("curve_shape", "Gentle nonlinear"))
    preserve_ending = bool(parameters.get("preserve_ending_coherence", True))
    max_distance = float(parameters.get("maximum_temporal_distance", max(duration, 1.0)))
    ordered_windows = sorted([item for item in windows if _duration(item) > 0], key=_start)
    mappings: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    used: set[str] = set()
    segment_rows: defaultdict[int, list[float]] = defaultdict(list)
    for index, window in enumerate(ordered_windows):
        position = min(1.0, max(0.0, _start(window) / max(duration, 0.001)))
        curve = _curve(position, curve_name)
        strength = start_strength + (end_strength - start_strength) * curve
        cadence = ((index * 0.61803398875) + (seed % 17) / 17.0) % 1.0
        if cadence > strength:
            continue
        candidates = [
            clip for clip in clips
            if _id(clip) not in used and abs(_start(clip) - _start(window)) > 0.5
            and abs(_start(clip) - _start(window)) <= max_distance
        ]
        if not candidates:
            rejected.append(_rejection(window, "no_bloom_candidate"))
            continue
        rng.shuffle(candidates)
        def score(clip: dict[str, Any]) -> float:
            fit = _duration_fit(_duration(clip), _duration(window))
            distance = min(1.0, abs(_start(clip) - _start(window)) / max(max_distance, 0.001))
            identity_mismatch = float(bool(_speaker(clip) and _speaker(window) and _speaker(clip) != _speaker(window)))
            early_score = 0.75 * fit + 0.2 * (1.0 - distance) + 0.05 * (1.0 - identity_mismatch)
            late_score = 0.25 * fit + 0.45 * distance + 0.3 * identity_mismatch
            return (1.0 - strength) * early_score + strength * late_score
        candidates.sort(key=score, reverse=True)
        chosen = candidates[0]
        used.add(_id(chosen))
        mapping = _mapping(chosen, window, "bloom", "position_weighted_progressive_transformation")
        distance = abs(_start(chosen) - _start(window))
        identity_instability = float(bool(_speaker(chosen) and _speaker(window) and _speaker(chosen) != _speaker(window)))
        performance_mismatch = 1.0 - _duration_fit(_duration(chosen), _duration(window))
        combined = 0.4 * strength + 0.25 * min(1.0, distance / max(max_distance, 0.001)) + 0.2 * identity_instability + 0.15 * performance_mismatch
        if preserve_ending and position > 0.92:
            combined = min(combined, 0.9)
        mapping.update({
            "normalized_output_position": round(position, 5), "progression_value": round(strength, 5),
            "temporal_displacement": round(distance, 3), "identity_instability": identity_instability,
            "performance_mismatch": round(performance_mismatch, 5), "combined_transformation_score": round(combined, 5),
        })
        mappings.append(mapping)
        segment_rows[min(4, int(position * 5))].append(combined)
    if not mappings:
        raise ValueError("Bloom found no viable progressive replacements.")
    profile = []
    for segment in range(5):
        rows = [item for item in mappings if min(4, int(float(item["normalized_output_position"]) * 5)) == segment]
        profile.append({
            "segment": segment + 1, "start_fraction": segment / 5, "end_fraction": (segment + 1) / 5,
            "eligible_windows": sum(1 for item in ordered_windows if min(4, int((_start(item) / max(duration, 0.001)) * 5)) == segment),
            "replacement_count": len(rows),
            "replacement_percentage": round(100 * len(rows) / max(1, sum(1 for item in ordered_windows if min(4, int((_start(item) / max(duration, 0.001)) * 5)) == segment)), 2),
            "average_transformation_score": round(sum(float(item["combined_transformation_score"]) for item in rows) / len(rows), 5) if rows else 0.0,
            "average_temporal_displacement": round(sum(float(item["temporal_displacement"]) for item in rows) / len(rows), 3) if rows else 0.0,
            "identity_instability": round(sum(float(item["identity_instability"]) for item in rows) / len(rows), 5) if rows else 0.0,
            "performance_mismatch": round(sum(float(item["performance_mismatch"]) for item in rows) / len(rows), 5) if rows else 0.0,
        })
    early = [item["average_transformation_score"] for item in profile[:2] if item["replacement_count"]]
    late = [item["average_transformation_score"] for item in profile[-2:] if item["replacement_count"]]
    increasing = bool(early and late and sum(late) / len(late) >= sum(early) / len(early))
    if not increasing:
        raise ValueError("Bloom validation failed: measured late transformation strength did not exceed early strength.")
    metrics = {"curve_shape": curve_name, "bloom_profile": profile, "transformed_windows": len(mappings), "progression_increased": increasing}
    summary = f"Bloom transformed {len(mappings)} windows along a {curve_name.lower()} curve; measured late-output transformation strength exceeded early-output strength."
    return _schedule("bloom", duration, mappings, rejected, metrics, {
        "passed": True, "aggregate_strength_increases": increasing,
        "late_strength_is_not_lower_than_early_strength": increasing,
        "ending_coherence_cap_is_respected": all(float(row["combined_transformation_score"]) <= 0.9 for row in mappings if float(row["normalized_output_position"]) > 0.92) if preserve_ending else True,
    }, summary)


def representative_preview_regions(filter_id: str, schedule: dict[str, Any], *, maximum: int = 3) -> list[dict[str, Any]]:
    mappings = [item for item in schedule.get("mappings", []) if item.get("enabled", True)]
    if not mappings:
        return []
    if filter_id == "bloom":
        ordered = sorted(mappings, key=lambda item: float(item.get("normalized_output_position", 0.0)))
        chosen = [ordered[0], ordered[len(ordered) // 2], ordered[-1]]
    elif filter_id == "foreshadow":
        chosen = sorted(mappings, key=lambda item: float(item.get("future_displacement", 0.0)), reverse=True)[:maximum]
    elif filter_id == "possession":
        chosen = sorted(mappings, key=lambda item: float(item.get("duration_fit", 0.0)), reverse=True)[:maximum]
    elif filter_id == "contagion":
        chosen = sorted(mappings, key=lambda item: (float(item.get("infection_time", 0.0)), float(item.get("destination_timestamp", 0.0))))[:maximum]
    else:
        chosen = mappings[:maximum]
    return [
        {"start": max(0.0, float(item.get("destination_timestamp", 0.0)) - 2.0), "duration": min(14.0, float(item.get("planned_render_duration", 0.0)) + 4.0), "mapping_id": item.get("window_id")}
        for item in chosen[:maximum]
    ]


def _schedule(filter_id: str, duration: float, mappings: list[dict[str, Any]], rejected: list[dict[str, Any]], metrics: dict[str, Any], validation: dict[str, Any], summary: str) -> dict[str, Any]:
    return {
        "schema_version": "1.0", "tool_version": __version__, "creation_timestamp": utc_now(),
        "transformation_name": f"filter_{filter_id}", "mutation_id": filter_id,
        "render_duration": round(duration, 3), "mappings": mappings,
        "rejected_candidates": rejected, "filter_metrics": metrics, "filter_validation": validation,
        "filter_summary": summary, "preview_regions": representative_preview_regions(filter_id, {"mappings": mappings}),
    }


def _mapping(clip: dict[str, Any], window: dict[str, Any], operation: str, reason: str) -> dict[str, Any]:
    clip_duration = _duration(clip)
    window_duration = _duration(window) or clip_duration
    trim = min(clip_duration, window_duration) if clip_duration > 0 and window_duration > 0 else max(clip_duration, window_duration)
    destination = _start(window)
    return {
        "window_id": _id(window), "clip_id": _id(clip), "clip_path": clip.get("path"), "enabled": True,
        "destination_timestamp": round(destination, 3), "alignment_slot_start": round(destination, 3),
        "alignment_slot_end": round(destination + window_duration, 3), "stretch_factor": 1.0,
        "clip_trim_start": 0.0, "clip_trim_duration": round(trim, 3), "leading_silence": 0.0, "trailing_silence": 0.0,
        "planned_render_duration": round(window_duration, 3), "score": 1.0, "score_components": {},
        "selection_reason": reason, "scheduling_mode": f"filter_{operation}", "timing_strategy": "whole_line_preserved",
        "render_operations": [], "shot_boundary_mode": "off", "visual_fit_score": 1.0,
        "mutation_operation": operation, "source_transcript": clip.get("transcript", clip.get("text", "")),
        "source_speaker_id": _speaker(clip), "destination_speaker_id": _speaker(window),
        "source_movie_timestamp": round(_start(clip), 3), "clip_movie_timestamp": round(_start(clip), 3),
    }


def _speaker_pools(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    pools: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        speaker = _speaker(row)
        if speaker and not str(speaker).startswith("unknown_") and not str(speaker).startswith("fallback_"):
            pools[str(speaker)].append(row)
    return dict(pools)


def _speaker(row: dict[str, Any]) -> str | None:
    value = row.get("speaker_id") or row.get("speaker") or row.get("dominant_speaker_id") or row.get("destination_speaker_id")
    return str(value) if value not in {None, ""} else None


def _start(row: dict[str, Any]) -> float:
    for key in ("movie_timestamp", "start", "destination_timestamp", "alignment_slot_start"):
        if row.get(key) is not None:
            return float(row[key])
    return 0.0


def _duration(row: dict[str, Any]) -> float:
    if row.get("duration") is not None:
        return max(0.0, float(row["duration"]))
    if row.get("end") is not None:
        return max(0.0, float(row["end"]) - _start(row))
    return max(0.0, float(row.get("planned_render_duration", row.get("clip_trim_duration", 0.0)) or 0.0))


def _id(row: dict[str, Any]) -> str:
    return str(row.get("id") or row.get("window_id") or row.get("clip_id") or f"row_{_start(row):.3f}")


def _id_from_mapping(row: dict[str, Any]) -> str:
    return str(row.get("clip_id"))


def _pool_duration(rows: list[dict[str, Any]]) -> float:
    return sum(_duration(item) for item in rows)


def _duration_fit(source: float, destination: float) -> float:
    if source <= 0 or destination <= 0:
        return 0.0
    return min(source, destination) / max(source, destination)


def _evenly_select(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if count >= len(rows):
        return list(rows)
    if count <= 0 or not rows:
        return []
    indices = sorted({min(len(rows) - 1, round(index * (len(rows) - 1) / max(1, count - 1))) for index in range(count)})
    return [rows[index] for index in indices]


def _rejection(window: dict[str, Any], reason: str) -> dict[str, Any]:
    return {"window_id": _id(window), "destination_start": round(_start(window), 3), "reason": reason}


def _scene_id(row: dict[str, Any]) -> str:
    value = row.get("scene_id") or row.get("dialogue_scene_id") or row.get("performance_id")
    return str(value) if value else f"time_{int(_start(row) // 30):04d}"


def _group_scenes(windows: list[dict[str, Any]]) -> list[tuple[str, list[dict[str, Any]]]]:
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for window in windows:
        grouped[_scene_id(window)].append(window)
    rows = [(key, sorted(items, key=_start)) for key, items in grouped.items()]
    return sorted(rows, key=lambda item: _start(item[1][0]) if item[1] else math.inf)


def _speaker_contact_graph(scenes: list[tuple[str, list[dict[str, Any]]]]) -> dict[str, dict[str, float]]:
    graph: defaultdict[str, defaultdict[str, float]] = defaultdict(lambda: defaultdict(float))
    for _scene, rows in scenes:
        speakers = sorted({_speaker(item) for item in rows if _speaker(item)})
        for index, source in enumerate(speakers):
            for target in speakers[index + 1:]:
                graph[source][target] += 1.0
                graph[target][source] += 1.0
        for left, right in zip(rows, rows[1:]):
            source, target = _speaker(left), _speaker(right)
            if source and target and source != target:
                graph[source][target] += 0.5
                graph[target][source] += 0.5
    return {source: {target: round(weight, 3) for target, weight in targets.items()} for source, targets in graph.items()}


def _curve(position: float, name: str) -> float:
    if name == "Linear":
        return position
    if name == "Late surge":
        return position ** 3
    if name == "Early surge":
        return math.sqrt(position)
    return position ** 1.7
