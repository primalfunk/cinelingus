from __future__ import annotations

from collections import defaultdict
from typing import Any

from cinelingus import __version__
from cinelingus.util import utc_now

from .acceptance import FULL_LENGTH_DIALOGUE_ACCEPTANCE_REQUIREMENTS
from .models import FilmInput
from .strategies import INTENSITY_RATIOS


SUPPORTED_MULTIWORLD_DIALOGUE_LAWS = {
    "multiworld.possession",
    "multiworld.contagion",
    "multiworld.echo_chamber",
    "multiworld.prophecy",
    "multiworld.triangle",
}


def build_multiworld_schedule(
    filter_id: str,
    *,
    films: tuple[FilmInput, ...],
    film_artifacts: dict[str, dict[str, Any]],
    parameters: dict[str, Any],
    seed: int = 1,
) -> dict[str, Any]:
    if filter_id not in SUPPORTED_MULTIWORLD_DIALOGUE_LAWS:
        raise ValueError(f"No executable Multiworld dialogue law is registered for '{filter_id}'.")
    anchor = next(film for film in films if film.is_anchor)
    donors = tuple(film for film in films if not film.is_anchor)
    anchor_artifacts = film_artifacts[anchor.id]
    if filter_id == "multiworld.triangle":
        mappings, rejected, metrics, validation, visual_segments, duration = _triangle(
            films, film_artifacts, parameters
        )
    else:
        duration = float(anchor_artifacts["movie"]["duration"])
        mappings, rejected, metrics, validation, visual_segments = [], [], {}, {}, []
    windows = sorted(
        (row for row in anchor_artifacts.get("windows", []) if _duration(row) > 0),
        key=lambda row: (_start(row), _id(row)),
    )
    if filter_id != "multiworld.triangle" and not windows:
        raise ValueError(f"{filter_id} requires positive-duration dialogue windows in the anchor film.")
    for film in films:
        if not film_artifacts.get(film.id, {}).get("clips"):
            raise ValueError(f"{filter_id} requires usable dialogue clips from {film.label}.")

    if filter_id == "multiworld.triangle":
        pass
    elif filter_id == "multiworld.possession":
        mappings, rejected, metrics, validation = _possession(
            anchor, donors[0], film_artifacts, windows, duration, parameters
        )
    elif filter_id == "multiworld.contagion":
        mappings, rejected, metrics, validation = _contagion(
            anchor, donors, film_artifacts, windows, duration, parameters
        )
    elif filter_id == "multiworld.echo_chamber":
        mappings, rejected, metrics, validation = _echo_chamber(
            anchor, films, film_artifacts, windows, duration, parameters
        )
    else:
        mappings, rejected, metrics, validation = _prophecy(
            anchor, donors[0], film_artifacts, windows, duration, parameters
        )
    if not mappings:
        raise ValueError(f"{filter_id} found no viable cross-film dialogue placements.")
    validation["passed"] = all(value is not False for key, value in validation.items() if key != "passed")
    if not validation["passed"]:
        failed = ", ".join(key for key, value in validation.items() if key != "passed" and value is False)
        raise ValueError(f"{filter_id} could not satisfy its defining law: {failed}.")
    source_hashes = sorted({str(row["source_media_hash"]) for row in mappings})
    result = {
        "schema_version": "1.0",
        "tool_version": __version__,
        "creation_timestamp": utc_now(),
        "transformation_name": f"filter_{filter_id.replace('.', '_')}",
        "mutation_id": filter_id,
        "filter_id": filter_id,
        "render_duration": round(duration, 3),
        "mappings": mappings,
        "rejected_candidates": rejected,
        "filter_metrics": metrics,
        "filter_validation": validation,
        "audio_activity_basis": "rendered_mix",
        "acceptance_requirements": dict(FULL_LENGTH_DIALOGUE_ACCEPTANCE_REQUIREMENTS),
        "filter_summary": (
            f"{filter_id.split('.', 1)[1].replace('_', ' ').title()} applied its cinematic law across "
            f"{len(films)} films with {len(mappings)} provenance-bearing placements."
        ),
        "destination_media_hash": str(anchor_artifacts["media_hash"]),
        "source_media_hashes": source_hashes,
        "multiworld": {
            "anchor_film_id": anchor.id,
            "film_ids": [film.id for film in films],
            "film_media_hashes": {film.id: str(film_artifacts[film.id]["media_hash"]) for film in films},
            "film_paths": {film.id: str(film.media_path) for film in films},
            "seed": int(seed),
        },
        "preview_regions": [
            {
                "start": max(0.0, float(row["destination_timestamp"]) - 2.0),
                "duration": min(14.0, float(row["planned_render_duration"]) + 4.0),
                "mapping_id": row["window_id"],
            }
            for row in mappings[:3]
        ],
    }
    if visual_segments:
        result["visual_segments"] = visual_segments
        result["visual_source_media_hashes"] = sorted({str(row["source_media_hash"]) for row in visual_segments})
    return result


def _triangle(
    films: tuple[FilmInput, ...],
    artifacts: dict[str, dict[str, Any]],
    parameters: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any], list[dict[str, Any]], float]:
    """Build a closed A->B->C visual cycle with B->C->A dialogue exchange."""
    if len(films) != 3:
        raise ValueError("Multiworld Triangle requires exactly three films.")
    duration = round(min(float(artifacts[film.id]["movie"]["duration"]) for film in films), 3)
    if duration <= 0:
        raise ValueError("Multiworld Triangle requires three positive-duration films.")
    boundaries = [round(duration * index / 3, 3) for index in range(4)]
    boundaries[-1] = duration
    mappings: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    visual_segments: list[dict[str, Any]] = []
    phase_pairs: list[dict[str, str | int]] = []
    for phase, visual_film in enumerate(films):
        dialogue_film = films[(phase + 1) % len(films)]
        start, end = boundaries[phase], boundaries[phase + 1]
        shots = [
            row for row in artifacts[visual_film.id].get("visual", {}).get("shots", {}).get("shots", [])
            if _start(row) < end and float(row.get("end", _start(row))) > start
        ]
        visual_segments.append({
            "id": f"triangle_phase_{phase + 1}",
            "phase": phase + 1,
            "source_film_id": visual_film.id,
            "source_media_hash": str(artifacts[visual_film.id]["media_hash"]),
            "source_path": str(visual_film.media_path),
            "source_start": start,
            "source_end": end,
            "output_start": start,
            "output_end": end,
            "shot_ids": [str(row.get("id")) for row in shots] or [f"{visual_film.id}_phase_{phase + 1}"],
        })
        phase_pairs.append({
            "phase": phase + 1,
            "visual_film_id": visual_film.id,
            "dialogue_film_id": dialogue_film.id,
        })
        carrier_windows = [
            row for row in artifacts[visual_film.id].get("windows", [])
            if _start(row) < end and _start(row) + _duration(row) > start
        ]
        windows = [
            row for row in artifacts[visual_film.id].get("windows", [])
            if start <= _start(row) and _start(row) + _duration(row) <= end
        ]
        visual_segments[-1]["carrier_speech_regions"] = [
            {
                "id": str(window.get("id") or f"{visual_film.id}_speech_{index + 1}"),
                "source_film_id": visual_film.id,
                "triangle_phase": phase + 1,
                "source_start": round(max(start, _start(window)), 3),
                "source_end": round(min(end, _start(window) + _duration(window)), 3),
            }
            for index, window in enumerate(carrier_windows)
            if min(end, _start(window) + _duration(window)) > max(start, _start(window))
        ]
        clips = list(artifacts[dialogue_film.id].get("clips", []))
        if not windows or not clips:
            rejected.append({
                "phase": phase + 1,
                "reason": "phase_requires_visual_film_windows_and_dialogue_film_clips",
                "visual_film_id": visual_film.id,
                "dialogue_film_id": dialogue_film.id,
            })
            continue
        selected = _select(windows, parameters)[:len(clips)]
        unused = list(clips)
        for index, window in enumerate(selected):
            source = _best_duration(unused, window, index)
            unused.remove(source)
            mapping = _mapping(source, window, dialogue_film, visual_film, artifacts, "closed_triangle_dialogue_exchange")
            mapping.update({
                "triangle_phase": phase + 1,
                "visual_film_id": visual_film.id,
                "dialogue_film_id": dialogue_film.id,
                "progression_value": round((phase + 1) / 3, 4),
            })
            mappings.append(mapping)
    observed_visual = [str(row["source_film_id"]) for row in visual_segments]
    observed_dialogue = {str(row["source_film_id"]) for row in mappings}
    expected = {film.id for film in films}
    validation = {
        "passed": True,
        "exactly_three_films": len(films) == 3,
        "visual_cycle_is_a_b_c": observed_visual == [film.id for film in films],
        "dialogue_cycle_is_b_c_a": all(
            row["dialogue_film_id"] == films[int(row["triangle_phase"]) % 3].id for row in mappings
        ),
        "no_phase_uses_its_own_dialogue": all(row["visual_film_id"] != row["dialogue_film_id"] for row in mappings),
        "every_film_contributes_visuals_and_dialogue": set(observed_visual) == expected and observed_dialogue == expected,
        "source_dialogue_is_not_reused": len({row["clip_id"] for row in mappings}) == len(mappings),
        "carrier_speech_regions_declared": all(
            bool(row.get("carrier_speech_regions")) for row in visual_segments
        ),
    }
    return mappings, rejected, {
        "phase_pairs": phase_pairs,
        "visual_contributing_films": observed_visual,
        "dialogue_contributing_films": sorted(observed_dialogue),
        "shared_timeline_duration": duration,
        "carrier_speech_region_count": sum(
            len(row.get("carrier_speech_regions", [])) for row in visual_segments
        ),
    }, validation, visual_segments, duration


def _possession(
    anchor: FilmInput,
    donor: FilmInput,
    artifacts: dict[str, dict[str, Any]],
    windows: list[dict[str, Any]],
    duration: float,
    parameters: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    window_pools = _speaker_pools(windows)
    donor_pools = _speaker_pools(artifacts[donor.id]["clips"])
    if not window_pools or not donor_pools:
        raise ValueError("Multiworld Possession requires speaker identities in both films.")
    host = max(window_pools, key=lambda key: (_pool_duration(window_pools[key]), key))
    possessor = max(donor_pools, key=lambda key: (_pool_duration(donor_pools[key]), key))
    selected = _select(window_pools[host], parameters)
    maximum_unique_placements = max(1, int(len(donor_pools[possessor]) / 0.8))
    selected = selected[:maximum_unique_placements]
    mappings = []
    source_use_counts: defaultdict[str, int] = defaultdict(int)
    for index, window in enumerate(selected):
        least_used = min(source_use_counts[_id(row)] for row in donor_pools[possessor])
        available = [row for row in donor_pools[possessor] if source_use_counts[_id(row)] == least_used]
        source = _best_duration(available, window, index)
        source_use_counts[_id(source)] += 1
        mapping = _mapping(source, window, donor, anchor, artifacts, "stable_cross_film_identity_possession")
        mapping.update({"host_speaker": _qualified_speaker(anchor, host), "possessing_speaker": _qualified_speaker(donor, possessor)})
        mappings.append(mapping)
    return mappings, [], {
        "host_speaker": _qualified_speaker(anchor, host),
        "possessing_speaker": _qualified_speaker(donor, possessor),
        "contributing_films": [donor.id],
    }, {
        "passed": True,
        "exactly_two_films": len(artifacts) == 2,
        "possessing_identity_is_stable": len({row["source_speaker_id"] for row in mappings}) == 1,
        "host_identity_is_stable": len({row["destination_speaker_id"] for row in mappings}) == 1,
        "every_source_comes_from_donor_film": all(row["source_film_id"] == donor.id for row in mappings),
    }


def _contagion(
    anchor: FilmInput,
    donors: tuple[FilmInput, ...],
    artifacts: dict[str, dict[str, Any]],
    windows: list[dict[str, Any]],
    duration: float,
    parameters: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    selected = _select(windows, parameters, minimum=len(donors))
    mappings: list[dict[str, Any]] = []
    donor_sequence: list[int] = []
    for index, window in enumerate(selected):
        donor_index = min(len(donors) - 1, (index * len(donors)) // len(selected))
        donor = donors[donor_index]
        clips = artifacts[donor.id]["clips"]
        source = _best_duration(clips, window, index)
        mapping = _mapping(source, window, donor, anchor, artifacts, "ordered_cross_film_infection_phase")
        mapping.update({
            "infection_phase": donor_index + 1,
            "infecting_film_id": donor.id,
            "progression_value": round((donor_index + 1) / len(donors), 4),
        })
        mappings.append(mapping)
        donor_sequence.append(donor_index)
    observed = {row["source_film_id"] for row in mappings}
    return mappings, [], {
        "infection_order": [film.id for film in donors],
        "contributing_films": sorted(observed),
        "phase_count": len(donors),
    }, {
        "passed": True,
        "infection_phases_never_revert": all(left <= right for left, right in zip(donor_sequence, donor_sequence[1:])),
        "every_donor_film_contributes": observed == {film.id for film in donors},
        "anchor_timeline_is_preserved": all(0 <= float(row["destination_timestamp"]) <= duration for row in mappings),
    }


def _echo_chamber(
    anchor: FilmInput,
    films: tuple[FilmInput, ...],
    artifacts: dict[str, dict[str, Any]],
    windows: list[dict[str, Any]],
    duration: float,
    parameters: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    selected = _select(windows, parameters)
    delay = max(0.04, float(parameters.get("layer_delay_seconds", 0.18)))
    attenuation = min(-0.1, float(parameters.get("layer_attenuation_db", -4.0)))
    mappings: list[dict[str, Any]] = []
    group_sizes: dict[str, int] = {}
    for window_index, window in enumerate(selected):
        group_id = f"echo_{_id(window)}"
        count = 0
        for layer, film in enumerate(films):
            source = _best_duration(artifacts[film.id]["clips"], window, window_index + layer)
            mapping = _mapping(source, window, film, anchor, artifacts, "multi_film_staggered_echo_layer")
            destination = min(max(0.0, duration - 0.001), _start(window) + layer * delay)
            rendered_duration = max(0.001, min(_duration(window), duration - destination))
            mapping.update({
                "destination_timestamp": round(destination, 3),
                "alignment_slot_start": round(destination, 3),
                "alignment_slot_end": round(destination + rendered_duration, 3),
                "planned_render_duration": round(rendered_duration, 3),
                "clip_trim_duration": round(min(float(mapping["clip_trim_duration"]), rendered_duration), 3),
                "echo_group_id": group_id,
                "echo_layer": layer,
                "echo_delay_seconds": round(layer * delay, 3),
                "gain_db": round(layer * attenuation, 2),
            })
            mappings.append(mapping)
            count += 1
        group_sizes[group_id] = count
    observed = {row["source_film_id"] for row in mappings}
    groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in mappings:
        groups[str(row["echo_group_id"])].append(row)
    return mappings, [], {
        "echo_group_count": len(groups),
        "echo_group_sizes": group_sizes,
        "contributing_films": sorted(observed),
        "layer_delay_seconds": delay,
        "layer_attenuation_db": attenuation,
    }, {
        "passed": True,
        "every_film_contributes": observed == {film.id for film in films},
        "every_echo_group_contains_every_film": all(len(rows) == len(films) for rows in groups.values()),
        "echo_delays_never_decrease": all(
            all(left["echo_delay_seconds"] <= right["echo_delay_seconds"] for left, right in zip(rows, rows[1:]))
            for rows in groups.values()
        ),
        "later_layers_are_not_louder": all(
            all(left["gain_db"] >= right["gain_db"] for left, right in zip(rows, rows[1:]))
            for rows in groups.values()
        ),
    }


def _prophecy(
    anchor: FilmInput,
    donor: FilmInput,
    artifacts: dict[str, dict[str, Any]],
    windows: list[dict[str, Any]],
    duration: float,
    parameters: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    donor_duration = float(artifacts[donor.id]["movie"]["duration"])
    lead = max(0.01, min(0.8, float(parameters.get("minimum_normalized_lead", 0.15))))
    selected = _select(windows, parameters)
    mappings: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for index, window in enumerate(selected):
        destination_position = _start(window) / max(duration, 0.001)
        candidates = [
            row for row in artifacts[donor.id]["clips"]
            if _start(row) / max(donor_duration, 0.001) >= destination_position + lead
        ]
        if not candidates:
            rejected.append({"window_id": _id(window), "reason": "no_donor_line_far_enough_in_normalized_future"})
            continue
        source = _best_duration(candidates, window, index)
        source_position = _start(source) / max(donor_duration, 0.001)
        mapping = _mapping(source, window, donor, anchor, artifacts, "cross_film_normalized_future_prediction")
        mapping.update({
            "source_normalized_position": round(source_position, 4),
            "destination_normalized_position": round(destination_position, 4),
            "normalized_prophecy_lead": round(source_position - destination_position, 4),
        })
        mappings.append(mapping)
    return mappings, rejected, {
        "minimum_normalized_lead": lead,
        "mean_normalized_lead": round(sum(row["normalized_prophecy_lead"] for row in mappings) / len(mappings), 4) if mappings else 0.0,
        "contributing_films": [donor.id] if mappings else [],
    }, {
        "passed": True,
        "every_prophecy_source_is_in_the_normalized_future": bool(mappings) and all(row["normalized_prophecy_lead"] >= lead for row in mappings),
        "every_source_comes_from_prophetic_film": all(row["source_film_id"] == donor.id for row in mappings),
        "anchor_timeline_is_preserved": all(0 <= float(row["destination_timestamp"]) <= duration for row in mappings),
    }


def _mapping(
    source: dict[str, Any],
    window: dict[str, Any],
    source_film: FilmInput,
    anchor: FilmInput,
    artifacts: dict[str, dict[str, Any]],
    reason: str,
) -> dict[str, Any]:
    source_duration = _duration(source)
    destination_duration = _duration(window)
    trim = min(source_duration, destination_duration)
    destination = _start(window)
    return {
        "window_id": _id(window),
        "clip_id": f"{source_film.id}:{_id(source)}",
        "clip_path": source.get("path"),
        "enabled": True,
        "destination_timestamp": round(destination, 3),
        "alignment_slot_start": round(destination, 3),
        "alignment_slot_end": round(destination + destination_duration, 3),
        "stretch_factor": 1.0,
        "clip_trim_start": 0.0,
        "clip_trim_duration": round(trim, 3),
        "planned_render_duration": round(destination_duration, 3),
        "leading_silence": 0.0,
        "trailing_silence": 0.0,
        "score": 1.0,
        "score_components": {},
        "selection_reason": reason,
        "scheduling_mode": "multiworld_dialogue_law",
        "timing_strategy": "anchor_window_preserved",
        "render_operations": [],
        "shot_boundary_mode": "off",
        "visual_fit_score": 1.0,
        "mutation_operation": reason,
        "source_transcript": source.get("transcript", source.get("text", "")),
        "source_speaker_id": _qualified_speaker(source_film, _speaker(source)),
        "destination_speaker_id": _qualified_speaker(anchor, _speaker(window)),
        "source_movie_timestamp": round(_start(source), 3),
        "clip_movie_timestamp": round(_start(source), 3),
        "source_film_id": source_film.id,
        "destination_film_id": anchor.id,
        "source_media_hash": str(artifacts[source_film.id]["media_hash"]),
        "destination_media_hash": str(artifacts[anchor.id]["media_hash"]),
    }


def _select(rows: list[dict[str, Any]], parameters: dict[str, Any], *, minimum: int = 1) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: (_start(row), _id(row)))
    ratio = INTENSITY_RATIOS.get(str(parameters.get("intensity", "Moderate")), 0.55)
    count = min(len(ordered), max(minimum, round(len(ordered) * ratio)))
    if count >= len(ordered):
        return ordered
    indices = sorted({round(index * (len(ordered) - 1) / max(1, count - 1)) for index in range(count)})
    return [ordered[index] for index in indices]


def _speaker_pools(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    pools: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        speaker = _speaker(row)
        if speaker and not speaker.startswith(("unknown_", "fallback_")):
            pools[speaker].append(row)
    return dict(pools)


def _best_duration(rows: list[dict[str, Any]], window: dict[str, Any], offset: int) -> dict[str, Any]:
    ranked = sorted(
        rows,
        key=lambda row: (
            min(_duration(row), _duration(window)) / max(_duration(row), _duration(window)),
            -abs(_start(row) - _start(window)),
            _id(row),
        ),
        reverse=True,
    )
    return ranked[offset % min(len(ranked), max(1, len(ranked)))]


def _qualified_speaker(film: FilmInput, speaker: str | None) -> str | None:
    return f"{film.id}:{speaker}" if speaker else None


def _speaker(row: dict[str, Any]) -> str | None:
    value = row.get("speaker_id") or row.get("speaker") or row.get("dominant_speaker_id")
    return str(value) if value not in {None, ""} else None


def _start(row: dict[str, Any]) -> float:
    for key in ("movie_timestamp", "start", "destination_timestamp"):
        if row.get(key) is not None:
            return float(row[key])
    return 0.0


def _duration(row: dict[str, Any]) -> float:
    if row.get("duration") is not None:
        return max(0.0, float(row["duration"]))
    if row.get("end") is not None:
        return max(0.0, float(row["end"]) - _start(row))
    return max(0.0, float(row.get("planned_render_duration", 0.0) or 0.0))


def _id(row: dict[str, Any]) -> str:
    return str(row.get("id") or row.get("window_id") or row.get("clip_id") or f"row_{_start(row):.3f}")


def _pool_duration(rows: list[dict[str, Any]]) -> float:
    return sum(_duration(row) for row in rows)
