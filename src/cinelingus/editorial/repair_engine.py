from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable

from .editorial_memory import EditorialMemory
from .repair_strategies import repair_strategy_for

ScoreCandidate = Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]
BuildMapping = Callable[[dict[str, Any], dict[str, Any], dict[str, Any]], dict[str, Any]]


def build_repair_batch(
    *,
    schedule: dict[str, Any],
    decisions: dict[str, Any],
    donor_candidates: list[dict[str, Any]],
    memory: EditorialMemory,
    score_candidate: ScoreCandidate,
    build_mapping: BuildMapping,
    maximum_repairs: int = 24,
    minimum_predicted_gain: float = 0.01,
) -> dict[str, Any]:
    repaired = deepcopy(schedule)
    mappings = repaired.get("mappings", [])
    used = {str(row.get("clip_id")) for row in mappings if row.get("enabled", True)}
    donors_by_id = {str(row.get("id")): row for row in donor_candidates if row.get("id")}
    proposals = []
    repairable = sorted(
        (row for row in decisions.get("decisions", []) if row.get("recommendation") == "repair"),
        key=lambda row: (float(row.get("overall_quality", 0.0)), int(row.get("mapping_index", 0))),
    )
    selected = _strategy_diverse_selection(repairable, max(0, int(maximum_repairs)))
    neighborhoods = build_repair_neighborhoods(repaired, {"decisions": repairable})
    handled: set[str] = set()
    plans_by_key = {str(row["placement_key"]): repair_strategy_for(row) for row in selected}
    attempts_by_key = {
        str(decision["placement_key"]): {
            "placement_key": str(decision["placement_key"]),
            "mapping_index": int(decision.get("mapping_index", -1)),
            "original_quality": round(float(decision.get("overall_quality", 0.0) or 0.0), 4),
            "failure_categories": sorted({str(row.get("category")) for row in decision.get("failures", []) if row.get("category")}),
            "repair_strategy": plans_by_key[str(decision["placement_key"])]["strategy"],
            "strategy_plan": plans_by_key[str(decision["placement_key"])],
            "donor_pool_size": len(donor_candidates),
            "candidates_considered": 0,
            "candidate_rejection_reasons": {},
            "candidate_pre_render_scores": [],
            "selected_candidate_id": None,
            "candidate_family": None,
            "proposed": False,
            "no_viable_alternative": False,
        }
        for decision in selected
    }

    # Apply local, low-disruption strategies before donor replacement.
    for decision in selected:
        key = str(decision["placement_key"])
        index = int(decision.get("mapping_index", -1))
        if index < 0 or index >= len(mappings):
            continue
        plan = plans_by_key[key]
        if plan["strategy"] == "conservative_uncertainty_retention":
            attempts_by_key[key].update({
                "no_viable_alternative": True,
                "candidate_loss_stage": "conservative_retention",
            })
            handled.add(key)
            continue
        candidate = _local_strategy_candidate(mappings[index], decision, plan)
        if candidate is None:
            continue
        current = mappings[index]
        predicted = {
            "score": min(1.0, float(decision.get("overall_quality", 0.0) or 0.0) + candidate["predicted_gain"]),
            "editorial_score_model": candidate["score_model"],
            "editorial_components": candidate["components"],
        }
        attempt = attempts_by_key[key]
        _record_candidate(attempt, current, predicted, candidate["family"])
        mappings[index] = candidate["mapping"]
        strategy_decision = {**decision, "repair_strategy": plan["strategy"]}
        proposals.append(_proposal(
            decision=strategy_decision, index=index, current=current, replacement=candidate["mapping"],
            current_score=float(decision.get("overall_quality", 0.0) or 0.0), score_data=predicted,
        ))
        attempt.update({
            "selected_candidate_id": f"{current.get('clip_id')}:{candidate['family']}",
            "candidate_family": candidate["family"], "proposed": True,
        })
        handled.add(key)

    # Boundary-integrity defects are cheapest to repair in place.  A number of
    # schedules intentionally trim a complete donor to the nominal speech slot
    # even though both the donor file and destination slot still have room.
    # Preserve identity/performance evidence and expose the clipped tail before
    # attempting a donor reassignment.
    for decision in selected:
        index = int(decision.get("mapping_index", -1))
        if index < 0 or index >= len(mappings):
            continue
        current = mappings[index]
        donor = donors_by_id.get(str(current.get("clip_id") or ""))
        candidate = _source_boundary_extension(current, donor, decision)
        if candidate is None:
            continue
        key = str(decision["placement_key"])
        attempt = attempts_by_key[key]
        predicted = {
            "score": min(1.0, float(decision.get("overall_quality", 0.0) or 0.0) + candidate["predicted_gain"]),
            "editorial_score_model": "source_boundary_feasibility_v1",
            "editorial_components": {
                "boundary_extension_seconds": candidate["extension_seconds"],
                "slot_headroom_seconds": candidate["slot_headroom_seconds"],
            },
        }
        _record_candidate(attempt, donor or {}, predicted, "source_boundary_extension")
        replacement = candidate["mapping"]
        mappings[index] = replacement
        proposals.append(_proposal(
            decision={**decision, "repair_strategy": "extend_source_boundary"},
            index=index, current=current, replacement=replacement,
            current_score=float(decision.get("overall_quality", 0.0) or 0.0), score_data=predicted,
        ))
        attempt.update({
            "repair_strategy": "extend_source_boundary",
            "selected_candidate_id": f"{current.get('clip_id')}:boundary+{candidate['extension_seconds']:.3f}",
            "candidate_family": "source_boundary_extension", "proposed": True,
        })
        handled.add(key)

    # With a nearly exhausted donor library, treating every occupied clip as
    # unavailable leaves the repair search with only one or two seed choices.
    # Safe two-way assignments expose alternatives without changing reuse.
    pair_options = []
    for left_offset, left in enumerate(selected):
        if str(left["placement_key"]) in handled or _has_boundary_integrity_failure(left):
            continue
        left_index = int(left.get("mapping_index", -1))
        if left_index < 0 or left_index >= len(mappings):
            continue
        left_mapping = mappings[left_index]
        left_clip = donors_by_id.get(str(left_mapping.get("clip_id") or ""))
        if left_clip is None:
            continue
        left_window = _repair_window(left_mapping, left, schedule)
        left_current = float(score_candidate(left_window, left_clip).get("score", 0.0) or 0.0)
        for right in selected[left_offset + 1:]:
            if str(right["placement_key"]) in handled or _has_boundary_integrity_failure(right):
                continue
            right_index = int(right.get("mapping_index", -1))
            if right_index < 0 or right_index >= len(mappings):
                continue
            right_mapping = mappings[right_index]
            right_clip = donors_by_id.get(str(right_mapping.get("clip_id") or ""))
            if right_clip is None:
                continue
            if memory.rejected(left["placement_key"], str(right_clip.get("id"))) or memory.rejected(
                right["placement_key"], str(left_clip.get("id"))
            ):
                continue
            right_window = _repair_window(right_mapping, right, schedule)
            right_current = float(score_candidate(right_window, right_clip).get("score", 0.0) or 0.0)
            left_new = score_candidate(left_window, right_clip)
            right_new = score_candidate(right_window, left_clip)
            _record_candidate(attempts_by_key[str(left["placement_key"])], right_clip, left_new, "atomic_donor_swap")
            _record_candidate(attempts_by_key[str(right["placement_key"])], left_clip, right_new, "atomic_donor_swap")
            left_gain = float(left_new.get("score", 0.0) or 0.0) - left_current
            right_gain = float(right_new.get("score", 0.0) or 0.0) - right_current
            if min(left_gain, right_gain) < -0.0001 or max(left_gain, right_gain) < minimum_predicted_gain:
                continue
            pair_options.append((
                left_gain + right_gain, str(left["placement_key"]), str(right["placement_key"]),
                left, right, left_window, right_window, left_clip, right_clip,
                left_current, right_current, left_new, right_new,
            ))
    for option in sorted(pair_options, key=lambda row: (-row[0], row[1], row[2])):
        (
            _gain, left_key, right_key, left, right, left_window, right_window,
            left_clip, right_clip, left_current, right_current, left_new, right_new,
        ) = option
        if left_key in handled or right_key in handled:
            continue
        group_id = f"editorial_swap_{min(left_key, right_key)}_{max(left_key, right_key)}"
        for decision, window, old_clip, new_clip, old_score, new_score in (
            (left, left_window, left_clip, right_clip, left_current, left_new),
            (right, right_window, right_clip, left_clip, right_current, right_new),
        ):
            index = int(decision["mapping_index"])
            current = mappings[index]
            memory.remember(decision, clip_id=str(old_clip.get("id") or ""))
            replacement = _preserve_context(current, build_mapping(window, new_clip, new_score))
            mappings[index] = replacement
            proposals.append(_proposal(
                decision={**decision, "repair_strategy": plans_by_key[str(decision["placement_key"])]["strategy"]}, index=index, current=current, replacement=replacement,
                current_score=old_score, score_data=new_score, assignment_group_id=group_id,
            ))
            attempt = attempts_by_key[str(decision["placement_key"])]
            attempt.update({
                "selected_candidate_id": str(new_clip.get("id") or ""),
                "candidate_family": "atomic_donor_swap", "proposed": True,
                "assignment_group_id": group_id,
            })
        handled.update({left_key, right_key})

    for decision in selected:
        if str(decision["placement_key"]) in handled:
            continue
        index = int(decision["mapping_index"])
        if index < 0 or index >= len(mappings):
            continue
        current = mappings[index]
        memory.remember(decision, clip_id=str(current.get("clip_id") or ""))
        window = _repair_window(current, decision, schedule)
        ranked = []
        attempt = attempts_by_key[str(decision["placement_key"])]
        current_donor = donors_by_id.get(str(current.get("clip_id") or ""))
        current_score_data = score_candidate(window, current_donor) if current_donor is not None else None
        current_score = float(
            current_score_data.get("score", 0.0) if current_score_data is not None
            else current.get("editorial_candidate_score", current.get("score", 0.0)) or 0.0
        )
        attempt["current_pre_render_score"] = round(current_score, 4)
        for clip in donor_candidates:
            clip_id = str(clip.get("id") or "")
            if not clip_id or clip_id == str(current.get("clip_id")):
                _reject_candidate(attempt, "missing_or_current_donor")
                continue
            if clip_id in used:
                _reject_candidate(attempt, "donor_already_occupied")
                continue
            if memory.rejected(decision["placement_key"], clip_id):
                _reject_candidate(attempt, "known_failed_donor_placement_pair")
                continue
            score = score_candidate(window, clip)
            same_performance = bool(current.get("source_performance_id")) and (
                str(clip.get("source_performance_id")) == str(current.get("source_performance_id"))
            )
            family = "same_performance_reassignment" if same_performance else "alternative_donor"
            _record_candidate(attempt, clip, score, family)
            rejection = _preflight_rejection(decision, window, clip, score, current_score)
            if rejection:
                _reject_candidate(attempt, rejection)
                continue
            ranked.append((float(score.get("score", 0.0)), clip_id, clip, score))
        if not ranked:
            attempt["no_viable_alternative"] = True
            reasons = attempt.get("candidate_rejection_reasons", {})
            attempt["candidate_loss_stage"] = (
                "pre_render_quality_ceiling"
                if reasons.get("predicted_quality_regression")
                else "pre_render_hard_constraints"
                if int(attempt.get("candidates_considered", 0)) > 0
                else "candidate_generation"
            )
            continue
        _score, _clip_id, clip, score_data = max(ranked, key=lambda row: (row[0], row[1]))
        rendered_failure = any(
            row["category"] in {"incomplete_sentence", "mid_word_cut", "low_rendered_coverage", "masking", "confidence_collapse"}
            for row in decision.get("failures", [])
        )
        predicted_delta = float(score_data.get("score", 0.0)) - current_score
        if predicted_delta < -0.05:
            _reject_candidate(attempt, "predicted_quality_regression")
            attempt["no_viable_alternative"] = True
            attempt["candidate_loss_stage"] = "pre_render_quality_ceiling"
            continue
        if predicted_delta < minimum_predicted_gain:
            legacy_unstructured_score = not (current_score_data or {}).get("editorial_components") and not score_data.get("editorial_components")
            bounded_exploration = _bounded_rendered_failure_exploration(
                decision, score_data, predicted_delta=predicted_delta,
            )
            if not rendered_failure or not (
                legacy_unstructured_score
                or _targeted_component_improved(decision, current_score_data or {}, score_data)
                or bounded_exploration
            ):
                _reject_candidate(attempt, "predicted_ceiling_below_required_delta")
                attempt["no_viable_alternative"] = True
                attempt["candidate_loss_stage"] = "pre_render_quality_ceiling"
                continue
            if bounded_exploration:
                attempt["bounded_rendered_exploration"] = True
                attempt["exploration_reason"] = "rendered_boundary_evidence_disagrees_with_pre_render_ceiling"
        replacement = _preserve_context(current, build_mapping(window, clip, score_data))
        mappings[index] = replacement
        used.discard(str(current.get("clip_id")))
        used.add(str(replacement.get("clip_id")))
        proposals.append(_proposal(
            decision={**decision, "repair_strategy": plans_by_key[str(decision["placement_key"])]["strategy"]}, index=index, current=current, replacement=replacement,
            current_score=current_score, score_data=score_data,
        ))
        attempt.update({
            "selected_candidate_id": str(clip.get("id") or ""),
            "candidate_family": (
                "same_performance_reassignment"
                if bool(current.get("source_performance_id")) and str(clip.get("source_performance_id")) == str(current.get("source_performance_id"))
                else "alternative_donor"
            ),
            "proposed": True,
        })
    for attempt in attempts_by_key.values():
        attempt["candidate_pre_render_scores"] = sorted(
            attempt["candidate_pre_render_scores"],
            key=lambda row: (-float(row.get("score", 0.0)), str(row.get("candidate_id"))),
        )[:12]
        if not attempt["proposed"]:
            attempt["no_viable_alternative"] = True
            attempt.setdefault("candidate_loss_stage", "candidate_generation")
    coordinated = _attach_repair_neighborhoods(proposals, attempts_by_key, neighborhoods)
    return {
        "schedule": repaired,
        "repairs": proposals,
        "regions": _merge_regions([row["region"] for row in proposals]),
        "attempted_count": min(len(repairable), max(0, int(maximum_repairs))),
        "repaired_count": len(proposals),
        "attempts": list(attempts_by_key.values()),
        "repair_neighborhoods": neighborhoods,
        "coordinated_neighborhood_count": coordinated,
    }


def _record_candidate(
    attempt: dict[str, Any], clip: dict[str, Any], score_data: dict[str, Any], family: str,
) -> None:
    attempt["candidates_considered"] = int(attempt.get("candidates_considered", 0)) + 1
    attempt["candidate_pre_render_scores"].append({
        "candidate_id": str(clip.get("id") or clip.get("clip_id") or "local_mapping"),
        "candidate_family": family,
        "score": round(float(score_data.get("score", 0.0) or 0.0), 4),
        "score_model": score_data.get("editorial_score_model"),
        "components": dict(score_data.get("editorial_components", {})),
    })


def _strategy_diverse_selection(repairable: list[dict[str, Any]], maximum: int) -> list[dict[str, Any]]:
    """Reserve one repair slot per observed strategy, then fill by severity/quality."""
    if maximum <= 0:
        return []
    selected: list[dict[str, Any]] = []
    selected_keys: set[str] = set()
    by_strategy: dict[str, list[dict[str, Any]]] = {}
    for decision in repairable:
        strategy = str(repair_strategy_for(decision).get("strategy"))
        by_strategy.setdefault(strategy, []).append(decision)
    strategy_order = sorted(
        by_strategy,
        key=lambda value: (
            min(float(row.get("overall_quality", 0.0) or 0.0) for row in by_strategy[value]),
            value,
        ),
    )
    for strategy in strategy_order:
        decision = by_strategy[strategy][0]
        selected.append(decision)
        selected_keys.add(str(decision["placement_key"]))
        if len(selected) >= maximum:
            return selected
    for decision in repairable:
        key = str(decision["placement_key"])
        if key not in selected_keys:
            selected.append(decision)
            selected_keys.add(key)
        if len(selected) >= maximum:
            break
    return selected


def build_repair_neighborhoods(
    schedule: dict[str, Any], decisions: dict[str, Any], *, maximum_size: int = 6,
) -> list[dict[str, Any]]:
    """Find adjacent failed placements that share editorial context."""
    mappings = schedule.get("mappings", [])
    eligible_categories = {
        "speaker_mismatch", "performance_mismatch", "transition_artifact",
        "residual_dialogue", "masking",
    }
    eligible_strategies = {
        "repair_speaker_role", "repair_performance_structure", "repair_transition_edges",
        "repair_local_suppression", "repair_audio_masking",
    }
    rows = []
    for decision in decisions.get("decisions", []):
        index = int(decision.get("mapping_index", -1))
        categories = {str(row.get("category")) for row in decision.get("failures", [])}
        primary_strategy = str(repair_strategy_for(decision).get("strategy"))
        if (
            index < 0 or index >= len(mappings)
            or primary_strategy not in eligible_strategies
            or not categories.intersection(eligible_categories)
        ):
            continue
        mapping = mappings[index]
        start = float(mapping.get("destination_timestamp", 0.0) or 0.0)
        end = start + float(mapping.get("planned_render_duration", 0.0) or 0.0)
        rows.append({
            "decision": decision, "mapping": mapping, "index": index,
            "start": start, "end": end, "categories": categories,
        })
    rows.sort(key=lambda row: (row["start"], row["index"]))
    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for row in rows:
        if not current:
            current = [row]
            continue
        prior = current[-1]
        same_performance = bool(row["mapping"].get("destination_performance_id")) and (
            str(row["mapping"].get("destination_performance_id"))
            == str(prior["mapping"].get("destination_performance_id"))
        )
        adjacent = row["start"] <= prior["end"] + 0.75
        contiguous_indices = row["index"] <= prior["index"] + 2
        if (adjacent or (same_performance and contiguous_indices)) and len(current) < maximum_size:
            current.append(row)
        else:
            if len(current) >= 2:
                groups.append(current)
            current = [row]
    if len(current) >= 2:
        groups.append(current)
    result = []
    for index, group in enumerate(groups, start=1):
        shared_performance = len({str(row["mapping"].get("destination_performance_id") or "") for row in group}) == 1
        result.append({
            "neighborhood_id": f"repair_neighborhood_{index:04d}",
            "placement_keys": [str(row["decision"]["placement_key"]) for row in group],
            "mapping_indices": [row["index"] for row in group],
            "start": round(min(row["start"] for row in group), 3),
            "end": round(max(row["end"] for row in group), 3),
            "failure_categories": sorted(set().union(*(row["categories"] for row in group))),
            "coordination_reasons": [
                value for value, present in (
                    ("shared_destination_performance", shared_performance),
                    ("adjacent_timing_pressure", True),
                ) if present
            ],
            "commit_policy": "atomic_neighborhood_non_regression",
        })
    return result


def _attach_repair_neighborhoods(
    proposals: list[dict[str, Any]], attempts: dict[str, dict[str, Any]],
    neighborhoods: list[dict[str, Any]],
) -> int:
    coordinated = 0
    by_key = {str(row.get("placement_key")): row for row in proposals}
    for neighborhood in neighborhoods:
        members = [by_key[key] for key in neighborhood["placement_keys"] if key in by_key]
        if len(members) < 2:
            continue
        group_id = str(neighborhood["neighborhood_id"])
        for proposal in members:
            proposal.setdefault("assignment_group_id", group_id)
            proposal["repair_neighborhood_id"] = group_id
            proposal["coordination_mode"] = "coordinated_neighborhood"
            attempt = attempts.get(str(proposal.get("placement_key")))
            if attempt is not None:
                attempt["repair_neighborhood_id"] = group_id
                attempt["coordinated_candidate"] = True
        coordinated += 1
    return coordinated


def _source_boundary_extension(
    mapping: dict[str, Any], donor: dict[str, Any] | None, decision: dict[str, Any],
    *, minimum_extension: float = 0.08,
) -> dict[str, Any] | None:
    categories = {str(row.get("category")) for row in decision.get("failures", [])}
    target = str(decision.get("target_failure_category") or "")
    if target and target not in {"incomplete_sentence", "mid_word_cut", "low_rendered_coverage"}:
        return None
    if not categories.intersection({"incomplete_sentence", "mid_word_cut", "low_rendered_coverage"}):
        return None
    if donor is None:
        return None
    if mapping.get("clip_trim_duration") is None:
        return None
    trim_start = max(0.0, float(mapping.get("clip_trim_start", 0.0) or 0.0))
    trim_duration = max(0.0, float(mapping.get("clip_trim_duration", 0.0) or 0.0))
    donor_duration = max(0.0, float(donor.get("duration", 0.0) or 0.0))
    stretch = max(0.001, float(mapping.get("stretch_factor", 1.0) or 1.0))
    slot_start = float(mapping.get("alignment_slot_start", mapping.get("destination_timestamp", 0.0)) or 0.0)
    slot_end = float(mapping.get(
        "alignment_slot_end",
        float(mapping.get("destination_timestamp", 0.0) or 0.0) + float(mapping.get("planned_render_duration", 0.0) or 0.0),
    ) or 0.0)
    slot_source_capacity = max(0.0, (slot_end - slot_start) / stretch)
    maximum_trim = min(max(0.0, donor_duration - trim_start), slot_source_capacity)
    extension = maximum_trim - trim_duration
    if extension < minimum_extension:
        return None
    replacement = deepcopy(mapping)
    replacement["clip_trim_duration"] = round(maximum_trim, 3)
    replacement["planned_render_duration"] = round(maximum_trim * stretch, 3)
    replacement["fade_duration"] = min(0.008, max(0.002, extension / 8.0))
    replacement["editorial_repair_generation"] = int(mapping.get("editorial_repair_generation", 0) or 0) + 1
    replacement["editorial_boundary_extension_seconds"] = round(extension, 3)
    replacement["editorial_candidate_family"] = "source_boundary_extension"
    return {
        "mapping": replacement,
        "extension_seconds": round(extension, 3),
        "slot_headroom_seconds": round(max(0.0, slot_end - slot_start - float(mapping.get("planned_render_duration", 0.0) or 0.0)), 3),
        "predicted_gain": round(min(0.16, 0.04 + 0.12 * extension / max(trim_duration, 0.25)), 4),
    }


def _local_strategy_candidate(
    mapping: dict[str, Any], decision: dict[str, Any], plan: dict[str, Any],
) -> dict[str, Any] | None:
    strategy = str(plan.get("strategy"))
    replacement = deepcopy(mapping)
    if strategy == "repair_local_suppression":
        previous = float(mapping.get("suppression_padding", 0.0) or 0.0)
        replacement.update({
            "suppression_padding": round(min(0.3, max(0.12, previous + 0.08)), 3),
            "suppression_leading_padding": round(min(0.24, max(0.1, previous + 0.06)), 3),
            "suppression_trailing_padding": round(min(0.35, max(0.16, previous + 0.12)), 3),
            "editorial_candidate_family": "suppression_expansion",
        })
        return _local_candidate(replacement, "suppression_expansion", 0.08, {
            "suppression_padding": replacement["suppression_padding"],
        })
    if strategy == "repair_audio_masking":
        gain = min(6.0, float(mapping.get("gain_db", 0.0) or 0.0) + 2.0)
        replacement.update({
            "gain_db": round(gain, 2), "highpass_hz": max(80.0, float(mapping.get("highpass_hz", 0.0) or 0.0)),
            "fade_duration": 0.006, "editorial_candidate_family": "audio_treatment",
        })
        return _local_candidate(replacement, "audio_treatment", 0.055, {"gain_db": gain, "fade_duration": 0.006})
    if strategy == "repair_transition_edges":
        start = float(mapping.get("destination_timestamp", 0.0) or 0.0)
        duration = float(mapping.get("planned_render_duration", 0.0) or 0.0)
        slot_start = float(mapping.get("alignment_slot_start", start) or start)
        shot_end = mapping.get("shot_end")
        shift = 0.0
        if shot_end is not None and start + duration > float(shot_end):
            shift = min(0.12, start + duration - float(shot_end), max(0.0, start - slot_start))
            replacement["destination_timestamp"] = round(start - shift, 3)
        replacement.update({"fade_duration": 0.035, "editorial_candidate_family": "audio_edge_adjustment"})
        return _local_candidate(replacement, "audio_edge_adjustment", 0.045, {"temporal_shift": round(-shift, 3), "fade_duration": 0.035})
    if strategy == "repair_duration_fit" and mapping.get("clip_trim_duration") is not None:
        stretch = float(mapping.get("stretch_factor", 1.0) or 1.0)
        if abs(stretch - 1.0) < 0.025:
            return None
        trim = float(mapping.get("clip_trim_duration", 0.0) or 0.0)
        start = float(mapping.get("destination_timestamp", 0.0) or 0.0)
        slot_end = float(mapping.get("alignment_slot_end", start + float(mapping.get("planned_render_duration", 0.0) or 0.0)) or 0.0)
        target = 1.0 + (stretch - 1.0) * 0.5
        planned = trim * target
        if planned > slot_end - start + 0.001:
            return None
        replacement.update({
            "stretch_factor": round(target, 4), "planned_render_duration": round(planned, 3),
            "editorial_candidate_family": "time_adaptation",
        })
        return _local_candidate(replacement, "time_adaptation", 0.05, {"old_stretch": stretch, "new_stretch": target})
    return None


def _local_candidate(mapping: dict[str, Any], family: str, gain: float, components: dict[str, Any]) -> dict[str, Any]:
    mapping["editorial_repair_generation"] = int(mapping.get("editorial_repair_generation", 0) or 0) + 1
    return {
        "mapping": mapping, "family": family, "predicted_gain": gain,
        "score_model": f"{family}_feasibility_v1", "components": components,
    }


def _preflight_rejection(
    decision: dict[str, Any], window: dict[str, Any], clip: dict[str, Any],
    score: dict[str, Any], current_score: float,
) -> str | None:
    primary = str(repair_strategy_for(decision).get("failure_category"))
    components = score.get("editorial_components", {})
    duration = float(clip.get("duration", 0.0) or 0.0)
    target = float(window.get("duration", 0.0) or 0.0)
    if primary == "duration_failure" and target > 0.0 and not 0.7 <= duration / target <= 1.3:
        return "impossible_duration_fit"
    if primary == "speaker_mismatch" and float(components.get("speaker_role_fit", 0.0) or 0.0) < 0.55:
        return "strong_speaker_mismatch"
    if primary == "performance_mismatch" and float(components.get("performance_fit", 0.0) or 0.0) < 0.42:
        return "incompatible_performance_structure"
    if primary == "visual_mismatch" and float(components.get("visual_fit", 0.0) or 0.0) < 0.3:
        return "poor_visual_intent_compatibility"
    if primary in {"incomplete_sentence", "mid_word_cut"}:
        transcript = str(clip.get("transcript", "")).strip()
        if transcript and (len(transcript.split()) < 2 or transcript[-1] not in ".!?"):
            return "insufficient_sentence_boundary_evidence"
    if float(score.get("score", 0.0) or 0.0) < current_score - 0.05:
        return "predicted_quality_regression"
    return None


def _has_boundary_integrity_failure(decision: dict[str, Any]) -> bool:
    target = str(decision.get("target_failure_category") or "")
    if target:
        return target in {
            "incomplete_sentence", "mid_word_cut", "low_rendered_coverage",
            "masking", "confidence_collapse",
        }
    return bool({
        str(row.get("category")) for row in decision.get("failures", [])
    }.intersection({"incomplete_sentence", "mid_word_cut", "low_rendered_coverage", "masking", "confidence_collapse"}))


def _reject_candidate(attempt: dict[str, Any], reason: str) -> None:
    reasons = attempt["candidate_rejection_reasons"]
    reasons[reason] = int(reasons.get(reason, 0)) + 1


def _repair_window(
    mapping: dict[str, Any], decision: dict[str, Any], schedule: dict[str, Any],
) -> dict[str, Any]:
    window = _window_from_mapping(mapping)
    window["editorial_failure_categories"] = [
        str(row.get("category")) for row in decision.get("failures", []) if row.get("category")
    ]
    window["editorial_repair_strategy"] = decision.get("repair_strategy")
    window["active_filter"] = mapping.get("active_filter", schedule.get("active_filter", "balanced"))
    return window


def _proposal(
    *, decision: dict[str, Any], index: int, current: dict[str, Any], replacement: dict[str, Any],
    current_score: float, score_data: dict[str, Any], assignment_group_id: str | None = None,
) -> dict[str, Any]:
    row = {
        "placement_key": decision["placement_key"],
        "mapping_index": index,
        "old_clip_id": current.get("clip_id"),
        "new_clip_id": replacement.get("clip_id"),
        "old_predicted_score": round(current_score, 4),
        "new_predicted_score": round(float(score_data.get("score", 0.0)), 4),
        "score_model": score_data.get("editorial_score_model"),
        "score_components": dict(score_data.get("editorial_components", {})),
        "repair_strategy": decision.get("repair_strategy"),
        "region": _replacement_region(current, replacement),
    }
    if assignment_group_id:
        row["assignment_group_id"] = assignment_group_id
    return row


def _targeted_component_improved(
    decision: dict[str, Any], current: dict[str, Any], candidate: dict[str, Any], *, minimum_gain: float = 0.03,
) -> bool:
    before = current.get("editorial_components", {})
    after = candidate.get("editorial_components", {})
    categories = {str(row.get("category")) for row in decision.get("failures", [])}
    component_by_failure = {
        "incomplete_sentence": "sentence_fit",
        "mid_word_cut": "sentence_fit",
        "low_rendered_coverage": "sentence_fit",
        "confidence_collapse": "confidence",
        "speaker_mismatch": "speaker_role_fit",
        "performance_mismatch": "performance_fit",
        "visual_mismatch": "visual_fit",
        "duration_failure": "timing_and_render_fit",
    }
    for category in categories:
        component = component_by_failure.get(category)
        if component and float(after.get(component, 0.0) or 0.0) >= float(before.get(component, 0.0) or 0.0) + minimum_gain:
            return True
    return False


def _bounded_rendered_failure_exploration(
    decision: dict[str, Any], candidate: dict[str, Any], *, predicted_delta: float,
) -> bool:
    """Admit one credible challenger when rendered evidence disproves the pre-render ceiling.

    The pass manager still renders, verifies, and rolls the proposal back unless measured
    quality improves. This only prevents an optimistic score for the current donor from
    suppressing every empirical repair attempt.
    """
    failures = {
        str(row.get("category")): str(row.get("severity") or "")
        for row in decision.get("failures", []) if row.get("category")
    }
    boundary_failure = any(
        failures.get(category) in {"high", "critical"}
        for category in ("incomplete_sentence", "low_rendered_coverage", "mid_word_cut")
    )
    components = dict(candidate.get("editorial_components") or {})
    return bool(
        boundary_failure
        and predicted_delta >= -0.03
        and float(candidate.get("score", 0.0) or 0.0) >= 0.65
        and float(components.get("sentence_fit", 0.0) or 0.0) >= 0.9
        and float(components.get("timing_and_render_fit", 0.0) or 0.0) >= 0.55
    )


def _replacement_region(current: dict[str, Any], replacement: dict[str, Any]) -> dict[str, float]:
    old_start = float(current.get("destination_timestamp", 0.0) or 0.0)
    new_start = float(replacement.get("destination_timestamp", old_start) or old_start)
    old_end = old_start + float(current.get("planned_render_duration", 0.0) or 0.0)
    new_end = new_start + float(replacement.get("planned_render_duration", 0.0) or 0.0)
    return {"start": round(min(old_start, new_start), 3), "end": round(max(old_end, new_end), 3)}


def _window_from_mapping(mapping: dict[str, Any]) -> dict[str, Any]:
    start = float(mapping.get("destination_timestamp", 0.0) or 0.0)
    slot_start, slot_end = mapping.get("alignment_slot_start"), mapping.get("alignment_slot_end")
    duration = (
        float(slot_end) - float(slot_start)
        if slot_start is not None and slot_end is not None and float(slot_end) > float(slot_start)
        else float(mapping.get("planned_render_duration", 0.0) or 0.0)
    )
    return {
        **mapping,
        "id": mapping.get("window_id") or mapping.get("id"),
        "start": start,
        "end": start + duration,
        "duration": max(0.001, duration),
        "signature": mapping.get("destination_performance_signature") or {},
        "speaker_sequence": mapping.get("destination_speaker_sequence") or [],
        "turn_pattern": mapping.get("destination_turn_pattern") or "",
        "speaker_id": mapping.get("destination_speaker_id"),
    }


def _preserve_context(current: dict[str, Any], replacement: dict[str, Any]) -> dict[str, Any]:
    preserved = dict(replacement)
    for key, value in current.items():
        if key.startswith("montage_") or key in {
            "destination_performance_id", "performance_id", "performance_type",
            "performance_dialogue_density", "performance_visible_windows", "performance_shot_count",
            "alignment_source_window_ids", "alignment_source_kind", "alignment_slot_start",
            "alignment_slot_end", "alignment_mode", "review_label",
            "editorial_placement_id",
        }:
            preserved[key] = value
    preserved["editorial_repair_generation"] = int(current.get("editorial_repair_generation", 0) or 0) + 1
    preserved["editorial_replaced_clip_id"] = current.get("clip_id")
    return preserved


def _merge_regions(regions: list[dict[str, float]], gap: float = 0.08) -> list[dict[str, float]]:
    ordered = sorted((dict(row) for row in regions), key=lambda row: row["start"])
    merged: list[dict[str, float]] = []
    for row in ordered:
        if merged and row["start"] <= merged[-1]["end"] + gap:
            merged[-1]["end"] = max(merged[-1]["end"], row["end"])
        else:
            merged.append(row)
    return [
        {"id": f"editorial_region_{index:06d}", "start": round(row["start"], 3), "end": round(row["end"], 3)}
        for index, row in enumerate(merged, start=1)
    ]
