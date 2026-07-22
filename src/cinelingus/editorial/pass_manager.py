from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable

from .decision_engine import evaluate_editorial_decisions
from .editorial_memory import EditorialMemory
from .repair_strategies import STRATEGIES, repair_strategy_for
from ..render_verification import merge_rendered_dialogue_verification


class EditorialPassManager:
    """Coordinate bounded repair passes and reject any quality regression."""

    def __init__(
        self, *, maximum_passes: int = 2, acceptance_threshold: float = 0.72,
        minimum_word_coverage: float = 0.72, max_time_stretch: float = 0.1,
        suppress_unresolved: bool = False,
        benchmark_target_failure_category: str | None = None,
    ) -> None:
        self.maximum_passes = max(0, int(maximum_passes))
        self.acceptance_threshold = float(acceptance_threshold)
        self.minimum_word_coverage = float(minimum_word_coverage)
        self.max_time_stretch = float(max_time_stretch)
        self.memory = EditorialMemory()
        self.problem_report: dict[str, Any] = {}
        self.suppress_unresolved = bool(suppress_unresolved)
        if benchmark_target_failure_category is not None and benchmark_target_failure_category not in STRATEGIES:
            raise ValueError(f"Unknown benchmark target failure category: {benchmark_target_failure_category}")
        self.benchmark_target_failure_category = benchmark_target_failure_category
        self.invalidated_problem_placements: set[str] = set()

    def decide(
        self, schedule: dict[str, Any], verification: dict[str, Any], residue: dict[str, Any] | None,
        *, additional_problem_invalidations: set[str] | None = None,
    ) -> dict[str, Any]:
        decisions = evaluate_editorial_decisions(
            schedule=schedule, rendered_verification=verification, residue_verification=residue,
            acceptance_threshold=self.acceptance_threshold,
            minimum_word_coverage=self.minimum_word_coverage,
            max_time_stretch=self.max_time_stretch,
            problem_report=self.problem_report,
            ignore_problem_placement_keys=(
                self.invalidated_problem_placements | set(additional_problem_invalidations or set())
            ),
        )
        if self.benchmark_target_failure_category:
            decisions = _focus_strategy_benchmark_decisions(
                decisions, self.benchmark_target_failure_category
            )
        return decisions

    def run(
        self, *, schedule: dict[str, Any], verification: dict[str, Any],
        residue: dict[str, Any] | None,
        problem_report: dict[str, Any] | None = None,
        repair_callback: Callable[[dict[str, Any], dict[str, Any], EditorialMemory, int], dict[str, Any]],
        render_verify_callback: Callable[
            [dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], int, dict[str, Any], dict[str, Any] | None],
            tuple[dict[str, Any], dict[str, Any] | None],
        ],
        rollback_callback: Callable[[dict[str, Any], list[dict[str, Any]], int], None] | None = None,
        reject_callback: Callable[[dict[str, Any], list[dict[str, Any]]], None] | None = None,
        progress_callback: Callable[[str, dict[str, Any]], None] | None = None,
        resume_state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        resumed = bool(resume_state and resume_state.get("stage") == "candidate_prepared")
        if resumed:
            state = dict(resume_state or {})
            self.problem_report = dict(state.get("problem_report") or {})
            self.invalidated_problem_placements = set(state.get("invalidated_problem_placements") or [])
            self.memory = EditorialMemory.from_dict(state.get("memory"))
            current_schedule = deepcopy(state["current_schedule"])
            current_verification = dict(state["current_verification"])
            current_residue = dict(state.get("current_residue") or {})
            decisions = deepcopy(state["decisions"])
            passes = deepcopy(state["passes"])
            repaired_keys = set(state.get("repaired_keys") or [])
            first_pass = int(state["pass_index"])
        else:
            self.problem_report = dict(problem_report or {})
            self.invalidated_problem_placements = set()
            current_schedule = deepcopy(schedule)
            current_verification = dict(verification)
            current_residue = dict(residue or {})
            decisions = self.decide(current_schedule, current_verification, current_residue)
            passes = [_pass_record(0, "initial", decisions, [], accepted=True)]
            repaired_keys: set[str] = set()
            first_pass = 1
        termination = "quality_threshold_reached" if decisions["quality_gate_passed"] else "maximum_passes_exceeded"
        for pass_index in range(first_pass, self.maximum_passes + 1):
            if decisions["quality_gate_passed"]:
                termination = "quality_threshold_reached"
                break
            previous_schedule = deepcopy(current_schedule)
            previous_verification = current_verification
            previous_residue = current_residue
            previous_decisions = decisions
            if resumed and pass_index == first_pass:
                batch = deepcopy(dict(resume_state or {})["batch"])
                resumed = False
            else:
                batch = repair_callback(current_schedule, decisions, self.memory, pass_index)
            if not batch.get("repairs"):
                termination = "no_viable_alternatives"
                passes.append(_pass_record(
                    pass_index, "repair", decisions, [], accepted=False,
                    attempts=_repair_attempt_outcomes(batch.get("attempts", []), [], decisions, set()),
                ))
                break
            if progress_callback is not None:
                progress_callback("candidate_prepared", {
                    "stage": "candidate_prepared",
                    "pass_index": pass_index,
                    "current_schedule": current_schedule,
                    "current_verification": current_verification,
                    "current_residue": current_residue,
                    "decisions": decisions,
                    "passes": passes,
                    "repaired_keys": sorted(repaired_keys),
                    "invalidated_problem_placements": sorted(self.invalidated_problem_placements),
                    "problem_report": self.problem_report,
                    "memory": self.memory.to_dict(),
                    "batch": batch,
                })
            candidate_schedule = batch["schedule"]
            candidate_verification, candidate_residue = render_verify_callback(
                candidate_schedule,
                batch.get("regions", []),
                batch.get("repairs", []),
                pass_index,
                current_verification,
                current_residue,
            )
            proposed_keys = {str(row.get("placement_key")) for row in batch["repairs"]}
            candidate_decisions = self.decide(
                candidate_schedule, candidate_verification, candidate_residue,
                additional_problem_invalidations=proposed_keys,
            )
            evaluated_candidate_decisions = candidate_decisions
            improved = _is_non_decreasing(previous_decisions, candidate_decisions)
            accepted_repairs = list(batch["repairs"])
            rejected_repairs: list[dict[str, Any]] = []
            if not improved:
                selective = _select_non_regressing_repair_groups(
                    previous_schedule=previous_schedule,
                    candidate_schedule=candidate_schedule,
                    previous_verification=previous_verification,
                    candidate_verification=candidate_verification,
                    previous_residue=previous_residue,
                    candidate_residue=candidate_residue or {},
                    previous_decisions=previous_decisions,
                    candidate_decisions=candidate_decisions,
                    repairs=batch["repairs"],
                )
                if selective["accepted_repairs"]:
                    selected_decisions = self.decide(
                        selective["schedule"], selective["verification"], selective["residue"],
                        additional_problem_invalidations={
                            str(row.get("placement_key")) for row in selective["accepted_repairs"]
                        },
                    )
                    if _is_non_decreasing(previous_decisions, selected_decisions):
                        candidate_schedule = selective["schedule"]
                        candidate_verification = selective["verification"]
                        candidate_residue = selective["residue"]
                        candidate_decisions = selected_decisions
                        accepted_repairs = selective["accepted_repairs"]
                        rejected_repairs = selective["rejected_repairs"]
                        improved = True
                        if rollback_callback is not None and selective["rejected_regions"]:
                            rollback_callback(previous_schedule, selective["rejected_regions"], pass_index)
            if improved:
                current_schedule = candidate_schedule
                current_verification = candidate_verification
                current_residue = dict(candidate_residue or {})
                decisions = candidate_decisions
                repaired_keys.update(str(row["placement_key"]) for row in accepted_repairs)
                self.invalidated_problem_placements.update(
                    str(row["placement_key"]) for row in accepted_repairs
                )
                for repair in rejected_repairs:
                    failed = next(
                        (row for row in candidate_decisions["decisions"] if row["placement_key"] == repair["placement_key"]),
                        None,
                    )
                    if failed:
                        self.memory.remember(failed, clip_id=str(repair.get("new_clip_id") or ""))
            else:
                for repair in batch["repairs"]:
                    failed = next(
                        (row for row in candidate_decisions["decisions"] if row["placement_key"] == repair["placement_key"]),
                        None,
                    )
                    if failed:
                        self.memory.remember(failed, clip_id=str(repair.get("new_clip_id") or ""))
                if rollback_callback is not None:
                    rollback_callback(previous_schedule, batch.get("regions", []), pass_index)
                current_schedule = previous_schedule
                current_verification = previous_verification
                current_residue = previous_residue
                decisions = previous_decisions
            passes.append(_pass_record(
                pass_index, "repair", candidate_decisions, accepted_repairs if improved else batch["repairs"],
                accepted=improved, rejected_repairs=rejected_repairs,
                attempts=_repair_attempt_outcomes(
                    batch.get("attempts", []), batch.get("repairs", []), evaluated_candidate_decisions,
                    {str(row.get("placement_key")) for row in accepted_repairs} if improved else set(),
                ),
            ))
            if decisions["quality_gate_passed"]:
                termination = "quality_threshold_reached"
                break
        rejected_regions = []
        skipped_unrepairable_keys = {
            str(attempt.get("placement_key"))
            for pass_row in passes
            for attempt in pass_row.get("repair_attempts", [])
            if attempt.get("candidate_loss_stage") == "conservative_retention"
        }
        unresolved = [row for row in decisions["decisions"] if row.get("recommendation") != "accept"]
        unresolved_keys = {str(row.get("placement_key")) for row in unresolved}
        for decision in decisions["decisions"]:
            key = str(decision.get("placement_key"))
            if key not in unresolved_keys:
                decision["final_state"] = "IMPROVED_ACCEPTED" if key in repaired_keys else "ACCEPTED"
        if unresolved:
            for decision in unresolved:
                key = str(decision.get("placement_key"))
                index = int(decision.get("mapping_index", -1))
                if index < 0 or index >= len(current_schedule.get("mappings", [])):
                    continue
                mapping = current_schedule["mappings"][index]
                mapping["editorial_rejection"] = {
                    "reason": decision.get("repair_strategy") or "quality_threshold_not_met",
                    "quality": decision.get("overall_quality"),
                    "failures": list(decision.get("failures", [])),
                }
                if self.suppress_unresolved:
                    mapping["enabled"] = False
                    decision["final_action"] = "suppressed_after_bounded_repair"
                    decision["final_state"] = "SUPPRESSED"
                else:
                    mapping["enabled"] = True
                    if key in skipped_unrepairable_keys:
                        decision["final_action"] = "skipped_unrepairable_due_to_uncertain_evidence"
                        decision["final_state"] = "SKIPPED_UNREPAIRABLE"
                    else:
                        decision["final_action"] = "retained_best_known_after_bounded_repair"
                        decision["final_state"] = "BEST_KNOWN_UNRESOLVED"
                rejected_regions.append({
                    "start": float(mapping.get("destination_timestamp", 0.0) or 0.0),
                    "end": float(decision.get("destination_end", 0.0) or 0.0),
                })
                decision["recommendation"] = "reject"
            rejected_regions = _merge_regions(rejected_regions)
            if self.suppress_unresolved and reject_callback is not None and rejected_regions:
                reject_callback(current_schedule, rejected_regions)
            decisions["repair_count"] = 0
            decisions["rejected_count"] = len(unresolved)
            decisions["quality_gate_passed"] = False
        final_state_counts: dict[str, int] = {}
        for decision in decisions["decisions"]:
            state = str(decision.get("final_state", "FAILED_DELIVERY"))
            final_state_counts[state] = final_state_counts.get(state, 0) + 1
        return {
            "schema_version": "1.0",
            "editorial_system_version": "reflective_rendering_v1",
            "status": "PASS" if decisions["quality_gate_passed"] else "LIMIT_REACHED",
            "termination_reason": termination,
            "maximum_passes": self.maximum_passes,
            "completed_repair_passes": max(0, len(passes) - 1),
            "initial_quality": passes[0]["average_quality"],
            "final_quality": decisions["average_quality"],
            "quality_improvement": round(decisions["average_quality"] - passes[0]["average_quality"], 4),
            "placements_accepted_immediately": passes[0]["accepted_count"],
            "placements_repaired": len(repaired_keys - unresolved_keys),
            "placements_rejected": len(unresolved),
            "rejected_regions": rejected_regions,
            "unresolved_output_policy": "suppress" if self.suppress_unresolved else "retain_best_known",
            "final_state_counts": dict(sorted(final_state_counts.items())),
            "repair_capabilities": _repair_capabilities(current_schedule),
            "final_decisions": decisions,
            "memory": self.memory.to_dict(),
            "resumed_from_candidate_checkpoint": bool(resume_state),
            "passes": passes,
            "schedule": current_schedule,
            "verification": current_verification,
            "residue": current_residue,
        }


def _repair_capabilities(schedule: dict[str, Any]) -> dict[str, Any]:
    mode = str(schedule.get("scheduling_mode", "unknown"))
    return {
        "scheduling_mode": mode,
        "candidate_families": [
            "source_boundary_extension", "destination_boundary_adjustment", "timing_shift",
            "audio_edge_adjustment", "alternative_donor", "atomic_donor_swap",
            "coordinated_neighborhood", "retain_best_known", "suppression",
        ],
        "implemented_families": ["source_boundary_extension", "alternative_donor", "atomic_donor_swap", "retain_best_known", "suppression"],
        "interruption_recovery": {
            "granularity": "prepared_candidate",
            "donor_selection_replayed": False,
            "render_replayed_atomically": True,
            "avoidance_memory_restored": True,
        },
        "unsupported_families_are_reported": True,
        "capability_version": "editorial_repair_capabilities_v1",
    }


def _focus_strategy_benchmark_decisions(
    decisions: dict[str, Any], target_category: str,
) -> dict[str, Any]:
    """Route only observed target failures during an explicit calibration benchmark."""
    focused = deepcopy(decisions)
    target_count = 0
    for row in focused.get("decisions", []):
        categories = {str(item.get("category")) for item in row.get("failures", [])}
        row["benchmark_original_recommendation"] = row.get("recommendation")
        row["benchmark_target_failure_category"] = target_category
        if target_category in categories:
            row["target_failure_category"] = target_category
            plan = repair_strategy_for(row)
            row["repair_strategy"] = plan["strategy"]
            row["repair_plan"] = plan
            row["recommendation"] = "repair"
            target_count += 1
        else:
            # Other evidence remains visible and still participates in quality
            # comparison, but it cannot consume this strategy-isolation pass.
            row["recommendation"] = "accept"
    focused["benchmark_mode"] = "strategy_isolation"
    focused["benchmark_target_failure_category"] = target_category
    focused["benchmark_target_placement_count"] = target_count
    focused["accepted_count"] = len(focused.get("decisions", [])) - target_count
    focused["repair_count"] = target_count
    focused["rejected_count"] = 0
    focused["quality_gate_passed"] = bool(focused.get("decisions")) and target_count == 0
    return focused


def _is_non_decreasing(before: dict[str, Any], after: dict[str, Any]) -> bool:
    epsilon = 0.0001
    before_by_key = {str(row.get("placement_key")): row for row in before.get("decisions", [])}
    no_placement_regressed = all(
        float(row.get("overall_quality", 0.0)) + epsilon
        >= float(before_by_key[key].get("overall_quality", 0.0))
        for key, row in ((str(item.get("placement_key")), item) for item in after.get("decisions", []))
        if key in before_by_key
    )
    average_before = float(before["average_quality"])
    average_after = float(after["average_quality"])
    fewer_failures = (
        int(after["rejected_count"]) < int(before["rejected_count"])
        or int(after["repair_count"]) < int(before["repair_count"])
    )
    materially_improved = average_after > average_before + epsilon or fewer_failures
    return (
        no_placement_regressed
        and materially_improved
        and average_after + epsilon >= average_before
        and int(after["rejected_count"]) <= int(before["rejected_count"])
        and int(after["repair_count"]) <= int(before["repair_count"])
    )


def _select_non_regressing_repair_groups(
    *, previous_schedule: dict[str, Any], candidate_schedule: dict[str, Any],
    previous_verification: dict[str, Any], candidate_verification: dict[str, Any],
    previous_residue: dict[str, Any], candidate_residue: dict[str, Any],
    previous_decisions: dict[str, Any], candidate_decisions: dict[str, Any],
    repairs: list[dict[str, Any]],
) -> dict[str, Any]:
    before = {str(row.get("placement_key")): row for row in previous_decisions.get("decisions", [])}
    after = {str(row.get("placement_key")): row for row in candidate_decisions.get("decisions", [])}
    accepted_groups: list[list[dict[str, Any]]] = []
    rejected_groups: list[list[dict[str, Any]]] = []
    for group in _overlapping_repair_groups(repairs):
        comparisons = [
            (
                float(before.get(str(row.get("placement_key")), {}).get("overall_quality", 0.0)),
                float(after.get(str(row.get("placement_key")), {}).get("overall_quality", 0.0)),
                before.get(str(row.get("placement_key")), {}).get("recommendation"),
                after.get(str(row.get("placement_key")), {}).get("recommendation"),
            )
            for row in group
        ]
        non_regressing = all(new + 0.0001 >= old for old, new, _before_action, _after_action in comparisons)
        materially_better = any(
            new > old + 0.0001 or (_before_action != "accept" and _after_action == "accept")
            for old, new, _before_action, _after_action in comparisons
        )
        (accepted_groups if non_regressing and materially_better else rejected_groups).append(group)

    # Restoring a rejected mapping can reclaim a clip that another provisional
    # accepted repair borrowed. Reject that entire disjoint group as well so the
    # scheduler's no-reuse contract remains true after selective rollback.
    while accepted_groups:
        provisional_rejected = [row for group in rejected_groups for row in group]
        provisional_schedule = deepcopy(candidate_schedule)
        for repair in provisional_rejected:
            index = int(repair.get("mapping_index", -1))
            if 0 <= index < len(provisional_schedule.get("mappings", [])):
                provisional_schedule["mappings"][index] = deepcopy(previous_schedule["mappings"][index])
        clip_counts: dict[str, int] = {}
        for mapping in provisional_schedule.get("mappings", []):
            if not mapping.get("enabled", True):
                continue
            clip_id = str(mapping.get("clip_id") or "")
            if clip_id:
                clip_counts[clip_id] = clip_counts.get(clip_id, 0) + 1
        conflicting_groups = [
            group for group in accepted_groups
            if any(clip_counts.get(str(row.get("new_clip_id") or ""), 0) > 1 for row in group)
        ]
        if not conflicting_groups:
            break
        for group in conflicting_groups:
            accepted_groups.remove(group)
            rejected_groups.append(group)

    accepted_repairs = [row for group in accepted_groups for row in group]
    rejected_repairs = [row for group in rejected_groups for row in group]

    accepted_keys = {str(row.get("placement_key")) for row in accepted_repairs}
    selected_schedule = deepcopy(candidate_schedule)
    for repair in rejected_repairs:
        index = int(repair.get("mapping_index", -1))
        if 0 <= index < len(selected_schedule.get("mappings", [])):
            selected_schedule["mappings"][index] = deepcopy(previous_schedule["mappings"][index])
    selected_rows = [
        row for row in candidate_verification.get("mappings", [])
        if str(row.get("editorial_placement_id") or row.get("mapping_id")) in accepted_keys
    ]
    selected_verification = merge_rendered_dialogue_verification(
        previous_verification, {"mappings": selected_rows},
    )
    accepted_regions = _merge_regions([dict(row.get("region") or {}) for row in accepted_repairs])
    rejected_regions = _merge_regions([dict(row.get("region") or {}) for row in rejected_repairs])
    selected_residue = _merge_residue_regions(previous_residue, candidate_residue, accepted_regions)
    return {
        "schedule": selected_schedule,
        "verification": selected_verification,
        "residue": selected_residue,
        "accepted_repairs": accepted_repairs,
        "rejected_repairs": rejected_repairs,
        "accepted_regions": accepted_regions,
        "rejected_regions": rejected_regions,
    }


def _overlapping_repair_groups(repairs: list[dict[str, Any]], guard_gap: float = 0.35) -> list[list[dict[str, Any]]]:
    ordered = sorted(repairs, key=lambda row: float((row.get("region") or {}).get("start", 0.0) or 0.0))
    parent = list(range(len(ordered)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left, right = find(left), find(right)
        if left != right:
            parent[right] = left

    for left_index, left in enumerate(ordered):
        left_region = left.get("region") or {}
        left_start = float(left_region.get("start", 0.0) or 0.0)
        left_end = float(left_region.get("end", left_start) or left_start)
        for right_index in range(left_index + 1, len(ordered)):
            right = ordered[right_index]
            right_region = right.get("region") or {}
            right_start = float(right_region.get("start", 0.0) or 0.0)
            right_end = float(right_region.get("end", right_start) or right_start)
            same_assignment = bool(left.get("assignment_group_id")) and (
                left.get("assignment_group_id") == right.get("assignment_group_id")
            )
            overlaps = left_start <= right_end + guard_gap and right_start <= left_end + guard_gap
            if same_assignment or overlaps:
                union(left_index, right_index)
    grouped: dict[int, list[dict[str, Any]]] = {}
    for index, repair in enumerate(ordered):
        grouped.setdefault(find(index), []).append(repair)
    return list(grouped.values())


def _merge_residue_regions(
    baseline: dict[str, Any], candidate: dict[str, Any], regions: list[dict[str, Any]],
) -> dict[str, Any]:
    if not regions:
        return dict(baseline)

    def overlaps(row: dict[str, Any]) -> bool:
        start = float(row.get("start", 0.0) or 0.0)
        end = float(row.get("end", start) or start)
        return any(min(end, region["end"]) > max(start, region["start"]) for region in regions)

    rows = [dict(row) for row in baseline.get("regions", []) if not overlaps(row)]
    rows.extend(dict(row) for row in candidate.get("regions", []) if overlaps(row))
    flagged = [row for row in rows if row.get("possible_residue")]
    return {
        **{key: value for key, value in baseline.items() if key != "regions"},
        "status": "POSSIBLE_DESTINATION_SPEECH_DETECTED" if flagged else "NONE_DETECTED" if rows else "INCONCLUSIVE",
        "evaluated_region_count": len(rows),
        "flagged_region_count": len(flagged),
        "regions": rows,
    }
def _pass_record(
    index: int, kind: str, decisions: dict[str, Any], repairs: list[dict[str, Any]], *,
    accepted: bool, rejected_repairs: list[dict[str, Any]] | None = None,
    attempts: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "pass": index, "kind": kind, "accepted": bool(accepted),
        "average_quality": decisions["average_quality"],
        "minimum_quality": decisions["minimum_quality"],
        "accepted_count": decisions["accepted_count"],
        "repair_count": decisions["repair_count"],
        "rejected_count": decisions["rejected_count"],
        "failure_counts": decisions["failure_counts"],
        "repairs": list(repairs),
        "rejected_repairs": list(rejected_repairs or []),
        "repair_attempts": list(attempts or []),
    }


def _repair_attempt_outcomes(
    attempts: list[dict[str, Any]], repairs: list[dict[str, Any]],
    candidate_decisions: dict[str, Any], accepted_keys: set[str],
) -> list[dict[str, Any]]:
    repair_by_key = {str(row.get("placement_key")): row for row in repairs}
    decision_by_key = {
        str(row.get("placement_key")): row for row in candidate_decisions.get("decisions", [])
    }
    outcomes = []
    for source in attempts:
        row = dict(source)
        key = str(row.get("placement_key"))
        repair = repair_by_key.get(key)
        rendered = repair is not None
        survived = rendered and key in accepted_keys
        candidate_decision = decision_by_key.get(key, {}) if rendered else {}
        rendered_quality = candidate_decision.get("overall_quality") if rendered else None
        row.update({
            "rendered": rendered,
            "rendered_verification_quality": rendered_quality,
            "survived": survived,
            "quality_delta": (
                round(float(rendered_quality) - float(row.get("original_quality", 0.0)), 4)
                if rendered_quality is not None else None
            ),
            "original_restored": rendered and not survived,
            "final_stage": (
                "accepted_after_rendered_verification" if survived
                else "restored_after_rendered_regression" if rendered
                else str(row.get("candidate_loss_stage") or "no_viable_alternative")
            ),
        })
        outcomes.append(row)
    return outcomes


def _merge_regions(regions: list[dict[str, Any]], gap: float = 0.08) -> list[dict[str, float]]:
    merged: list[dict[str, float]] = []
    for row in sorted(regions, key=lambda item: float(item.get("start", 0.0) or 0.0)):
        start = float(row.get("start", 0.0) or 0.0)
        end = float(row.get("end", start) or start)
        if end <= start:
            continue
        if merged and start <= merged[-1]["end"] + gap:
            merged[-1]["end"] = max(merged[-1]["end"], end)
        else:
            merged.append({"start": start, "end": end})
    return merged
