from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from . import __version__
from .audio_provenance import ACTIVITY_THRESHOLD_DBFS, ACTIVITY_WINDOW_SECONDS, MAXIMUM_DEAD_AIR_SECONDS, analyze_wav_intervals
from .util import stable_hash, utc_now, write_json


MONTAGE_SCHEMA_VERSION = "1.0"
CORE_HEURISTIC_VERSION = "core_moments_v1"
MONTAGE_PLANNER_VERSION = "montage_planner_v10"
MONTAGE_AUDIO_POLICY_VERSION = "source_soundtrack_activity_v1"
PLACEMENT_AUDIO_POLICY_VERSION = "placement_audio_qualification_v2"
PLACEMENT_GUARD_SECONDS = 0.35
FULL_TIMELINE_POLICY_VERSION = "full_source_timeline_v1"
SHARED_TIMELINE_POLICY_VERSION = "shared_world_timeline_v1"


class CapabilityTag(StrEnum):
    HUMAN_ANNOTATION = "HUMAN_ANNOTATION"
    CORE_HEURISTIC = "CORE_HEURISTIC"
    ENHANCED_VISION = "ENHANCED_VISION"
    ENHANCED_AUDIO = "ENHANCED_AUDIO"
    FALLBACK_INFERENCE = "FALLBACK_INFERENCE"
    CONTRACT_RULE = "CONTRACT_RULE"
    PLANNER_DERIVATION = "PLANNER_DERIVATION"


class StructuralRole(StrEnum):
    BEGINNING = "BEGINNING"
    DEVELOPMENT = "DEVELOPMENT"
    CLIMAX = "CLIMAX"
    RESOLUTION = "RESOLUTION"


@dataclass(frozen=True)
class EvidenceAssertion:
    id: str
    name: str
    capability_tag: CapabilityTag
    value: bool | float | str
    confidence: float
    backend: str
    backend_version: str
    source_artifact: str
    supporting_evidence_ids: tuple[str, ...] = ()
    fallback: bool = False
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["capability_tag"] = self.capability_tag.value
        data["supporting_evidence_ids"] = list(self.supporting_evidence_ids)
        return data


@dataclass(frozen=True)
class CinematicMoment:
    id: str
    source_id: str
    source_media_hash: str
    scene_id: str
    shot_ids: tuple[str, ...]
    start: float
    end: float
    visual_start: float
    visual_end: float
    audio_start: float
    audio_end: float
    assertions: tuple[EvidenceAssertion, ...]
    fallback_status: str = "NONE"

    @property
    def duration(self) -> float:
        return round(self.end - self.start, 3)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source_id": self.source_id,
            "source_media_hash": self.source_media_hash,
            "scene_id": self.scene_id,
            "shot_ids": list(self.shot_ids),
            "start": self.start,
            "end": self.end,
            "duration": self.duration,
            "visual_boundary": {"start": self.visual_start, "end": self.visual_end},
            "audio_boundary": {"start": self.audio_start, "end": self.audio_end},
            "assertions": [row.to_dict() for row in self.assertions],
            "fallback_status": self.fallback_status,
        }


def stable_moment_id(*, source_media_hash: str, start: float, end: float, shot_ids: list[str] | tuple[str, ...]) -> str:
    signature = stable_hash({
        "schema_version": MONTAGE_SCHEMA_VERSION,
        "source_media_hash": source_media_hash,
        "start": round(float(start), 3),
        "end": round(float(end), 3),
        "shot_ids": list(shot_ids),
    })
    return f"moment_{signature[:20]}"


def build_core_moments(
    *,
    source_id: str,
    source_media_hash: str,
    shots: list[dict[str, Any]],
    speech_intervals: list[dict[str, Any]] | None = None,
    transition_intervals: list[dict[str, Any]] | None = None,
    boundary_stability: list[dict[str, Any]] | None = None,
    stillness_intervals: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Conservatively group consecutive shots without inventing internal cuts."""
    ordered = sorted(shots, key=lambda row: (float(row["start"]), str(row["id"])))
    speech = list(speech_intervals or [])
    transitions = list(transition_intervals or [])
    stability = list(boundary_stability or [])
    stillness = list(stillness_intervals or [])
    duration = max((float(row["end"]) for row in ordered), default=0.0)
    silence = _silence_intervals(speech, duration=duration)
    ordered = _subdivide_supported_long_takes(ordered, silence=silence, stillness=stillness, transitions=transitions)
    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for shot in ordered:
        if current:
            boundary = float(shot["start"])
            virtual_boundary = bool(shot.get("_safe_virtual_start"))
            same_scene = _scene_id(current[-1]) == _scene_id(shot)
            high_boundary_motion = _boundary_low_motion(boundary, stability) is False
            if virtual_boundary or not (same_scene or _interval_crosses(boundary, speech) or _interval_crosses(boundary, transitions) or high_boundary_motion):
                groups.append(current)
                current = []
        current.append(shot)
    if current:
        groups.append(current)

    moments = [
        _moment_from_group(
            source_id=source_id,
            source_media_hash=source_media_hash,
            shots=group,
            speech=speech,
            transitions=transitions,
            boundary_stability=stability,
        )
        for group in groups
    ]
    return {
        "schema_version": MONTAGE_SCHEMA_VERSION,
        "tool_version": __version__,
        "creation_timestamp": utc_now(),
        "config_signature": stable_hash({"backend": CORE_HEURISTIC_VERSION, "shots": ordered, "speech": speech, "transitions": transitions, "boundary_stability": stability, "stillness": stillness}),
        "source_id": source_id,
        "source_media_hash": source_media_hash,
        "requested_backend": "core",
        "actual_backend": "core",
        "backend_version": CORE_HEURISTIC_VERSION,
        "capability_tags": [CapabilityTag.CORE_HEURISTIC.value, CapabilityTag.FALLBACK_INFERENCE.value],
        "fallback_path": "preserve_longer_complete_moment",
        "moment_count": len(moments),
        "moments": [moment.to_dict() for moment in moments],
    }


def annotate_moment_audio_activity(moment_artifact: dict[str, Any], *, audio_path: Path) -> dict[str, Any]:
    """Attach literal soundtrack activity evidence used by every montage planner."""
    artifact = dict(moment_artifact)
    moments = [dict(row) for row in moment_artifact.get("moments", [])]
    measurements = analyze_wav_intervals(
        audio_path,
        [{"id": row["id"], "start": row["audio_boundary"]["start"], "end": row["audio_boundary"]["end"]} for row in moments],
    )
    by_id = {str(row.get("id")): row for row in measurements}
    for moment in moments:
        measured = by_id.get(str(moment["id"]))
        if measured is None:
            continue
        maximum_silent_run = float(measured.get("maximum_silent_run_seconds", moment["duration"]) or 0.0)
        eligible = maximum_silent_run <= MAXIMUM_DEAD_AIR_SECONDS
        moment["audio_activity"] = {
            "policy_version": MONTAGE_AUDIO_POLICY_VERSION,
            "threshold_dbfs": ACTIVITY_THRESHOLD_DBFS,
            "window_seconds": ACTIVITY_WINDOW_SECONDS,
            "active_ratio": measured.get("active_ratio", 0.0),
            "silence_ratio": measured.get("silent_ratio", 1.0),
            "maximum_silent_run_seconds": round(maximum_silent_run, 3),
            "maximum_allowed_dead_air_seconds": MAXIMUM_DEAD_AIR_SECONDS,
            "eligible": eligible,
            "silent_intervals": list(measured.get("silent_intervals", [])),
        }
        assertion = _assertion(
            "SOURCE_SOUNDTRACK_HAS_NO_SUSTAINED_DEAD_AIR",
            eligible,
            str(moment["source_id"]),
            float(moment["start"]),
            float(moment["end"]),
            capability_tag=CapabilityTag.ENHANCED_AUDIO,
            note=f"Measured at {ACTIVITY_THRESHOLD_DBFS:.1f} dBFS in {ACTIVITY_WINDOW_SECONDS:.1f}s windows; maximum silent run {maximum_silent_run:.3f}s.",
        ).to_dict()
        moment["assertions"] = [*moment.get("assertions", []), assertion]
    tags = list(artifact.get("capability_tags", []))
    if CapabilityTag.ENHANCED_AUDIO.value not in tags:
        tags.append(CapabilityTag.ENHANCED_AUDIO.value)
    artifact["capability_tags"] = tags
    artifact["moments"] = moments
    artifact["config_signature"] = stable_hash({
        "moment_signature": artifact.get("config_signature"),
        "audio_policy": MONTAGE_AUDIO_POLICY_VERSION,
        "threshold_dbfs": ACTIVITY_THRESHOLD_DBFS,
        "maximum_dead_air_seconds": MAXIMUM_DEAD_AIR_SECONDS,
    })
    return artifact


def build_placement_qualification(
    windows: list[dict[str, Any]],
    moment_artifact: dict[str, Any],
    *,
    shots_artifact: dict[str, Any],
    audio_path: Path,
    guard_seconds: float = PLACEMENT_GUARD_SECONDS,
) -> dict[str, Any]:
    """Qualify dialogue locally and construct only evidence-backed complete-shot submoments."""
    moments = list(moment_artifact.get("moments", []))
    shots = sorted(
        shots_artifact.get("shots", []),
        key=lambda row: (float(row["start"]), str(row["id"])),
    )
    transitions = list(shots_artifact.get("transitions", []))
    preferred_moment_ids = {
        str(moment["id"])
        for moment in moments
        if (moment.get("audio_activity") or {}).get("eligible") is not False
    }
    dialogue_bearing_moment_ids: set[str] = set()
    prepared: list[dict[str, Any]] = []
    candidate_by_id: dict[str, dict[str, Any]] = {}
    local_intervals: list[dict[str, Any]] = []

    for source_window in windows:
        row = dict(source_window)
        start = float(row.get("start", row.get("destination_timestamp", 0.0)) or 0.0)
        end = float(row.get("end", start + float(row.get("duration", 0.0) or 0.0)) or start)
        containing = sorted(
            (
                moment
                for moment in moments
                if float(moment["start"]) <= start and end <= float(moment["end"])
            ),
            key=lambda moment: (float(moment["duration"]), str(moment["id"])),
        )
        parent = containing[0] if containing else None
        if parent is not None:
            dialogue_bearing_moment_ids.add(str(parent["id"]))
        preferred = next(
            (moment for moment in containing if str(moment["id"]) in preferred_moment_ids),
            None,
        )
        if preferred is not None:
            row.update({
                "montage_parent_moment_id": str(preferred["id"]),
                "montage_moment_id": str(preferred["id"]),
                "montage_audio_eligible": True,
                "montage_placement_eligible": True,
                "montage_eligibility_reason": "PREFERRED_WHOLE_MOMENT_AUDIO_QUALIFIED",
                "montage_qualification_stage": "PREFERRED_WHOLE_MOMENT",
            })
            prepared.append({"window": row, "status": "preferred"})
            continue
        if parent is None:
            row.update({
                "montage_parent_moment_id": None,
                "montage_moment_id": None,
                "montage_audio_eligible": False,
                "montage_placement_eligible": False,
                "montage_eligibility_reason": "NOT_CONTAINED_IN_COMPLETE_CINEMATIC_MOMENT",
                "montage_qualification_stage": "LOCAL_COMPLETE_SHOT",
            })
            prepared.append({"window": row, "status": "rejected"})
            continue

        parent_shot_ids = {str(value) for value in parent.get("shot_ids", [])}
        sequence = [
            shot
            for shot in shots
            if (
                str(shot["id"]) in parent_shot_ids
                and float(shot["start"]) < end
                and float(shot["end"]) > start
            )
        ]
        sequence.sort(key=lambda shot: (float(shot["start"]), str(shot["id"])))
        if (
            not sequence
            or float(sequence[0]["start"]) > start + 0.001
            or float(sequence[-1]["end"]) + 0.001 < end
        ):
            row.update({
                "montage_parent_moment_id": str(parent["id"]),
                "montage_moment_id": None,
                "montage_audio_eligible": False,
                "montage_placement_eligible": False,
                "montage_eligibility_reason": "NO_COMPLETE_SHOT_SEQUENCE_CONTAINS_DIALOGUE",
                "montage_qualification_stage": "LOCAL_COMPLETE_SHOT",
            })
            prepared.append({"window": row, "status": "rejected"})
            continue

        guarded_start = max(float(parent["start"]), start - guard_seconds)
        guarded_end = min(float(parent["end"]), end + guard_seconds)
        transition_overlap = any(
            float(transition["start"]) < guarded_end
            and float(transition["end"]) > guarded_start
            for transition in transitions
        )
        if transition_overlap:
            row.update({
                "montage_parent_moment_id": str(parent["id"]),
                "montage_moment_id": None,
                "montage_audio_eligible": False,
                "montage_placement_eligible": False,
                "montage_eligibility_reason": "LOCAL_GUARD_OVERLAPS_TRANSITION",
                "montage_qualification_stage": "LOCAL_COMPLETE_SHOT",
            })
            prepared.append({"window": row, "status": "rejected"})
            continue

        sub_start = round(float(sequence[0]["start"]), 3)
        sub_end = round(float(sequence[-1]["end"]), 3)
        shot_ids = [str(shot["id"]) for shot in sequence]
        submoment_id = stable_moment_id(
            source_media_hash=str(parent["source_media_hash"]),
            start=sub_start,
            end=sub_end,
            shot_ids=shot_ids,
        )
        candidate_by_id.setdefault(submoment_id, {
            "id": submoment_id,
            "parent": parent,
            "start": sub_start,
            "end": sub_end,
            "shot_ids": shot_ids,
        })
        local_id = f"placement_{stable_hash([row.get('id'), start, end, submoment_id])[:20]}"
        local_intervals.append({"id": local_id, "start": guarded_start, "end": guarded_end})
        prepared.append({
            "window": row,
            "status": "candidate",
            "submoment_id": submoment_id,
            "local_id": local_id,
        })

    submeasurements = analyze_wav_intervals(
        audio_path,
        [
            {"id": candidate["id"], "start": candidate["start"], "end": candidate["end"]}
            for candidate in candidate_by_id.values()
        ],
    ) if candidate_by_id else []
    local_measurements = analyze_wav_intervals(audio_path, local_intervals) if local_intervals else []
    submeasurement_by_id = {str(row["id"]): row for row in submeasurements}
    local_measurement_by_id = {str(row["id"]): row for row in local_measurements}
    submoments: dict[str, dict[str, Any]] = {}
    locally_rejected_ids: set[str] = set()
    preferred_placements = 0
    local_placements = 0
    annotated_windows: list[dict[str, Any]] = []

    for item in prepared:
        row = item["window"]
        if item["status"] == "preferred":
            preferred_placements += 1
        elif item["status"] == "candidate":
            submoment_id = str(item["submoment_id"])
            submeasured = submeasurement_by_id.get(submoment_id, {})
            local_measured = local_measurement_by_id.get(str(item["local_id"]), {})
            submaximum = float(submeasured.get("maximum_silent_run_seconds", float("inf")) or 0.0)
            localmaximum = float(local_measured.get("maximum_silent_run_seconds", float("inf")) or 0.0)
            local_safe = localmaximum <= MAXIMUM_DEAD_AIR_SECONDS
            segment_safe = submaximum <= MAXIMUM_DEAD_AIR_SECONDS
            if local_safe and segment_safe:
                candidate = candidate_by_id[submoment_id]
                parent = candidate["parent"]
                if submoment_id not in submoments:
                    submoments[submoment_id] = {
                        "id": submoment_id,
                        "source_id": parent["source_id"],
                        "source_media_hash": parent["source_media_hash"],
                        "scene_id": parent["scene_id"],
                        "shot_ids": list(candidate["shot_ids"]),
                        "start": candidate["start"],
                        "end": candidate["end"],
                        "duration": round(candidate["end"] - candidate["start"], 3),
                        "visual_boundary": {"start": candidate["start"], "end": candidate["end"]},
                        "audio_boundary": {"start": candidate["start"], "end": candidate["end"]},
                        "audio_activity": {
                            "policy_version": PLACEMENT_AUDIO_POLICY_VERSION,
                            "threshold_dbfs": ACTIVITY_THRESHOLD_DBFS,
                            "window_seconds": ACTIVITY_WINDOW_SECONDS,
                            "active_ratio": submeasured.get("active_ratio", 0.0),
                            "silence_ratio": submeasured.get("silent_ratio", 1.0),
                            "maximum_silent_run_seconds": round(submaximum, 3),
                            "maximum_allowed_dead_air_seconds": MAXIMUM_DEAD_AIR_SECONDS,
                            "eligible": True,
                            "silent_intervals": list(submeasured.get("silent_intervals", [])),
                        },
                        "assertions": [
                            *parent.get("assertions", []),
                            _assertion(
                                "LOCAL_DIALOGUE_GUARD_AUDIO_SAFE",
                                True,
                                str(parent["source_id"]),
                                candidate["start"],
                                candidate["end"],
                                capability_tag=CapabilityTag.ENHANCED_AUDIO,
                                note=f"Dialogue-local guard measured maximum silent run {localmaximum:.3f}s.",
                            ).to_dict(),
                            _assertion(
                                "COMPLETE_SHOT_SUBMOMENT_PRESERVED",
                                True,
                                str(parent["source_id"]),
                                candidate["start"],
                                candidate["end"],
                                capability_tag=CapabilityTag.CONTRACT_RULE,
                                note="Placement interval expanded only to existing complete-shot boundaries.",
                            ).to_dict(),
                        ],
                        "fallback_status": "LOCAL_COMPLETE_SHOT_RESCUE",
                        "parent_moment_id": str(parent["id"]),
                        "placement_submoment": True,
                    }
                row.update({
                    "montage_parent_moment_id": str(parent["id"]),
                    "montage_moment_id": submoment_id,
                    "montage_audio_eligible": True,
                    "montage_placement_eligible": True,
                    "montage_eligibility_reason": "LOCAL_COMPLETE_SHOT_AUDIO_QUALIFIED",
                    "montage_qualification_stage": "LOCAL_COMPLETE_SHOT",
                    "montage_submoment_start": candidate["start"],
                    "montage_submoment_end": candidate["end"],
                    "montage_submoment_shot_ids": list(candidate["shot_ids"]),
                })
                local_placements += 1
            else:
                locally_rejected_ids.add(submoment_id)
                row.update({
                    "montage_parent_moment_id": str(candidate_by_id[submoment_id]["parent"]["id"]),
                    "montage_moment_id": None,
                    "montage_audio_eligible": False,
                    "montage_placement_eligible": False,
                    "montage_eligibility_reason": (
                        "LOCAL_GUARD_CONTAINS_SUSTAINED_DEAD_AIR"
                        if not local_safe
                        else "COMPLETE_SHOT_SUBMOMENT_CONTAINS_SUSTAINED_DEAD_AIR"
                    ),
                    "montage_qualification_stage": "LOCAL_COMPLETE_SHOT",
                })
        annotated_windows.append(row)

    rescue_invoked = preferred_placements == 0 and local_placements > 0
    report = {
        "policy_version": PLACEMENT_AUDIO_POLICY_VERSION,
        "cinematic_moments_detected": len(moments),
        "preferred_audio_qualified_moments": len(preferred_moment_ids),
        "dialogue_bearing_moments": len(dialogue_bearing_moment_ids),
        "dialogue_windows_examined": len(windows),
        "complete_shot_submoments_constructed": len(candidate_by_id),
        "locally_rejected_submoments": len(locally_rejected_ids),
        "preferred_legal_placements": preferred_placements,
        "local_legal_placements": local_placements,
        "final_legal_authored_placements": preferred_placements + local_placements,
        "deterministic_rescue_invoked": rescue_invoked,
        "final_output_dead_air_limit_seconds": MAXIMUM_DEAD_AIR_SECONDS,
        "qualification_outcome": (
            "LOCAL_COMPLETE_SHOT_RESCUE"
            if rescue_invoked
            else "PREFERRED_AND_LOCAL_PLACEMENTS"
            if local_placements
            else "PREFERRED_WHOLE_MOMENT_PLACEMENTS"
            if preferred_placements
            else "EXHAUSTED_WITHOUT_LEGAL_PLACEMENT"
        ),
    }
    return {
        "windows": annotated_windows,
        "submoments": list(submoments.values()),
        "report": report,
    }


def moment_artifact_with_authored_submoments(
    moment_artifact: dict[str, Any],
    qualification: dict[str, Any],
    schedule: dict[str, Any],
) -> dict[str, Any]:
    """Add only schedule-referenced placement submoments to the private planning view."""
    referenced = {
        str(mapping.get("montage_moment_id"))
        for mapping in schedule.get("mappings", [])
        if mapping.get("enabled", True) and mapping.get("montage_moment_id")
    }
    authored_submoments = [
        dict(row)
        for row in qualification.get("submoments", [])
        if str(row.get("id")) in referenced
    ]
    artifact = dict(moment_artifact)
    artifact["moments"] = [*moment_artifact.get("moments", []), *authored_submoments]
    artifact["moment_count"] = len(artifact["moments"])
    artifact["placement_submoment_count"] = len(authored_submoments)
    artifact["placement_qualification"] = dict(qualification.get("report", {}))
    return artifact


def build_naive_shot_moments(
    *,
    source_id: str,
    source_media_hash: str,
    shots: list[dict[str, Any]],
) -> dict[str, Any]:
    """Create the explicitly limited baseline used by formal evaluation."""
    moments = []
    for shot in sorted(shots, key=lambda row: (float(row["start"]), str(row["id"]))):
        start = round(float(shot["start"]), 3)
        end = round(float(shot["end"]), 3)
        moment = CinematicMoment(
            id=stable_moment_id(source_media_hash=source_media_hash, start=start, end=end, shot_ids=[str(shot["id"])]),
            source_id=source_id,
            source_media_hash=source_media_hash,
            scene_id=_scene_id(shot),
            shot_ids=(str(shot["id"]),),
            start=start,
            end=end,
            visual_start=start,
            visual_end=end,
            audio_start=start,
            audio_end=end,
            assertions=(
                _assertion(
                    "NAIVE_COMPLETE_SHOT",
                    True,
                    source_id,
                    start,
                    end,
                    capability_tag=CapabilityTag.PLANNER_DERIVATION,
                    note="Baseline uses complete detected shots without speech, motion, transition, or grouping evidence.",
                ),
            ),
        )
        moments.append(moment.to_dict())
    return {
        "schema_version": MONTAGE_SCHEMA_VERSION,
        "tool_version": __version__,
        "creation_timestamp": utc_now(),
        "config_signature": stable_hash({"backend": "naive_shot_sampler_v1", "shots": shots}),
        "source_id": source_id,
        "source_media_hash": source_media_hash,
        "requested_backend": "naive_shot_sampler",
        "actual_backend": "naive_shot_sampler",
        "backend_version": "naive_shot_sampler_v1",
        "capability_tags": [CapabilityTag.PLANNER_DERIVATION.value],
        "fallback_path": "none",
        "moment_count": len(moments),
        "moments": moments,
    }


def build_montage_plan(
    *,
    filter_id: str,
    filter_contract_version: str,
    moment_artifacts: list[dict[str, Any]],
    target_duration: float,
    minimum_moments: int,
    random_seed: int,
    governing_relationship: str,
    laws: dict[str, str],
    maximum_moments: int = 60,
    schedule: dict[str, Any] | None = None,
    repetition_authorized: bool = False,
    repetition_authorization_basis: str | None = None,
    configured_minimum_duration: float | None = None,
) -> dict[str, Any]:
    required_laws = {"visual", "temporal", "dialogue", "requested_audio", "actual_audio_method"}
    missing_laws = sorted(required_laws - set(laws))
    if missing_laws:
        raise ValueError(f"Montage plan is missing declared laws: {', '.join(missing_laws)}")
    if target_duration <= 0:
        raise ValueError("Montage target duration must be positive.")
    if minimum_moments <= 0:
        raise ValueError("Montage minimum moment count must be positive.")
    if repetition_authorized and not repetition_authorization_basis:
        raise ValueError("Authorized repetition must name its filter-plan authorization basis.")
    if configured_minimum_duration is not None and configured_minimum_duration <= 0:
        raise ValueError("Configured montage minimum duration must be positive when supplied.")

    repetition_policy = _repetition_policy(
        schedule=schedule,
        authorized=repetition_authorized,
        authorization_basis=repetition_authorization_basis,
    )

    all_moments = [dict(moment) for artifact in moment_artifacts for moment in artifact.get("moments", [])]
    upstream_rejections = [dict(row) for artifact in moment_artifacts for row in artifact.get("candidate_rejections", [])]
    if not all_moments:
        raise ValueError("Montage planning requires at least one cinematic moment.")
    all_moments.sort(key=lambda row: (str(row["source_id"]), float(row["start"]), str(row["id"])))
    authored_metadata_present = any("authored_placement_count" in row for row in all_moments)
    authored_moments = [row for row in all_moments if int(row.get("authored_placement_count", 0) or 0) > 0]
    if authored_metadata_present and not authored_moments:
        raise ValueError("Montage planning requires at least one audio-safe moment containing an authored filter placement.")
    selected = _select_for_target_duration(
        all_moments,
        target_duration=float(target_duration),
        minimum_moments=minimum_moments,
        maximum_moments=maximum_moments,
        random_seed=random_seed,
    )
    selected_ids = {str(row["id"]) for row in selected}
    roles = _structural_roles(len(selected))
    selected_rows = []
    for index, (moment, role) in enumerate(zip(selected, roles)):
        selected_rows.append({
            **moment,
            "montage_index": index,
            "structural_role": role.value,
            "planner_assertions": [
                _planner_assertion("MOMENT_SELECTED", True, moment, note=f"Selected by {governing_relationship} planning."),
                _planner_assertion("STRUCTURAL_ROLE_ASSIGNED", role.value, moment),
            ],
        })
    rejected = upstream_rejections + [
        {"moment_id": row["id"], "source_id": row["source_id"], "reason": "NOT_SELECTED_WITHIN_TARGET_SHAPE"}
        for row in all_moments
        if str(row["id"]) not in selected_ids
    ]
    actual_duration = round(sum(float(row["duration"]) for row in selected_rows), 3)
    available_duration = round(sum(float(row["duration"]) for row in all_moments), 3)
    available_budget = round(min(float(target_duration), available_duration), 3)
    utilization_ratio = round(min(1.0, actual_duration / available_budget), 4) if available_budget > 0 else 0.0
    minimum_utilization_ratio = 0.8
    selected_authored_count = sum(int(row.get("authored_placement_count", 0) or 0) > 0 for row in selected_rows)
    selected_context_count = sum(str(row.get("montage_content_role", "")) == "CONTEXT" for row in selected_rows)
    placement_qualification = dict((schedule or {}).get("placement_qualification") or {
        "policy_version": PLACEMENT_AUDIO_POLICY_VERSION,
        "cinematic_moments_detected": len(all_moments),
        "preferred_audio_qualified_moments": sum(
            (row.get("audio_activity") or {}).get("eligible") is not False
            for row in all_moments
        ),
        "dialogue_bearing_moments": 0,
        "dialogue_windows_examined": 0,
        "complete_shot_submoments_constructed": 0,
        "locally_rejected_submoments": 0,
        "preferred_legal_placements": len((schedule or {}).get("mappings", [])),
        "local_legal_placements": 0,
        "final_legal_authored_placements": len((schedule or {}).get("mappings", [])),
        "deterministic_rescue_invoked": False,
        "final_output_dead_air_limit_seconds": MAXIMUM_DEAD_AIR_SECONDS,
        "qualification_outcome": "LEGACY_PLACEMENT_PATH_REPORTED",
    })
    if authored_metadata_present and selected_authored_count == 0:
        raise ValueError("Montage selection omitted every authored filter placement.")
    if utilization_ratio + 0.0001 < minimum_utilization_ratio:
        raise ValueError(
            f"Montage planning used only {actual_duration:.3f}s of {available_budget:.3f}s available within the target "
            f"({utilization_ratio:.1%}); safe material must be substantially utilized."
        )
    source_durations: dict[str, float] = {}
    source_counts: dict[str, int] = {}
    for row in selected_rows:
        source = str(row["source_id"])
        source_durations[source] = source_durations.get(source, 0.0) + float(row["duration"])
        source_counts[source] = source_counts.get(source, 0) + 1
    fallback_decisions = []
    if len(selected_rows) < minimum_moments:
        fallback_decisions.append({
            "id": f"fallback_{stable_hash([filter_id, random_seed, 'minimum_moments'])[:20]}",
            "capability_tag": CapabilityTag.FALLBACK_INFERENCE.value,
            "reason": "INSUFFICIENT_SAFE_MOMENTS",
            "requested_minimum": minimum_moments,
            "actual_count": len(selected_rows),
            "action": "RELAXED_MINIMUM_MOMENT_COUNT",
        })
    if actual_duration > float(target_duration) + 0.001:
        fallback_decisions.append({
            "id": f"fallback_{stable_hash([filter_id, random_seed, 'target_duration'])[:20]}",
            "capability_tag": CapabilityTag.FALLBACK_INFERENCE.value,
            "reason": "MINIMUM_SAFE_MOMENT_COUNT_EXCEEDS_TARGET_DURATION",
            "requested_duration": round(float(target_duration), 3),
            "actual_duration": actual_duration,
            "action": "PRESERVED_MINIMUM_COMPLETE_MOMENTS",
        })
    shortened = actual_duration + 0.001 < float(target_duration)
    if shortened:
        shortage = available_duration + 0.001 < float(target_duration)
        fallback_decisions.append({
            "id": f"fallback_{stable_hash([filter_id, random_seed, 'shorten_to_audio'])[:20]}",
            "capability_tag": CapabilityTag.FALLBACK_INFERENCE.value,
            "reason": (
                "INSUFFICIENT_NON_REPEATED_SOURCE_AUDIO_FOR_TARGET_DURATION"
                if shortage
                else "COMPLETE_AUDIO_BEARING_MOMENT_PACKING_BELOW_TARGET_DURATION"
            ),
            "requested_duration": round(float(target_duration), 3),
            "available_non_repeating_montage_duration": available_duration,
            "actual_duration": actual_duration,
            "configured_minimum_duration": (
                round(float(configured_minimum_duration), 3)
                if configured_minimum_duration is not None
                else None
            ),
            "action": "SHORTENED_TARGET_VIDEO_TO_AVAILABLE_SOURCE_AUDIO",
            "repetition_used": repetition_policy["observed_repeated_placement_count"] > 0,
        })
    configured_minimum_relaxed = bool(
        configured_minimum_duration is not None
        and actual_duration + 0.001 < float(configured_minimum_duration)
    )
    source_participation = {
        source: {
            "moment_count": source_counts[source],
            "duration": round(duration, 3),
            "share": round(duration / actual_duration, 4) if actual_duration > 0 else 0.0,
        }
        for source, duration in sorted(source_durations.items())
    }
    opening = selected_rows[0]
    same_source_eligible = sorted(
        (row for row in all_moments if str(row["source_id"]) == str(opening["source_id"])),
        key=lambda row: (float(row["start"]), str(row["id"])),
    )
    opening_rank = next(index for index, row in enumerate(same_source_eligible) if str(row["id"]) == str(opening["id"]))
    opening_selection = {
        "moment_id": opening["id"],
        "source_id": opening["source_id"],
        "source_start": opening["start"],
        "eligible_moment_count_for_source": len(same_source_eligible),
        "eligible_timeline_rank": opening_rank,
        "normalized_timeline_position": round(opening_rank / max(1, len(same_source_eligible) - 1), 4),
        "earliest_eligible_selected": opening_rank == 0,
        "chronology_required_by_filter": False,
        "timeline_position_primary_tiebreaker": False,
        "selection_basis": ["STRUCTURAL_ROLE", "GOVERNING_RELATIONSHIP", "CINEMATIC_INTEGRITY", "SEEDED_DIVERSITY"],
    }
    return {
        "schema_version": MONTAGE_SCHEMA_VERSION,
        "planner_version": MONTAGE_PLANNER_VERSION,
        "filter_id": filter_id,
        "filter_contract_version": filter_contract_version,
        "random_seed": int(random_seed),
        "governing_relationship": governing_relationship,
        "laws": {key: laws[key] for key in ("visual", "temporal", "dialogue", "requested_audio", "actual_audio_method")},
        "requested_duration": round(float(target_duration), 3),
        "requested_minimum_moments": int(minimum_moments),
        "actual_duration": actual_duration,
        "duration_resolution": {
            "policy": "TARGET_IS_CEILING_SHORTEN_TO_NON_REPEATED_SOURCE_AUDIO",
            "requested_target_duration": round(float(target_duration), 3),
            "configured_minimum_duration": (
                round(float(configured_minimum_duration), 3)
                if configured_minimum_duration is not None
                else None
            ),
            "available_non_repeating_montage_duration": available_duration,
            "resolved_duration": actual_duration,
            "shortened": shortened,
            "configured_minimum_relaxed": configured_minimum_relaxed,
        },
        "repetition_policy": repetition_policy,
        "material_utilization": {
            "policy": "MAXIMIZE_AUDIO_SAFE_MATERIAL_WITHIN_TARGET",
            "available_audio_safe_duration": available_duration,
            "available_within_target_duration": available_budget,
            "selected_duration": actual_duration,
            "utilization_ratio": utilization_ratio,
            "minimum_utilization_ratio": minimum_utilization_ratio,
            "utilization_sufficient": utilization_ratio >= minimum_utilization_ratio,
            "available_moment_count": len(all_moments),
            "selected_moment_count": len(selected_rows),
            "available_authored_moment_count": len(authored_moments),
            "selected_authored_moment_count": selected_authored_count,
            "selected_context_moment_count": selected_context_count,
        },
        "placement_qualification": placement_qualification,
        "structural_roles": [role.value for role in roles],
        "opening_selection": opening_selection,
        "source_participation": source_participation,
        "selected_moments": selected_rows,
        "rejected_candidates": rejected,
        "fallback_decisions": fallback_decisions,
        "provenance": {
            "source_artifact_ids": [str(row.get("source_id")) for row in moment_artifacts],
            "source_media_hashes": sorted({str(row.get("source_media_hash")) for row in moment_artifacts}),
            "moment_schema_version": MONTAGE_SCHEMA_VERSION,
            "planner_capability_tag": CapabilityTag.PLANNER_DERIVATION.value,
        },
        "verdict": "EXPERIMENTAL",
    }


def build_full_timeline_plan(
    *,
    filter_id: str,
    filter_contract_version: str,
    anchor_source_id: str,
    anchor_media_hash: str,
    anchor_duration: float,
    supporting_audio_durations: list[float] | tuple[float, ...] = (),
    shot_ids: list[str] | tuple[str, ...] = (),
    random_seed: int,
    governing_relationship: str,
    laws: dict[str, str],
    schedule: dict[str, Any] | None = None,
    repetition_authorized: bool = False,
    repetition_authorization_basis: str | None = None,
) -> dict[str, Any]:
    """Consume the anchor continuously, curtailed only by finite audio support."""
    required_laws = {"visual", "temporal", "dialogue", "requested_audio", "actual_audio_method"}
    missing_laws = sorted(required_laws - set(laws))
    if missing_laws:
        raise ValueError(f"Full-timeline plan is missing declared laws: {', '.join(missing_laws)}")
    anchor_duration = float(anchor_duration)
    if anchor_duration <= 0:
        raise ValueError("Full-timeline planning requires a positive anchor duration.")
    supports = [float(value) for value in supporting_audio_durations if float(value) > 0]
    resolved_duration = round(min([anchor_duration, *supports]), 3)
    if resolved_duration <= 0:
        raise ValueError("Full-timeline planning found no positive audio-supported duration.")
    repetition_policy = _repetition_policy(
        schedule=schedule,
        authorized=repetition_authorized,
        authorization_basis=repetition_authorization_basis,
    )
    timeline_shot_ids = list(shot_ids) or ["full_source_timeline"]
    moment_id = stable_moment_id(
        source_media_hash=anchor_media_hash,
        start=0.0,
        end=resolved_duration,
        shot_ids=timeline_shot_ids,
    )
    selected = {
        "id": moment_id,
        "source_id": anchor_source_id,
        "source_media_hash": anchor_media_hash,
        "scene_id": "full_source_timeline",
        "shot_ids": timeline_shot_ids,
        "start": 0.0,
        "end": resolved_duration,
        "duration": resolved_duration,
        "visual_boundary": {"start": 0.0, "end": resolved_duration},
        "audio_boundary": {"start": 0.0, "end": resolved_duration},
        "assertions": [],
        "fallback_status": "NONE",
        "montage_index": 0,
        "structural_role": StructuralRole.BEGINNING.value,
        "planner_assertions": [{
            "name": "FULL_SOURCE_TIMELINE_SELECTED",
            "value": True,
            "note": "The anchor is consumed continuously from timestamp zero without short-form selection.",
        }],
    }
    shortened = resolved_duration + 0.001 < anchor_duration
    fallback_decisions = []
    if shortened:
        fallback_decisions.append({
            "id": f"fallback_{stable_hash([filter_id, random_seed, 'supporting_audio_end'])[:20]}",
            "capability_tag": CapabilityTag.CONTRACT_RULE.value,
            "reason": "SUPPORTING_AUDIO_SHORTER_THAN_ANCHOR_VIDEO",
            "requested_duration": round(anchor_duration, 3),
            "available_supporting_audio_duration": round(min(supports), 3),
            "actual_duration": resolved_duration,
            "action": "CURTAILED_VIDEO_TO_SUPPORTING_AUDIO",
        })
    placement_qualification = dict((schedule or {}).get("placement_qualification") or {
        "policy_version": FULL_TIMELINE_POLICY_VERSION,
        "cinematic_moments_detected": 1,
        "preferred_audio_qualified_moments": 1,
        "dialogue_bearing_moments": 1 if (schedule or {}).get("mappings") else 0,
        "dialogue_windows_examined": len((schedule or {}).get("mappings", [])),
        "complete_shot_submoments_constructed": 0,
        "locally_rejected_submoments": 0,
        "preferred_legal_placements": len((schedule or {}).get("mappings", [])),
        "local_legal_placements": 0,
        "final_legal_authored_placements": len((schedule or {}).get("mappings", [])),
        "deterministic_rescue_invoked": False,
        "final_output_dead_air_limit_seconds": MAXIMUM_DEAD_AIR_SECONDS,
        "qualification_outcome": "FULL_TIMELINE_NO_SHORT_FORM_SELECTION",
    })
    return {
        "schema_version": MONTAGE_SCHEMA_VERSION,
        "planner_version": FULL_TIMELINE_POLICY_VERSION,
        "filter_id": filter_id,
        "filter_contract_version": filter_contract_version,
        "random_seed": int(random_seed),
        "governing_relationship": governing_relationship,
        "laws": {key: laws[key] for key in ("visual", "temporal", "dialogue", "requested_audio", "actual_audio_method")},
        "requested_duration": round(anchor_duration, 3),
        "requested_minimum_moments": 1,
        "actual_duration": resolved_duration,
        "duration_resolution": {
            "policy": "FULL_SOURCE_TIMELINE_LIMITED_BY_SUPPORTING_AUDIO",
            "requested_target_duration": round(anchor_duration, 3),
            "configured_minimum_duration": None,
            "available_non_repeating_montage_duration": resolved_duration,
            "resolved_duration": resolved_duration,
            "shortened": shortened,
            "configured_minimum_relaxed": False,
        },
        "repetition_policy": {
            **repetition_policy,
            "default_when_source_audio_is_short": "CURTAIL_VIDEO_TO_SUPPORTING_AUDIO",
        },
        "material_utilization": {
            "policy": "USE_COMPLETE_SOURCE_TIMELINE",
            "available_audio_safe_duration": resolved_duration,
            "available_within_target_duration": resolved_duration,
            "selected_duration": resolved_duration,
            "utilization_ratio": 1.0,
            "minimum_utilization_ratio": 1.0,
            "utilization_sufficient": True,
            "available_moment_count": 1,
            "selected_moment_count": 1,
            "available_authored_moment_count": 1 if (schedule or {}).get("mappings") else 0,
            "selected_authored_moment_count": 1 if (schedule or {}).get("mappings") else 0,
            "selected_context_moment_count": 0,
        },
        "placement_qualification": placement_qualification,
        "structural_roles": [StructuralRole.BEGINNING.value],
        "opening_selection": {
            "moment_id": moment_id,
            "source_id": anchor_source_id,
            "source_start": 0.0,
            "eligible_moment_count_for_source": 1,
            "eligible_timeline_rank": 0,
            "normalized_timeline_position": 0.0,
            "earliest_eligible_selected": True,
            "chronology_required_by_filter": True,
            "timeline_position_primary_tiebreaker": False,
            "selection_basis": ["FULL_SOURCE_TIMELINE_CONTRACT"],
        },
        "source_participation": {
            anchor_source_id: {"moment_count": 1, "duration": resolved_duration, "share": 1.0},
        },
        "selected_moments": [selected],
        "rejected_candidates": [],
        "fallback_decisions": fallback_decisions,
        "provenance": {
            "source_artifact_ids": [anchor_source_id],
            "source_media_hashes": [anchor_media_hash],
            "moment_schema_version": MONTAGE_SCHEMA_VERSION,
            "planner_capability_tag": CapabilityTag.CONTRACT_RULE.value,
            "input_scope": "COMPLETE_MEDIA_FILES",
        },
        "verdict": "PRODUCTION_READY",
    }


def build_shared_timeline_plan(
    *,
    filter_id: str,
    filter_contract_version: str,
    anchor_source_id: str,
    anchor_media_hash: str,
    anchor_duration: float,
    supporting_audio_durations: list[float] | tuple[float, ...],
    random_seed: int,
    governing_relationship: str,
    laws: dict[str, str],
    schedule: dict[str, Any],
) -> dict[str, Any]:
    """Compile an ordered, non-repeating timeline whose picture comes from several films."""
    plan = build_full_timeline_plan(
        filter_id=filter_id,
        filter_contract_version=filter_contract_version,
        anchor_source_id=anchor_source_id,
        anchor_media_hash=anchor_media_hash,
        anchor_duration=anchor_duration,
        supporting_audio_durations=supporting_audio_durations,
        random_seed=random_seed,
        governing_relationship=governing_relationship,
        laws=laws,
        schedule=schedule,
    )
    segments = list(schedule.get("visual_segments", []))
    if len(segments) < 2:
        raise ValueError("Shared-timeline planning requires at least two visual source segments.")
    selected = []
    participation: dict[str, dict[str, Any]] = {}
    roles = [StructuralRole.BEGINNING.value, StructuralRole.DEVELOPMENT.value, StructuralRole.RESOLUTION.value]
    for index, segment in enumerate(segments):
        source_start = float(segment["source_start"])
        source_end = float(segment["source_end"])
        duration = round(source_end - source_start, 3)
        source_id = str(segment["source_film_id"])
        selected.append({
            "id": str(segment["id"]),
            "source_id": source_id,
            "source_media_hash": str(segment["source_media_hash"]),
            "source_path": str(segment["source_path"]),
            "scene_id": f"shared_phase_{index + 1}",
            "shot_ids": list(segment["shot_ids"]),
            "start": source_start,
            "end": source_end,
            "duration": duration,
            "visual_boundary": {"start": source_start, "end": source_end},
            "audio_boundary": {"start": source_start, "end": source_end},
            "carrier_speech_regions": list(segment.get("carrier_speech_regions", [])),
            "assertions": [],
            "fallback_status": "NONE",
            "montage_index": index,
            "structural_role": roles[min(index, len(roles) - 1)],
            "planner_assertions": [{
                "name": "DECLARED_MULTI_SOURCE_PHASE_SELECTED",
                "value": True,
                "note": f"Phase {index + 1} preserves its declared source-film picture and soundtrack.",
            }],
        })
        participation[source_id] = {"moment_count": 1, "duration": duration, "share": round(duration / plan["actual_duration"], 4)}
    plan.update({
        "planner_version": SHARED_TIMELINE_POLICY_VERSION,
        "requested_minimum_moments": len(selected),
        "structural_roles": roles[:len(selected)],
        "source_participation": participation,
        "selected_moments": selected,
        "opening_selection": {
            "moment_id": selected[0]["id"],
            "source_id": selected[0]["source_id"],
            "source_start": selected[0]["start"],
            "eligible_moment_count_for_source": 1,
            "eligible_timeline_rank": 0,
            "normalized_timeline_position": 0.0,
            "earliest_eligible_selected": True,
            "chronology_required_by_filter": True,
            "timeline_position_primary_tiebreaker": False,
            "selection_basis": ["DECLARED_CYCLIC_FILM_ORDER"],
        },
    })
    plan["duration_resolution"]["policy"] = "SHARED_TIMELINE_LIMITED_BY_SHORTEST_FILM"
    plan["material_utilization"].update({
        "policy": "BUILD_SHARED_TIMELINE_WITHOUT_REPETITION",
        "available_moment_count": len(selected),
        "selected_moment_count": len(selected),
        "available_authored_moment_count": len(selected),
        "selected_authored_moment_count": len(selected),
    })
    plan["placement_qualification"].update({
        "cinematic_moments_detected": len(selected),
        "preferred_audio_qualified_moments": len(selected),
        "dialogue_bearing_moments": len(selected),
        "qualification_outcome": "DECLARED_MULTI_SOURCE_PHASES",
    })
    plan["provenance"].update({
        "source_artifact_ids": [row["source_id"] for row in selected],
        "source_media_hashes": sorted({row["source_media_hash"] for row in selected}),
        "planner_capability_tag": CapabilityTag.CONTRACT_RULE.value,
        "input_scope": "COMPLETE_MEDIA_FILES",
    })
    for decision in plan["fallback_decisions"]:
        if decision.get("action") == "CURTAILED_VIDEO_TO_SUPPORTING_AUDIO":
            decision.update({
                "reason": "SHARED_TIMELINE_BOUNDED_BY_SHORTEST_FILM",
                "action": "BUILT_SHARED_TIMELINE_TO_COMMON_DURATION",
            })
    return plan


def _repetition_policy(
    *,
    schedule: dict[str, Any] | None,
    authorized: bool,
    authorization_basis: str | None,
) -> dict[str, Any]:
    mappings = [row for row in (schedule or {}).get("mappings", []) if row.get("enabled", True)]
    clip_counts: dict[str, int] = {}
    for mapping in mappings:
        clip_id = str(mapping.get("clip_id") or "").strip()
        if clip_id:
            clip_counts[clip_id] = clip_counts.get(clip_id, 0) + 1
    repeated = {clip_id: count for clip_id, count in sorted(clip_counts.items()) if count > 1}
    repeated_placements = sum(count - 1 for count in repeated.values())
    if repeated_placements and not authorized:
        raise ValueError(
            "Montage schedule repeats source audio without explicit authorization in the active filter plan: "
            + ", ".join(f"{clip_id} x{count}" for clip_id, count in repeated.items())
        )
    return {
        "default_when_source_audio_is_short": "SHORTEN_TARGET_VIDEO",
        "authorized": bool(authorized),
        "authorization_basis": authorization_basis if authorized else "FORBIDDEN_BY_DEFAULT",
        "observed_repeated_source_clip_ids": list(repeated),
        "observed_repeated_placement_count": repeated_placements,
    }


def moments_with_schedule_coverage(
    moment_artifact: dict[str, Any],
    schedule: dict[str, Any],
    *,
    include_audio_safe_context: bool = False,
) -> dict[str, Any]:
    """Build safe montage candidates, optionally retaining untransformed contextual material."""
    mappings = [row for row in schedule.get("mappings", []) if row.get("enabled", True)]
    artifact = dict(moment_artifact)
    eligible = []
    rejected = []
    for moment in moment_artifact.get("moments", []):
        covering_mappings = [
            row for row in mappings
            if (
            float(moment["start"]) <= float(row.get("destination_timestamp", -1.0))
            and float(row.get("destination_timestamp", -1.0)) + float(row.get("planned_render_duration", row.get("clip_trim_duration", 0.0)) or 0.0) <= float(moment["end"])
            )
        ]
        covered = bool(covering_mappings)
        audio = moment.get("audio_activity") or {}
        audio_eligible = audio.get("eligible") is not False
        if audio_eligible and (covered or include_audio_safe_context):
            row = dict(moment)
            row["authored_placement_count"] = len(covering_mappings)
            row["montage_content_role"] = "TRANSFORMED" if covered else "CONTEXT"
            eligible.append(row)
        elif covered:
            rejected.append({
                "moment_id": moment["id"],
                "source_id": moment["source_id"],
                "reason": "SOURCE_SOUNDTRACK_SUSTAINED_DEAD_AIR",
                "maximum_silent_run_seconds": audio.get("maximum_silent_run_seconds"),
                "maximum_allowed_dead_air_seconds": audio.get("maximum_allowed_dead_air_seconds", MAXIMUM_DEAD_AIR_SECONDS),
            })
    artifact["moments"] = eligible
    artifact["moment_count"] = len(artifact["moments"])
    artifact["coverage_filter"] = (
        "AUDIO_SAFE_CONTEXT_WITH_REQUIRED_AUTHORED_PLACEMENT"
        if include_audio_safe_context
        else "ENABLED_DIALOGUE_PLACEMENT_WITH_AUDIO_CONTINUITY"
    )
    artifact["authored_placement_moment_count"] = sum(
        int(row.get("authored_placement_count", 0) or 0) > 0 for row in eligible
    )
    artifact["context_moment_count"] = sum(row.get("montage_content_role") == "CONTEXT" for row in eligible)
    artifact["candidate_rejections"] = rejected
    return artifact


def annotate_windows_with_montage_eligibility(
    windows: list[dict[str, Any]],
    moment_artifact: dict[str, Any],
) -> list[dict[str, Any]]:
    """Attach complete-moment and audio eligibility before a filter authors placements."""
    moments = list(moment_artifact.get("moments", []))
    annotated = []
    for window in windows:
        row = dict(window)
        start = float(row.get("start", row.get("destination_timestamp", 0.0)) or 0.0)
        end = float(row.get("end", start + float(row.get("duration", 0.0) or 0.0)) or start)
        containing = [
            moment for moment in moments
            if float(moment["start"]) <= start and end <= float(moment["end"])
        ]
        containing.sort(key=lambda moment: (float(moment["duration"]), str(moment["id"])))
        safe = next(
            (
                moment for moment in containing
                if (moment.get("audio_activity") or {}).get("eligible") is not False
            ),
            None,
        )
        selected = safe or (containing[0] if containing else None)
        row["montage_moment_id"] = str(selected["id"]) if selected is not None else None
        row["montage_audio_eligible"] = safe is not None
        row["montage_placement_eligible"] = safe is not None
        row["montage_eligibility_reason"] = (
            "COMPLETE_AUDIO_SAFE_MOMENT"
            if safe is not None
            else "SOURCE_SOUNDTRACK_SUSTAINED_DEAD_AIR"
            if containing
            else "NOT_CONTAINED_IN_COMPLETE_CINEMATIC_MOMENT"
        )
        annotated.append(row)
    return annotated


def rebase_schedule_to_montage(schedule: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    """Map full-film dialogue placements into the deterministic montage timeline."""
    rebased = dict(schedule)
    source_mappings = [row for row in schedule.get("mappings", []) if row.get("enabled", True)]
    source_speech_regions = list(schedule.get("destination_speech_regions", []))
    source_performance_fills = list(schedule.get("destination_performance_fills", []))
    mappings = []
    audio_segments = []
    carrier_speech_regions = []
    destination_speech_regions = []
    destination_performance_fills = []
    cursor = 0.0
    for moment in plan.get("selected_moments", []):
        start, end = float(moment["start"]), float(moment["end"])
        moment_id = str(moment["id"])
        speech_region_ids: dict[str, str] = {}
        speech_region_rows: dict[str, dict[str, Any]] = {}
        audio_segments.append({
            "moment_id": moment_id,
            "output_start": round(cursor, 3),
            "output_end": round(cursor + end - start, 3),
            "source_start": round(start, 3),
            "source_end": round(end, 3),
            "source_id": moment.get("source_id"),
            "source_media_hash": moment.get("source_media_hash"),
            "classification_basis": "CONTINUOUS_SELECTED_SOURCE_SOUNDTRACK",
        })
        for region_index, region in enumerate(source_speech_regions):
            region_start = float(region.get("start", 0.0) or 0.0)
            region_end = float(region.get("end", region_start + float(region.get("duration", 0.0) or 0.0)) or region_start)
            overlap_start, overlap_end = max(start, region_start), min(end, region_end)
            if overlap_end <= overlap_start:
                continue
            source_id = str(region.get("id") or f"speech_{region_index + 1:06d}")
            rebased_id = f"{source_id}@{moment_id}"
            row = dict(region)
            row.update({
                "id": rebased_id,
                "source_region_id": source_id,
                "source_start": round(overlap_start, 3),
                "source_end": round(overlap_end, 3),
                "start": round(cursor + overlap_start - start, 3),
                "end": round(cursor + overlap_end - start, 3),
                "duration": round(overlap_end - overlap_start, 3),
                "montage_moment_id": moment_id,
            })
            destination_speech_regions.append(row)
            speech_region_ids[source_id] = rebased_id
            speech_region_rows[source_id] = row
        for region in moment.get("carrier_speech_regions", []):
            source_start = max(start, float(region.get("source_start", start)))
            source_end = min(end, float(region.get("source_end", source_start)))
            if source_end <= source_start:
                continue
            carrier_speech_regions.append({
                "id": str(region.get("id") or f"carrier_speech_{len(carrier_speech_regions) + 1}"),
                "source_film_id": str(region.get("source_film_id") or moment.get("source_id") or ""),
                "triangle_phase": region.get("triangle_phase"),
                "source_start": round(source_start, 3),
                "source_end": round(source_end, 3),
                "start": round(cursor + source_start - start, 3),
                "duration": round(source_end - source_start, 3),
            })
        for mapping in source_mappings:
            timestamp = float(mapping.get("destination_timestamp", -1.0))
            mapping_end = timestamp + float(mapping.get("planned_render_duration", mapping.get("clip_trim_duration", 0.0)) or 0.0)
            if not (start <= timestamp and mapping_end <= end):
                continue
            row = dict(mapping)
            row["source_destination_timestamp"] = round(timestamp, 3)
            row["destination_timestamp"] = round(cursor + timestamp - start, 3)
            row["montage_moment_id"] = moment_id
            row["montage_index"] = int(moment["montage_index"])
            source_window_ids = [str(item) for item in row.get("alignment_source_window_ids", [])]
            row["alignment_source_window_ids"] = [
                speech_region_ids[item] for item in source_window_ids if item in speech_region_ids
            ]
            if row.get("window_id") is not None and str(row["window_id"]) in speech_region_ids:
                row["source_window_id"] = str(row["window_id"])
                row["window_id"] = speech_region_ids[str(row["window_id"])]
            slot_start = row.get("alignment_slot_start")
            slot_end = row.get("alignment_slot_end")
            if slot_start is not None and slot_end is not None:
                row["source_alignment_slot_start"] = round(float(slot_start), 3)
                row["source_alignment_slot_end"] = round(float(slot_end), 3)
                row["alignment_slot_start"] = round(cursor + max(start, float(slot_start)) - start, 3)
                row["alignment_slot_end"] = round(cursor + min(end, float(slot_end)) - start, 3)
            mappings.append(row)
        for fill_index, fill in enumerate(source_performance_fills):
            fill_start = float(fill.get("start", 0.0) or 0.0)
            fill_end = fill_start + float(fill.get("duration", 0.0) or 0.0)
            overlap_start, overlap_end = max(start, fill_start), min(end, fill_end)
            if overlap_end <= overlap_start:
                continue
            speech_windows = [
                dict(speech_region_rows[str(slot.get("id"))])
                for slot in fill.get("speech_windows", [])
                if str(slot.get("id")) in speech_region_rows
            ]
            if not speech_windows:
                continue
            row = dict(fill)
            row.update({
                "start": round(cursor + overlap_start - start, 3),
                "duration": round(overlap_end - overlap_start, 3),
                "speech_windows": speech_windows,
                "montage_moment_id": moment_id,
                "source_performance_fill_index": fill_index,
            })
            destination_performance_fills.append(row)
        cursor += end - start
    rebased["mappings"] = sorted(mappings, key=lambda row: (float(row["destination_timestamp"]), str(row.get("clip_id"))))
    if source_speech_regions:
        rebased["destination_speech_regions"] = destination_speech_regions
    if source_performance_fills:
        _refresh_rebased_fill_coverage(destination_performance_fills, rebased["mappings"])
        rebased["destination_performance_fills"] = destination_performance_fills
    rebased["render_duration"] = round(cursor, 3)
    rebased["self_shuffle_render_strategy"] = "shot_aware_montage_v1"
    rebased["montage_plan_verdict"] = plan.get("verdict", "EXPERIMENTAL")
    rebased["montage_plan_selected_moment_ids"] = [row["id"] for row in plan.get("selected_moments", [])]
    rebased["montage_audio_segments"] = audio_segments
    if carrier_speech_regions:
        rebased["carrier_speech_regions"] = carrier_speech_regions
        rebased["carrier_speech_policy"] = "HARD_SUPPRESS_ALL_DETECTED_CARRIER_SPEECH"
    if plan.get("planner_version") == FULL_TIMELINE_POLICY_VERSION:
        rebased["audio_continuity_policy"] = "FULL_SOURCE_SOUNDTRACK_BED"
        rebased["input_scope"] = "complete_media_files"
    elif plan.get("planner_version") == SHARED_TIMELINE_POLICY_VERSION:
        rebased["audio_continuity_policy"] = (
            "MULTI_SOURCE_NON_SPEECH_BED_WITH_CARRIER_SPEECH_SUPPRESSION"
            if carrier_speech_regions
            else "DECLARED_MULTI_SOURCE_SOUNDTRACK_BED"
        )
        rebased["input_scope"] = "complete_media_files"
    return rebased


def _refresh_rebased_fill_coverage(fills: list[dict[str, Any]], mappings: list[dict[str, Any]]) -> None:
    """Make montage diagnostics describe output coordinates, not the source timeline."""
    for fill in fills:
        windows = list(fill.get("speech_windows", []))
        total_duration = sum(float(row.get("duration", 0.0) or 0.0) for row in windows)
        covered_duration = 0.0
        covered_windows = 0
        for window in windows:
            window_id = str(window.get("id"))
            window_start = float(window.get("start", 0.0) or 0.0)
            window_end = float(window.get("end", window_start + float(window.get("duration", 0.0) or 0.0)) or window_start)
            intervals = []
            for mapping in mappings:
                if window_id not in {str(item) for item in mapping.get("alignment_source_window_ids", [])}:
                    continue
                mapping_start = float(mapping.get("destination_timestamp", 0.0) or 0.0)
                mapping_end = mapping_start + float(mapping.get("planned_render_duration", mapping.get("clip_trim_duration", 0.0)) or 0.0)
                left, right = max(window_start, mapping_start), min(window_end, mapping_end)
                if right > left:
                    intervals.append((left, right))
            intervals.sort()
            merged = []
            for left, right in intervals:
                if merged and left <= merged[-1][1]:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], right))
                else:
                    merged.append((left, right))
            window_coverage = sum(right - left for left, right in merged)
            covered_duration += window_coverage
            if window_coverage > 0:
                covered_windows += 1
        fill["speech_window_count"] = len(windows)
        fill["covered_speech_window_count"] = covered_windows
        fill["uncovered_speech_window_count"] = max(0, len(windows) - covered_windows)
        fill["coverage"] = round(min(1.0, covered_duration / total_duration), 4) if total_duration else 0.0


def build_montage_render_acceptance(
    *,
    plan: dict[str, Any],
    encoded_probe: dict[str, Any],
    output_path: Path,
    timing_tolerance_seconds: float = 0.05,
) -> dict[str, Any]:
    """Record encoded conformance without promoting an experimental montage."""
    streams = encoded_probe.get("streams", [])
    encoded_duration = round(float((encoded_probe.get("format") or {}).get("duration") or 0.0), 3)
    planned_duration = round(float(plan.get("actual_duration") or 0.0), 3)
    selected = list(plan.get("selected_moments", []))
    provenance_complete = bool(selected) and all(
        row.get("source_media_hash")
        and row.get("shot_ids")
        and isinstance(row.get("visual_boundary"), dict)
        and isinstance(row.get("audio_boundary"), dict)
        for row in selected
    )
    checks = {
        "encoded_duration_matches_plan": abs(encoded_duration - planned_duration) <= timing_tolerance_seconds,
        "video_stream_present": any(row.get("codec_type") == "video" for row in streams),
        "audio_stream_present": any(row.get("codec_type") == "audio" for row in streams),
        "complete_selected_moment_provenance": provenance_complete,
        "minimum_moment_count_or_fallback_recorded": len(selected) >= int(plan.get("requested_minimum_moments", 1)) or any(
            row.get("action") == "RELAXED_MINIMUM_MOMENT_COUNT" for row in plan.get("fallback_decisions", [])
        ),
        "actual_audio_method_declared": bool((plan.get("laws") or {}).get("actual_audio_method")),
        "destination_intro_non_privilege_declared": (plan.get("opening_selection") or {}).get("timeline_position_primary_tiebreaker") is False,
        "available_material_substantially_utilized": bool((plan.get("material_utilization") or {}).get("utilization_sufficient")),
    }
    artifact = {
        "schema_version": "1.0",
        "creation_timestamp": utc_now(),
        "planner_version": plan.get("planner_version"),
        "filter_id": plan.get("filter_id"),
        "plan_verdict": plan.get("verdict", "EXPERIMENTAL"),
        "acceptance_status": "PASS" if all(checks.values()) else "FAIL",
        "timing_tolerance_seconds": timing_tolerance_seconds,
        "planned_duration": planned_duration,
        "encoded_duration": encoded_duration,
        "selected_moment_ids": [str(row["id"]) for row in selected],
        "checks": checks,
        "fallback_decisions": list(plan.get("fallback_decisions", [])),
        "placement_qualification": dict(plan.get("placement_qualification", {})),
        "laws": dict(plan.get("laws", {})),
        "provenance": {
            "source_media_hashes": sorted({str(row["source_media_hash"]) for row in selected if row.get("source_media_hash")}),
            "shot_ids": sorted({str(shot_id) for row in selected for shot_id in row.get("shot_ids", [])}),
            "plan_signature": stable_hash({
                "planner_version": plan.get("planner_version"),
                "random_seed": plan.get("random_seed"),
                "selected_moment_ids": [str(row["id"]) for row in selected],
                "laws": plan.get("laws"),
                "fallback_decisions": plan.get("fallback_decisions"),
            }),
        },
    }
    write_json(output_path, artifact)
    return artifact


def _moment_from_group(
    *,
    source_id: str,
    source_media_hash: str,
    shots: list[dict[str, Any]],
    speech: list[dict[str, Any]],
    transitions: list[dict[str, Any]],
    boundary_stability: list[dict[str, Any]],
) -> CinematicMoment:
    start = round(float(shots[0]["start"]), 3)
    end = round(float(shots[-1]["end"]), 3)
    shot_ids = tuple(str(row["id"]) for row in shots)
    scene_ids = {_scene_id(row) for row in shots}
    scene_id = next(iter(scene_ids)) if len(scene_ids) == 1 else "scene_grouped"
    assertions = [
        _assertion("SHOT_SEQUENCE_PRESERVED", True, source_id, start, end, confidence=1.0),
        _assertion("SPEECH_ENDED_BEFORE_BOUNDARY", not _interval_crosses(end, speech), source_id, start, end),
        _assertion("TRANSITION_COMPLETE", not _interval_crosses(start, transitions) and not _interval_crosses(end, transitions), source_id, start, end),
    ]
    virtual_boundaries = []
    if shots[0].get("_safe_virtual_start"):
        virtual_boundaries.append(float(shots[0]["start"]))
    if shots[-1].get("_safe_virtual_end"):
        virtual_boundaries.append(float(shots[-1]["end"]))
    for boundary in virtual_boundaries:
        assertions.extend([
            _assertion("SILENCE_WINDOW_PRESENT", True, source_id, start, end, note=f"Virtual boundary {boundary:.3f}s lies inside a measured speech-free window."),
            _assertion("LOW_FRAME_DIFFERENCE_AT_BOUNDARY", True, source_id, start, end, note=f"Virtual boundary {boundary:.3f}s lies inside sustained low frame-difference evidence."),
            _assertion("VIRTUAL_BOUNDARY_CORE_SUPPORTED", True, source_id, start, end, note="Core subdivision requires both literal silence and stillness evidence."),
        ])
    for label, boundary in (("ENTRANCE", start), ("EXIT", end)):
        row = _boundary_stability_row(boundary, boundary_stability)
        if row is not None:
            assertions.append(_assertion(
                f"{label}_{row.get('evidence_name', 'BOUNDARY_STABILITY')}",
                row.get("low_boundary_motion") if row.get("low_boundary_motion") is not None else "UNAVAILABLE",
                source_id,
                start,
                end,
                confidence=float(row.get("confidence", 0.0) or 0.0),
                capability_tag=CapabilityTag(str(row.get("capability_tag") or CapabilityTag.FALLBACK_INFERENCE.value)),
                fallback=row.get("status") != "AVAILABLE",
                note=f"Boundary {boundary:.3f}s; frame-difference evidence only, not a completed-action claim.",
            ))
    fallback = "PRESERVED_COMPLETE_LONG_TAKE" if len(shots) == 1 and end - start > 12.0 and not virtual_boundaries else "NONE"
    if fallback != "NONE":
        assertions.append(
            _assertion(
                "NO_SAFE_INTERNAL_BOUNDARY_FOUND",
                True,
                source_id,
                start,
                end,
                capability_tag=CapabilityTag.FALLBACK_INFERENCE,
                fallback=True,
                note="Core received no supported internal boundary and preserved the complete shot.",
            )
        )
    return CinematicMoment(
        id=stable_moment_id(source_media_hash=source_media_hash, start=start, end=end, shot_ids=shot_ids),
        source_id=source_id,
        source_media_hash=source_media_hash,
        scene_id=scene_id,
        shot_ids=shot_ids,
        start=start,
        end=end,
        visual_start=start,
        visual_end=end,
        audio_start=start,
        audio_end=end,
        assertions=tuple(assertions),
        fallback_status=fallback,
    )


def _assertion(
    name: str,
    value: bool | float | str,
    source_id: str,
    start: float,
    end: float,
    *,
    confidence: float = 1.0,
    capability_tag: CapabilityTag = CapabilityTag.CORE_HEURISTIC,
    fallback: bool = False,
    note: str = "",
) -> EvidenceAssertion:
    assertion_id = f"assertion_{stable_hash([name, source_id, start, end, value, capability_tag.value])[:20]}"
    return EvidenceAssertion(
        id=assertion_id,
        name=name,
        capability_tag=capability_tag,
        value=value,
        confidence=confidence,
        backend="core",
        backend_version=CORE_HEURISTIC_VERSION,
        source_artifact=source_id,
        fallback=fallback,
        note=note,
    )


def _planner_assertion(name: str, value: bool | float | str, moment: dict[str, Any], *, note: str = "") -> dict[str, Any]:
    return EvidenceAssertion(
        id=f"assertion_{stable_hash([name, moment['id'], value, MONTAGE_PLANNER_VERSION])[:20]}",
        name=name,
        capability_tag=CapabilityTag.PLANNER_DERIVATION,
        value=value,
        confidence=1.0,
        backend="montage_planner",
        backend_version=MONTAGE_PLANNER_VERSION,
        source_artifact=str(moment["id"]),
        supporting_evidence_ids=tuple(row["id"] for row in moment.get("assertions", [])),
        note=note,
    ).to_dict()


def _evenly_sample(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if count >= len(rows):
        return list(rows)
    if count == 1:
        return [rows[len(rows) // 2]]
    indices = [round(index * (len(rows) - 1) / (count - 1)) for index in range(count)]
    return [rows[index] for index in indices]


def _select_for_target_duration(
    rows: list[dict[str, Any]],
    *,
    target_duration: float,
    minimum_moments: int,
    maximum_moments: int,
    random_seed: int,
) -> list[dict[str, Any]]:
    """Seeded breadth-first packing that never trims a safe moment to hit runtime."""
    ordered = sorted(rows, key=lambda row: stable_hash([random_seed, row["id"]]))
    preferred_max_duration = target_duration / max(1, minimum_moments)
    authored = [row for row in ordered if int(row.get("authored_placement_count", 0) or 0) > 0]
    contextual = [row for row in ordered if int(row.get("authored_placement_count", 0) or 0) <= 0]
    authored_preferred = [row for row in authored if float(row["duration"]) <= preferred_max_duration + 0.001]
    authored_remainder = [row for row in authored if float(row["duration"]) > preferred_max_duration + 0.001]
    context_preferred = [row for row in contextual if float(row["duration"]) <= preferred_max_duration + 0.001]
    context_remainder = [row for row in contextual if float(row["duration"]) > preferred_max_duration + 0.001]
    selected: list[dict[str, Any]] = []
    total = 0.0
    for row in [*authored_preferred, *authored_remainder, *context_preferred, *context_remainder]:
        if len(selected) >= maximum_moments:
            break
        duration = float(row["duration"])
        if total + duration <= target_duration + 0.001:
            selected.append(row)
            total += duration
    if not selected and rows:
        fallback_rows = authored or rows
        selected.append(min(fallback_rows, key=lambda row: (float(row["duration"]), stable_hash([random_seed, row["id"]]))))
    selected.sort(key=lambda row: stable_hash([random_seed, row["id"]]))
    return selected


def _structural_roles(count: int) -> list[StructuralRole]:
    if count <= 0:
        return []
    if count == 1:
        return [StructuralRole.RESOLUTION]
    if count == 2:
        return [StructuralRole.BEGINNING, StructuralRole.RESOLUTION]
    roles = [StructuralRole.DEVELOPMENT] * count
    roles[0] = StructuralRole.BEGINNING
    roles[-2] = StructuralRole.CLIMAX
    roles[-1] = StructuralRole.RESOLUTION
    return roles


def _interval_crosses(boundary: float, intervals: list[dict[str, Any]]) -> bool:
    return any(float(row["start"]) < boundary < float(row["end"]) for row in intervals)


def _silence_intervals(speech: list[dict[str, Any]], *, duration: float, guard: float = 0.15) -> list[dict[str, float]]:
    merged: list[list[float]] = []
    for row in sorted(speech, key=lambda item: float(item["start"])):
        start = max(0.0, float(row["start"]) - guard)
        end = min(duration, float(row["end"]) + guard)
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    silence = []
    cursor = 0.0
    for start, end in merged:
        if start > cursor:
            silence.append({"start": round(cursor, 3), "end": round(start, 3)})
        cursor = max(cursor, end)
    if cursor < duration:
        silence.append({"start": round(cursor, 3), "end": round(duration, 3)})
    return silence


def _subdivide_supported_long_takes(
    shots: list[dict[str, Any]],
    *,
    silence: list[dict[str, float]],
    stillness: list[dict[str, Any]],
    transitions: list[dict[str, Any]],
    long_take_threshold: float = 12.0,
    minimum_segment: float = 4.0,
    minimum_overlap: float = 0.5,
) -> list[dict[str, Any]]:
    expanded: list[dict[str, Any]] = []
    for shot in shots:
        shot_start, shot_end = float(shot["start"]), float(shot["end"])
        if shot_end - shot_start <= long_take_threshold:
            expanded.append(shot)
            continue
        candidates = []
        for quiet in silence:
            for still in stillness:
                overlap_start = max(shot_start + minimum_segment, float(quiet["start"]), float(still["start"]))
                overlap_end = min(shot_end - minimum_segment, float(quiet["end"]), float(still["end"]))
                if overlap_end - overlap_start < minimum_overlap:
                    continue
                boundary = round((overlap_start + overlap_end) / 2.0, 3)
                if not _interval_crosses(boundary, transitions):
                    candidates.append(boundary)
        boundaries = []
        cursor = shot_start
        for boundary in sorted(set(candidates)):
            if boundary - cursor >= minimum_segment and shot_end - boundary >= minimum_segment:
                boundaries.append(boundary)
                cursor = boundary
        points = [shot_start, *boundaries, shot_end]
        if len(points) == 2:
            expanded.append(shot)
            continue
        for index, (start, end) in enumerate(zip(points, points[1:])):
            segment = dict(shot)
            segment.update({"start": start, "end": end, "duration": round(end - start, 3)})
            segment["_source_shot_id"] = str(shot["id"])
            segment["_safe_virtual_start"] = index > 0
            segment["_safe_virtual_end"] = index < len(points) - 2
            expanded.append(segment)
    return expanded


def _boundary_stability_row(boundary: float, rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    return next((row for row in rows if abs(float(row.get("boundary", -1.0)) - boundary) <= 0.001), None)


def _boundary_low_motion(boundary: float, rows: list[dict[str, Any]]) -> bool | None:
    row = _boundary_stability_row(boundary, rows)
    if row is None:
        return None
    value = row.get("low_boundary_motion")
    return value if isinstance(value, bool) else None


def _scene_id(shot: dict[str, Any]) -> str:
    return str(shot.get("scene_id") or shot.get("sequence_id") or shot["id"])
