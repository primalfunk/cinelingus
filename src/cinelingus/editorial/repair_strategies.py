from __future__ import annotations

from typing import Any


STRATEGIES: dict[str, dict[str, Any]] = {
    "incomplete_sentence": {
        "strategy": "repair_sentence_boundaries", "families": ["source_boundary_extension", "destination_boundary_adjustment", "alternative_donor"],
        "hard_constraints": ["complete_transcript", "safe_source_handles", "bounded_time_adaptation"], "scoring_focus": ["sentence_fit", "timing_and_render_fit"],
        "maximum_attempts": 4, "fallback": "retain_best_known", "verification": ["word_coverage", "sentence_beginning", "sentence_ending"],
    },
    "mid_word_cut": {
        "strategy": "repair_word_boundary", "families": ["source_boundary_extension", "destination_boundary_adjustment", "alternative_donor"],
        "hard_constraints": ["safe_word_boundary_margin", "no_new_overlap"], "scoring_focus": ["sentence_fit"],
        "maximum_attempts": 4, "fallback": "suppress", "verification": ["mid_word_cut", "word_coverage"],
    },
    "low_rendered_coverage": {
        "strategy": "repair_rendered_coverage", "families": ["source_boundary_extension", "time_adaptation", "audio_treatment", "alternative_donor"],
        "hard_constraints": ["bounded_time_adaptation", "duration_fit"], "scoring_focus": ["sentence_fit", "timing_and_render_fit", "confidence"],
        "maximum_attempts": 5, "fallback": "retain_best_known", "verification": ["word_coverage", "masking", "confidence"],
    },
    "duration_failure": {
        "strategy": "repair_duration_fit", "families": ["time_adaptation", "destination_boundary_adjustment", "alternative_donor"],
        "hard_constraints": ["tight_duration_band", "verified_destination_headroom"], "scoring_focus": ["timing_and_render_fit"],
        "maximum_attempts": 4, "fallback": "retain_best_known", "verification": ["rendered_duration", "word_coverage"],
    },
    "speaker_mismatch": {
        "strategy": "repair_speaker_role", "families": ["same_performance_reassignment", "alternative_donor"],
        "hard_constraints": ["speaker_role_fit", "turn_order"], "scoring_focus": ["speaker_role_fit"],
        "maximum_attempts": 5, "fallback": "retain_best_known", "verification": ["speaker_role_fit", "participant_balance"],
    },
    "visual_mismatch": {
        "strategy": "repair_visual_intent", "families": ["same_performance_reassignment", "alternative_donor", "suppression"],
        "hard_constraints": ["visual_intent_fit", "action_dialogue_compatibility"], "scoring_focus": ["visual_fit"],
        "maximum_attempts": 5, "fallback": "suppress", "verification": ["visual_fit", "mouth_activity", "action_conflict"],
    },
    "transition_artifact": {
        "strategy": "repair_transition_edges", "families": ["audio_edge_adjustment", "timing_shift", "alternative_donor"],
        "hard_constraints": ["no_new_overlap", "bounded_shift"], "scoring_focus": ["transition_cleanliness"],
        "maximum_attempts": 3, "fallback": "retain_best_known", "verification": ["fade_masking", "transition_overlap"],
    },
    "residual_dialogue": {
        "strategy": "repair_local_suppression", "families": ["suppression_expansion", "ambience_rebuild"],
        "hard_constraints": ["local_only", "inserted_dialogue_unchanged"], "scoring_focus": ["residue_clearance"],
        "maximum_attempts": 3, "fallback": "suppress", "verification": ["residue", "masking"],
    },
    "masking": {
        "strategy": "repair_audio_masking", "families": ["audio_treatment", "suppression_expansion", "alternative_donor"],
        "hard_constraints": ["bounded_gain", "bounded_equalization"], "scoring_focus": ["intelligibility", "confidence"],
        "maximum_attempts": 3, "fallback": "retain_best_known", "verification": ["word_coverage", "masking", "confidence"],
    },
    "performance_mismatch": {
        "strategy": "repair_performance_structure", "families": ["same_performance_reassignment", "alternative_performance", "coordinated_neighborhood"],
        "hard_constraints": ["turn_pattern", "pause_structure", "energy_continuity"], "scoring_focus": ["performance_fit"],
        "maximum_attempts": 5, "fallback": "retain_best_known", "verification": ["performance_fit", "turn_pattern"],
    },
    "reuse_exhaustion": {
        "strategy": "repair_reuse_pressure", "families": ["controlled_reuse", "alternative_performance"],
        "hard_constraints": ["declared_reuse", "minimum_reuse_distance"], "scoring_focus": ["reuse_integrity"],
        "maximum_attempts": 3, "fallback": "retain_best_known", "verification": ["reuse_distance", "quality_delta"],
    },
    "confidence_collapse": {
        "strategy": "conservative_uncertainty_retention", "families": ["retain_best_known"],
        "hard_constraints": ["positive_verified_delta"], "scoring_focus": ["confidence"],
        "maximum_attempts": 1, "fallback": "retain_best_known", "verification": ["evidence_source", "confidence"],
    },
}


PRIORITY = (
    "residual_dialogue", "mid_word_cut", "incomplete_sentence", "low_rendered_coverage",
    "confidence_collapse", "masking", "duration_failure", "transition_artifact",
    "speaker_mismatch", "performance_mismatch", "visual_mismatch", "reuse_exhaustion",
)


def repair_strategy_for(decision: dict[str, Any]) -> dict[str, Any]:
    categories = {str(row.get("category")) for row in decision.get("failures", [])}
    target = str(decision.get("target_failure_category") or "")
    category = target if target in categories and target in STRATEGIES else next(
        (value for value in PRIORITY if value in categories), None
    )
    if category is None:
        return {
            "failure_category": "uncategorized", "strategy": "generic_donor_reassignment",
            "families": ["alternative_donor"], "hard_constraints": ["positive_quality_ceiling"],
            "scoring_focus": ["overall_quality"], "maximum_attempts": 3,
            "fallback": "retain_best_known", "verification": ["quality_delta"],
        }
    return {"failure_category": category, **STRATEGIES[category]}
