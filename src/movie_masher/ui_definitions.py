from __future__ import annotations


QUALITY_DEFINITIONS = {
    "Preview": "Uses a faster examination suitable for exploratory work, with reduced transcription precision.",
    "Balanced": "Balances transcription fidelity and processing time for routine experiments.",
    "Precision": "Uses a more exacting examination at substantially greater processing cost.",
    "Fast Preview": "Legacy name for Preview fidelity.",
    "High Accuracy": "Legacy name for Precision fidelity.",
}

TRANSFORMATION_DEFINITIONS = {
    "Transposition": "Transfers spoken performances from one film into another.",
    "Movie Masher": "Legacy name for Transposition.",
    "Self Shuffle": "Reassigns one film's own dialogue to different speaking intervals while requiring changed line placement.",
    "Echo": "Repeats selected dialogue at configured later positions over the original film.",
    "Drift": "Moves dialogue progressively away from its original timing while preserving the picture.",
}

WORKFLOW_DEFINITIONS = {
    "Best Short Remix": "Scores candidate performances, then assembles the strongest compatible moments near the requested length.",
    "Full Movie Remix": "Processes detected speaking portions across the complete destination rather than selecting a short reel.",
}

PREFERENCE_DEFINITIONS = {
    "Balanced": "Balances timing fit, dialogue density, speaker pattern, energy, reuse, and editorial contrast.",
    "Funniest result": "Prefers surprising mismatches, energetic exchanges, contrast, and absurd line substitutions. It does not understand humor like a person; these measurable proxies raise the score.",
    "Funniest": "Prefers surprising mismatches, energetic exchanges, contrast, and absurd line substitutions. It does not understand humor like a person; these measurable proxies raise the score.",
    "Best realism": "Prioritizes similar duration, turn rhythm, speaker pattern, dialogue density, and minimal time-stretching.",
    "Most Surprising": "Prioritizes contrast in performance type, energy, and dialogue context while retaining basic timing safety.",
}

MATCHING_DEFINITIONS = {
    "Balanced": "Uses the baseline weighted mix of duration, rhythm, speaker pattern, density, energy, and reuse.",
    "Rhythmic": "Gives more weight to turn duration, pauses, response timing, and overall conversational cadence.",
    "High Energy": "Favors dense, fast-moving, higher-energy dialogue with less dead air.",
    "Deadpan": "Favors slower delivery and pause-heavy exchanges.",
    "Contrast": "Rewards source and destination performances that differ in energy or dramatic character.",
    "Low Repetition": "Penalizes reuse more strongly and searches farther for distinct donor material.",
    "Surreal": "Rewards unusual contrast and energetic mismatch while relaxing conventional similarity.",
}


def setting_definition(group: str, value: str) -> str:
    tables = {
        "transformation": TRANSFORMATION_DEFINITIONS,
        "quality": QUALITY_DEFINITIONS,
        "workflow": WORKFLOW_DEFINITIONS,
        "preference": PREFERENCE_DEFINITIONS,
        "matching": MATCHING_DEFINITIONS,
    }
    return tables.get(group, {}).get(value, "This option uses the pipeline's documented default scoring behavior.")
