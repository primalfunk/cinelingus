from cinelingus.residue import build_residue_correction_regions, evaluate_voice_residue


def _schedule() -> dict:
    return {
        "destination_speech_regions": [{
            "id": "destination_line", "start": 1.0, "end": 3.0, "duration": 2.0,
            "transcript": "bring the blue lantern downstairs",
        }],
        "mappings": [{
            "enabled": True, "destination_timestamp": 1.0, "planned_render_duration": 2.0,
            "source_transcript": "close every window before morning",
        }],
    }


def test_residue_verifier_flags_destination_words_that_survive_the_render() -> None:
    result = evaluate_voice_residue(
        schedule=_schedule(),
        rendered_timeline={"windows": [{
            "start": 1.0, "end": 3.0, "transcript": "bring the blue lantern downstairs",
        }]},
    )

    assert result["status"] == "POSSIBLE_DESTINATION_SPEECH_DETECTED"
    assert result["flagged_region_count"] == 1


def test_residue_verifier_accepts_intended_donor_words() -> None:
    result = evaluate_voice_residue(
        schedule=_schedule(),
        rendered_timeline={"windows": [{
            "start": 1.0, "end": 3.0, "transcript": "close every window before morning",
        }]},
    )

    assert result["status"] == "NONE_DETECTED"
    assert result["regions"][0]["donor_similarity"] == 1.0


def test_residue_verifier_is_inconclusive_without_distinctive_destination_text() -> None:
    schedule = _schedule()
    schedule["destination_speech_regions"][0]["transcript"] = "I am here"

    result = evaluate_voice_residue(
        schedule=schedule,
        rendered_timeline={"windows": [{"start": 1.0, "end": 3.0, "transcript": "I am here"}]},
    )

    assert result["status"] == "INCONCLUSIVE"


def test_residue_verifier_checks_suppressed_region_without_donor_mapping() -> None:
    schedule = _schedule()
    schedule["mappings"] = []

    result = evaluate_voice_residue(
        schedule=schedule,
        rendered_timeline={"windows": [{
            "start": 1.0, "end": 3.0, "transcript": "bring the blue lantern downstairs", "confidence": 0.9,
        }]},
    )

    assert result["status"] == "POSSIBLE_DESTINATION_SPEECH_DETECTED"
    assert result["regions"][0]["evidence_kind"] == "unexpected_destination_like_speech_in_suppressed_region"
    assert result["regions"][0]["replacement_mapping_count"] == 0


def test_residue_verifier_reports_unrelated_unattributed_speech_without_correcting_it() -> None:
    schedule = _schedule()
    schedule["mappings"] = []

    result = evaluate_voice_residue(
        schedule=schedule,
        rendered_timeline={"windows": [{
            "start": 1.0, "end": 3.0, "transcript": "thanks for watching this channel", "confidence": 0.9,
        }]},
    )

    assert result["status"] == "NONE_DETECTED"
    assert result["flagged_region_count"] == 0
    assert result["unattributed_speech_region_count"] == 1
    assert result["regions"][0]["evidence_kind"] == "unattributed_speech_without_destination_match"


def test_residue_correction_regions_are_padded_and_bounded() -> None:
    regions = build_residue_correction_regions(
        {
            "regions": [{
                "destination_region_id": "line", "start": 0.05, "end": 1.9,
                "possible_residue": True, "evidence_kind": "destination_transcript_contrast",
            }]
        },
        padding=0.12,
        duration=2.0,
    )

    assert regions == [{
        "id": "residue_correction_000001",
        "start": 0.0,
        "end": 2.0,
        "duration": 2.0,
        "confidence": 0.95,
        "source_kind": "post_render_residue_correction",
        "recovered": True,
        "evidence_kind": "destination_transcript_contrast",
        "destination_region_id": "line",
    }]
