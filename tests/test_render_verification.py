from cinelingus.render_verification import evaluate_rendered_dialogue


def test_render_verification_detects_missing_sentence_edges_and_word_loss() -> None:
    result = evaluate_rendered_dialogue(
        schedule={"mappings": [{
            "enabled": True, "clip_id": "c1", "destination_timestamp": 1.0,
            "planned_render_duration": 3.0, "source_transcript": "Bring the blue lantern downstairs now.",
            "render_operations": [{"operation": "fade_in_out"}],
        }]},
        rendered_timeline={"windows": [{
            "start": 1.0, "end": 4.0, "transcript": "blue lantern downstairs", "confidence": 0.8,
        }]},
    )

    row = result["mappings"][0]
    assert result["status"] == "FAIL"
    assert row["missing_sentence_beginning"] is True
    assert row["missing_sentence_ending"] is True
    assert row["word_coverage_percentage"] == 50.0
    assert row["fade_masking_possible"] is True


def test_render_verification_accepts_complete_rendered_line() -> None:
    result = evaluate_rendered_dialogue(
        schedule={"mappings": [{
            "enabled": True, "clip_id": "c1", "destination_timestamp": 1.0,
            "planned_render_duration": 3.0, "source_transcript": "Bring the blue lantern downstairs.",
        }]},
        rendered_timeline={"windows": [{
            "start": 1.0, "end": 4.0, "transcript": "Bring the blue lantern downstairs.", "confidence": 0.9,
        }]},
    )

    assert result["status"] == "PASS"
    assert result["average_word_coverage_percentage"] == 100.0
    assert all(label["status"] == "pass" for label in result["mappings"][0]["labels"])


def test_neighboring_whisper_words_are_overlap_warning_not_false_truncation() -> None:
    result = evaluate_rendered_dialogue(
        schedule={"mappings": [{
            "enabled": True, "clip_id": "c1", "destination_timestamp": 1.0,
            "planned_render_duration": 2.0, "source_transcript": "No problem.",
        }]},
        rendered_timeline={"windows": [{
            "start": 0.5, "end": 3.5, "transcript": "Outta here! No problem. Keep moving.",
            "confidence": 0.9,
        }]},
    )

    row = result["mappings"][0]
    assert row["word_coverage_percentage"] == 100.0
    assert row["missing_sentence_beginning"] is False
    assert row["missing_sentence_ending"] is False
    assert row["adjacent_dialogue_before"] is True
    assert row["adjacent_dialogue_after"] is True
    assert row["status"] == "warning"


def test_expressive_interjection_elongation_is_not_false_word_loss() -> None:
    result = evaluate_rendered_dialogue(
        schedule={"mappings": [{
            "enabled": True, "clip_id": "c1", "destination_timestamp": 1.0,
            "planned_render_duration": 1.0, "source_transcript": "Ah!",
        }]},
        rendered_timeline={"windows": [{
            "start": 1.0, "end": 2.0, "transcript": "AAAAAAAH!", "confidence": 0.9,
        }]},
    )

    assert result["status"] == "PASS"
    assert result["mappings"][0]["word_coverage_percentage"] == 100.0
