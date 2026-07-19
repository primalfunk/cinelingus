from cinelingus.clips import coalesce_dialogue_events


def test_coalesce_dialogue_events_joins_incomplete_transcript_fragments() -> None:
    events = [
        {
            "id": "e1",
            "start": 65.0,
            "end": 67.0,
            "duration": 2.0,
            "transcript": "You know, they all kind of look up to me around here",
            "speaker_id": "speaker_003",
            "speaker": "speaker_003",
            "confidence": 0.9,
        },
        {
            "id": "e2",
            "start": 67.0,
            "end": 69.0,
            "duration": 2.0,
            "transcript": "for advice and riding lessons.",
            "speaker_id": "speaker_002",
            "speaker": "speaker_002",
            "confidence": 0.8,
        },
    ]

    utterances = coalesce_dialogue_events(events)

    assert len(utterances) == 1
    assert utterances[0]["duration"] == 4.0
    assert utterances[0]["event_ids"] == ["e1", "e2"]
    assert utterances[0]["transcript"] == "You know, they all kind of look up to me around here for advice and riding lessons."


def test_coalesce_dialogue_events_keeps_complete_sentences_separate() -> None:
    events = [
        {"id": "e1", "start": 0.0, "end": 2.0, "transcript": "You work here?"},
        {"id": "e2", "start": 2.0, "end": 4.0, "transcript": "Senior staff?"},
    ]

    assert len(coalesce_dialogue_events(events)) == 2
