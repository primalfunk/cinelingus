from movie_masher.operator_language import (
    LEGACY_TRANSPOSITION,
    MAJOR_STAGE_KEYS,
    TRANSPOSITION,
    display_mode_name,
    internal_mode_name,
    journal_messages_for_lines,
    migrate_mode_value,
    operator_message_for_log,
    operator_text_is_backend_free,
    stage_key_for_diagnostic,
    stage_message,
)


def test_movie_masher_displays_as_transposition_and_internal_alias_survives() -> None:
    assert display_mode_name("Movie Masher") == TRANSPOSITION
    assert display_mode_name("movie_masher") == TRANSPOSITION
    assert internal_mode_name(TRANSPOSITION) == LEGACY_TRANSPOSITION


def test_legacy_mode_values_migrate_with_an_explicit_note() -> None:
    value, note = migrate_mode_value("Movie Masher")

    assert value == TRANSPOSITION
    assert note == "Movie Masher migrated to Transposition"


def test_every_major_stage_has_backend_free_operator_language() -> None:
    assert MAJOR_STAGE_KEYS
    for key in MAJOR_STAGE_KEYS:
        message = stage_message(key)
        assert message.title
        assert message.message
        assert operator_text_is_backend_free(message)


def test_technical_backend_names_remain_in_diagnostic_detail() -> None:
    technical = "Whisper transcription started with model=small, device=cuda."
    message = operator_message_for_log(technical)

    assert message is not None
    assert operator_text_is_backend_free(message)
    assert message.diagnostic_detail == technical


def test_timeout_and_fallback_are_truthful_and_visible() -> None:
    timeout = operator_message_for_log("Pyannote inference timeout after 780 seconds")
    fallback = operator_message_for_log("speaker diarization fallback to heuristic labels")

    assert timeout is not None and timeout.severity == "warning"
    assert "allotted observation period" in timeout.message
    assert "Pyannote" in timeout.diagnostic_detail
    assert fallback is not None and fallback.severity == "warning"
    assert "alternate method" in fallback.message


def test_speaker_validation_does_not_claim_final_artifact_validation():
    assert stage_key_for_diagnostic("validation_errors: []") is None
    assert stage_key_for_diagnostic("validating destination speakers") is None
    assert stage_key_for_diagnostic("validating final artifact") == "finalize"


def test_cache_messages_can_be_deduplicated_by_event_id() -> None:
    first = operator_message_for_log("reused media inspection for destination")
    second = operator_message_for_log("reused source dialogue events")

    assert first is not None and second is not None
    assert first.event_id == second.event_id == "cache_recovered"


def test_journal_milestones_are_ordered_and_heartbeats_do_not_flood() -> None:
    messages = journal_messages_for_lines([
        "inspecting destination media",
        "[heartbeat] inspecting destination media",
        "reused media inspection for destination",
        "reused source dialogue events",
        "performance grouping complete",
        "scheduling mappings",
        "muxing final video",
    ])

    assert [row.event_id for row in messages] == [
        "inspect_media",
        "cache_recovered",
        "performance_grouping",
        "scheduling",
        "muxing",
    ]
