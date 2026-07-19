from types import SimpleNamespace

import pytest

from cinelingus.analysis_trust import (
    AnalysisCapabilityError,
    AnalysisTrust,
    require_speaker_map_trust,
    speaker_cache_signature_payload,
    speaker_map_trust,
)


def _direct_map() -> dict:
    return {
        "actual_backend": "pyannote",
        "speaker_segments": [{"speaker_id": "speaker_0", "fallback_label": False}],
        "diagnostics": {"diarization_status": "SUCCESS", "direct_item_rate": 0.9},
    }


def test_direct_pyannote_map_satisfies_identity_requirement() -> None:
    assert speaker_map_trust(_direct_map()) == AnalysisTrust.DIRECT
    assert require_speaker_map_trust(_direct_map(), "required") == AnalysisTrust.DIRECT


def test_fallback_map_is_weak_and_rejected_for_identity_filter() -> None:
    fallback = {
        "actual_backend": "fallback",
        "speaker_segments": [{"speaker_id": "unknown_0", "fallback_label": True}],
        "diagnostics": {"status": "FALLBACK"},
    }
    assert speaker_map_trust(fallback) == AnalysisTrust.WEAK
    assert require_speaker_map_trust(fallback, "optional") == AnalysisTrust.WEAK
    with pytest.raises(AnalysisCapabilityError, match="requires direct"):
        require_speaker_map_trust(fallback, "required")


def test_cache_signature_captures_backend_model_device_and_policy() -> None:
    payload = speaker_cache_signature_payload(
        SimpleNamespace(
            enable_speaker_awareness=True,
            speaker_diarization_backend="pyannote",
            speaker_diarization_model="model-a",
            speaker_diarization_device="cuda",
        )
    )
    assert payload["backend"] == "pyannote"
    assert payload["model"] == "model-a"
    assert payload["device"] == "cuda"
    assert payload["trust_policy_version"] == "analysis_trust_v1"
