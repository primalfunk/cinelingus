from __future__ import annotations

from enum import IntEnum
from typing import Any

from .speakers import speaker_map_has_real_diarization, speaker_map_identity_ready


TRUST_POLICY_VERSION = "analysis_trust_v1"


class AnalysisTrust(IntEnum):
    FAILED = 0
    WEAK = 1
    INFERRED = 2
    DIRECT = 3

    @property
    def label(self) -> str:
        return self.name.lower()


class AnalysisCapabilityError(RuntimeError):
    pass


def speaker_map_trust(speaker_map: dict[str, Any]) -> AnalysisTrust:
    if speaker_map_identity_ready(speaker_map):
        return AnalysisTrust.DIRECT
    if speaker_map_has_real_diarization(speaker_map):
        return AnalysisTrust.INFERRED
    diagnostics = dict(speaker_map.get("diagnostics") or {})
    status = str(diagnostics.get("status") or diagnostics.get("diarization_status") or "").upper()
    has_assignments = bool(speaker_map.get("speaker_segments")) or int(
        diagnostics.get("labeled_item_count") or 0
    ) > 0
    if has_assignments or status in {"FALLBACK", "PARTIAL", "WEAK"}:
        return AnalysisTrust.WEAK
    return AnalysisTrust.FAILED


def require_speaker_map_trust(speaker_map: dict[str, Any], required: str) -> AnalysisTrust:
    trust = speaker_map_trust(speaker_map)
    minimum = {
        "required": AnalysisTrust.DIRECT,
        "optional": AnalysisTrust.WEAK,
        "none": AnalysisTrust.FAILED,
    }.get(str(required).lower(), AnalysisTrust.DIRECT)
    if trust < minimum:
        raise AnalysisCapabilityError(
            f"Speaker identity requires {minimum.label} evidence; cached or generated analysis is {trust.label}."
        )
    return trust


def speaker_cache_signature_payload(config: Any) -> dict[str, Any]:
    """Facts that determine whether speaker evidence is reusable for this run."""
    return {
        "trust_policy_version": TRUST_POLICY_VERSION,
        "speaker_awareness_enabled": bool(config.enable_speaker_awareness),
        "backend": str(config.speaker_diarization_backend),
        "model": str(config.speaker_diarization_model),
        "device": str(config.speaker_diarization_device),
    }

