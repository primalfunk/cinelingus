from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EditorialMemory:
    rejected_pairs: set[tuple[str, str]] = field(default_factory=set)
    failure_history: list[dict[str, Any]] = field(default_factory=list)
    duration_failures: set[tuple[str, str]] = field(default_factory=set)
    sentence_failures: set[tuple[str, str]] = field(default_factory=set)
    speaker_failures: set[tuple[str, str]] = field(default_factory=set)

    def remember(self, decision: dict[str, Any], *, clip_id: str | None = None) -> None:
        key = str(decision.get("placement_key"))
        donor = str(clip_id or decision.get("clip_id") or "")
        pair = (key, donor)
        if donor:
            self.rejected_pairs.add(pair)
        categories = {row["category"] for row in decision.get("failures", [])}
        if "duration_failure" in categories:
            self.duration_failures.add(pair)
        if {"incomplete_sentence", "mid_word_cut", "low_rendered_coverage"} & categories:
            self.sentence_failures.add(pair)
        if "speaker_mismatch" in categories:
            self.speaker_failures.add(pair)
        self.failure_history.append({
            "placement_key": key, "clip_id": donor,
            "categories": sorted(categories), "quality": decision.get("overall_quality"),
        })

    def rejected(self, placement_key: str, clip_id: str) -> bool:
        return (str(placement_key), str(clip_id)) in self.rejected_pairs

    def to_dict(self) -> dict[str, Any]:
        def pairs(values: set[tuple[str, str]]) -> list[dict[str, str]]:
            return [{"placement_key": left, "clip_id": right} for left, right in sorted(values)]
        return {
            "rejected_pairs": pairs(self.rejected_pairs),
            "duration_failures": pairs(self.duration_failures),
            "sentence_failures": pairs(self.sentence_failures),
            "speaker_failures": pairs(self.speaker_failures),
            "failure_history": list(self.failure_history),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> "EditorialMemory":
        data = dict(value or {})

        def pairs(name: str) -> set[tuple[str, str]]:
            return {
                (str(row.get("placement_key") or ""), str(row.get("clip_id") or ""))
                for row in data.get(name, [])
                if row.get("placement_key") is not None
            }

        return cls(
            rejected_pairs=pairs("rejected_pairs"),
            duration_failures=pairs("duration_failures"),
            sentence_failures=pairs("sentence_failures"),
            speaker_failures=pairs("speaker_failures"),
            failure_history=list(data.get("failure_history", [])),
        )
