from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

SEMANTIC_LIMITATION = (
    "This subsystem measures transcript-vector similarity only. It does not understand dialogue function, "
    "intention, emotion, character, relationship, scene meaning, narrative purpose, irony, or comedy."
)


def render_semantic_report(bundle: dict[str, Any], cache_report: dict[str, Any] | None = None) -> str:
    coverage = bundle.get("coverage") or {}
    provider = bundle.get("provider_metadata") or {}
    counts = coverage.get("status_counts") or {}
    language_counts: dict[str, int] = {}
    for row in bundle.get("entities") or []:
        language = str(row.get("language_state") or "unknown")
        language_counts[language] = language_counts.get(language, 0) + 1
    lines = [
        f"CINELINGUS SEMANTIC {str(bundle.get('entity_type', 'speech_passage')).replace('_', ' ').upper()} REPORT", "=" * 35, "",
        f"Film ID: {bundle.get('film_id')}",
        f"FilmModel signature: {bundle.get('film_model_signature')}",
        f"Construction state: {bundle.get('construction_state')}",
        f"Schema / builder: {bundle.get('schema_version')} / {bundle.get('builder_version')}",
        f"Provider: {provider.get('provider')}", f"Model: {provider.get('model_id')}",
        f"Revision: {provider.get('model_revision')}", f"Tokenizer: {provider.get('tokenizer_id')}",
        f"Dimensions / precision: {provider.get('dimensions')} / {provider.get('precision')}",
        f"Device: {provider.get('execution_device')}", f"Token limit: {provider.get('token_limit')}",
        f"Pooling / normalization: {provider.get('pooling_policy')} / {provider.get('normalization')}", "",
        "COVERAGE", "--------", f"Source entities: {coverage.get('source_entity_count', 0)}",
        f"Accounted entities: {coverage.get('accounted_entity_count', 0)}",
        *[f"{name}: {value}" for name, value in sorted(counts.items())],
        f"Languages: {dict(sorted(language_counts.items()))}", "",
    ]
    exclusions = bundle.get("structural_exclusions") or []
    if exclusions:
        lines.extend(["STRUCTURAL EXCLUSIONS", "---------------------", f"Excluded structures: {len(exclusions)}", *[
            f"{row.get('source_entity_type')} {row.get('source_entity_id')}: {', '.join(row.get('reasons') or [])}"
            for row in exclusions
        ], ""])
    if cache_report:
        lines.extend([
            "CACHE", "-----", f"Cache hits: {cache_report.get('cache_hits', 0)}",
            f"Entities encoded: {cache_report.get('entities_encoded', 0)}",
            f"Resume used: {cache_report.get('resume_used', False)}", "",
        ])
    lines.extend(["LIMITATION", "----------", SEMANTIC_LIMITATION, ""])
    return "\n".join(lines)


def write_semantic_report(path: Path, bundle: dict[str, Any], cache_report: dict[str, Any] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = render_semantic_report(bundle, cache_report)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
