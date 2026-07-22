from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

from ..util import stable_hash, write_json
from .builder import BuildResult
from .cache import storage_footprint
from .serialization import canonical_json, write_film_model

SEMANTIC_LIMITATION = (
    "This model does not contain semantic scene understanding, character identity, "
    "active-speaker recognition, dialogue-function classification, relationship "
    "inference, or narrative meaning."
)


def render_model_report(model: dict[str, Any], *, storage_bytes: int | None = None) -> str:
    counts = {
        "Shots": len(model.get("shots") or []), "Transitions": len(model.get("transitions") or []),
        "Speech passages": len(model.get("speech_passages") or []), "Speaker clusters": len(model.get("speaker_clusters") or []),
        "Dialogue turns": len(model.get("dialogue_turns") or []), "Performances": len(model.get("performances") or []),
        "Cinematic moments": len(model.get("cinematic_moments") or []), "Editorial observations": len(model.get("editorial_observations") or []),
    }
    capability_lines = [f"  {name}: {row['status']}" for name, row in sorted((model.get("capabilities") or {}).items())]
    fallback_count = sum(1 for row in (model.get("capabilities") or {}).values() if row.get("status") == "FALLBACK")
    warnings = model.get("validation_state", {}).get("warnings") or []
    lines = [
        "CINELINGUS FILM MODEL REPORT", "=" * 29, "",
        f"Film ID: {model.get('film_id')}", f"Media hash: {model.get('media', {}).get('media_hash')}",
        f"Filename: {model.get('media', {}).get('filename')}", f"Duration: {float(model.get('timeline', {}).get('duration', 0.0)):.3f} seconds",
        f"Schema version: {model.get('schema_version')}", f"Builder version: {model.get('builder_version')}",
        f"Cache signature: {model.get('created_from_signature')}", f"Validation: {model.get('validation_state', {}).get('status')}",
        f"Schedule-trace readiness: {'AVAILABLE' if model.get('capabilities', {}).get('schedule_provenance', {}).get('status') == 'AVAILABLE' else 'NOT READY'}",
        f"Storage footprint: {storage_bytes if storage_bytes is not None else 'not measured'} bytes", "",
        "OBJECT COUNTS", "-------------",
        *[f"{name}: {value}" for name, value in counts.items()], "",
        f"Source artifacts: {len(model.get('source_artifacts') or [])}",
        f"Confidence coverage: {model.get('confidence_summary', {}).get('records_by_state', {})}",
        f"Fallback capabilities: {fallback_count}", f"Validation warnings: {len(warnings)}", "",
        "CAPABILITIES", "------------", *capability_lines, "", "LIMITATIONS", "-----------", SEMANTIC_LIMITATION, "",
    ]
    return "\n".join(lines)


def write_model_bundle(output_dir: Path, result: BuildResult) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "model": output_dir / "film_model.json", "validation": output_dir / "validation_report.json",
        "migration": output_dir / "migration_report.json", "build": output_dir / "build_report.json",
        "report": output_dir / "model_report.txt", "manifest": output_dir / "artifact_manifest.json",
    }
    write_film_model(paths["model"], result.model)
    write_json(paths["validation"], result.validation_report)
    write_json(paths["migration"], result.migration_report)
    write_json(paths["build"], result.build_report)
    report_text = render_model_report(result.model)
    _write_text_atomic(paths["report"], report_text)
    manifest = {
        "schema_version": "1.0", "film_id": result.model["film_id"],
        "created_from_signature": result.model["created_from_signature"],
        "canonical_model_signature": stable_hash(result.model),
        "artifacts": {name: {"filename": path.name, "content_signature": stable_hash(path.read_text(encoding="utf-8"))} for name, path in paths.items() if name != "manifest"},
    }
    write_json(paths["manifest"], manifest)
    report_text = render_model_report(result.model, storage_bytes=storage_footprint(output_dir))
    _write_text_atomic(paths["report"], report_text)
    manifest["artifacts"]["report"]["content_signature"] = stable_hash(report_text)
    write_json(paths["manifest"], manifest)
    return paths


def compare_models(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    left_json, right_json = canonical_json(left), canonical_json(right)
    fields = ("schema_version", "builder_version", "film_id", "created_from_signature")
    differences = [{"field": field, "left": left.get(field), "right": right.get(field)} for field in fields if left.get(field) != right.get(field)]
    return {
        "equivalent": left_json == right_json, "canonical_left_signature": stable_hash(left),
        "canonical_right_signature": stable_hash(right), "top_level_identity_differences": differences,
    }


def write_model_report(path: Path, model: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_text_atomic(path, render_model_report(model, storage_bytes=storage_footprint(path.parent)))


def _write_text_atomic(path: Path, text: str) -> None:
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
