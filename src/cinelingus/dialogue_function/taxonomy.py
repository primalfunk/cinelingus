from __future__ import annotations

from pathlib import Path
from typing import Any

from ..util import read_json, stable_hash

TAXONOMY_VERSION = "dialogue_function_taxonomy_v1"
TAXONOMY_PATH = Path(__file__).with_name("taxonomy_v1.json")
AXES = ("surface_form", "interaction_function", "sequence_position")


def load_taxonomy(path: Path | None = None) -> dict[str, Any]:
    taxonomy = read_json(path or TAXONOMY_PATH)
    validate_taxonomy(taxonomy)
    return taxonomy


def validate_taxonomy(taxonomy: dict[str, Any]) -> dict[str, Any]:
    if taxonomy.get("taxonomy_version") != TAXONOMY_VERSION:
        raise ValueError(f"Unsupported dialogue-function taxonomy: {taxonomy.get('taxonomy_version')}")
    axes = taxonomy.get("axes") or {}
    if set(axes) != set(AXES):
        raise ValueError("Dialogue-function taxonomy must define exactly the three contracted axes")
    seen: set[str] = set()
    for axis in AXES:
        labels = axes[axis].get("labels") or []
        if not labels:
            raise ValueError(f"Taxonomy axis {axis} has no labels")
        for label in labels:
            required = {"label_id", "name", "definition", "inclusion_rules", "exclusion_rules", "positive_examples", "counterexamples"}
            missing = required - set(label)
            if missing:
                raise ValueError(f"Taxonomy label {axis} is missing {sorted(missing)}")
            label_id = str(label["label_id"])
            if label_id in seen or not label_id.startswith(f"{axis}."):
                raise ValueError(f"Invalid or duplicate label ID: {label_id}")
            seen.add(label_id)
    if not taxonomy.get("ambiguity_rules") or not taxonomy.get("abstention_rules"):
        raise ValueError("Taxonomy must define ambiguity and abstention rules")
    if not taxonomy.get("migration_policy"):
        raise ValueError("Taxonomy must define a migration policy")
    return {"status": "VALID", "taxonomy_version": TAXONOMY_VERSION, "taxonomy_signature": stable_hash(taxonomy), "label_count": len(seen)}
