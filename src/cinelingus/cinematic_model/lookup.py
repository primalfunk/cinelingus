from __future__ import annotations

from typing import Any

ENTITY_COLLECTIONS = (
    "shots", "transitions", "speech_passages", "speaker_clusters", "dialogue_turns",
    "performances", "cinematic_moments", "editorial_observations", "provenance", "source_artifacts",
)
ID_KEYS = (
    "shot_id", "transition_id", "speech_passage_id", "speaker_cluster_id", "dialogue_turn_id",
    "performance_id", "cinematic_moment_id", "editorial_observation_id", "provenance_id", "source_artifact_id",
)
COLLECTION_ID_KEYS = {
    "shots": "shot_id", "transitions": "transition_id", "speech_passages": "speech_passage_id",
    "speaker_clusters": "speaker_cluster_id", "dialogue_turns": "dialogue_turn_id",
    "performances": "performance_id", "cinematic_moments": "cinematic_moment_id",
    "editorial_observations": "editorial_observation_id", "provenance": "provenance_id",
    "source_artifacts": "source_artifact_id",
}


class FilmModelView:
    """Minimal read-only lookup surface for a canonical FilmModel dictionary."""

    def __init__(self, model: dict[str, Any]) -> None:
        self._model = model
        self._objects: dict[str, dict[str, Any]] = {}
        self._collections: dict[str, str] = {}
        for collection in ENTITY_COLLECTIONS:
            for row in model.get(collection) or []:
                id_key = COLLECTION_ID_KEYS[collection]
                entity_id = str(row[id_key]) if id_key in row else None
                if entity_id:
                    self._objects[entity_id] = row
                    self._collections[entity_id] = collection

    def get(self, entity_id: str) -> dict[str, Any]:
        return self._objects[entity_id]

    def list(self, object_type: str) -> tuple[dict[str, Any], ...]:
        collection = self._collection_name(object_type)
        return tuple(self._model.get(collection) or [])

    def overlapping(self, object_type: str, start: float, end: float) -> tuple[dict[str, Any], ...]:
        if end < start:
            raise ValueError("Lookup interval end must not precede start")
        return tuple(
            row for row in self.list(object_type)
            if isinstance(row.get("start"), (int, float)) and isinstance(row.get("end"), (int, float))
            and float(row["start"]) < end and start < float(row["end"])
        )

    def source_artifact_for(self, entity_id: str) -> dict[str, Any] | None:
        row = self.get(entity_id)
        if self._collections[entity_id] == "source_artifacts":
            return row
        provenance_id = row.get("provenance_id")
        if not provenance_id:
            return None
        provenance = self._objects.get(str(provenance_id), {})
        artifact_id = provenance.get("source_artifact_id")
        return self._objects.get(str(artifact_id)) if artifact_id else None

    def provenance_chain(self, entity_id: str) -> tuple[dict[str, Any], ...]:
        row = self.get(entity_id)
        first = entity_id if self._collections[entity_id] == "provenance" else row.get("provenance_id")
        ordered: list[dict[str, Any]] = []
        visited: set[str] = set()

        def visit(provenance_id: str | None) -> None:
            if not provenance_id or provenance_id in visited:
                return
            visited.add(provenance_id)
            provenance = self._objects.get(provenance_id)
            if not provenance:
                return
            ordered.append(provenance)
            for parent in provenance.get("parent_provenance_ids") or []:
                visit(str(parent))

        visit(str(first) if first else None)
        return tuple(ordered)

    def confidence(self, entity_id: str, field: str = "confidence") -> dict[str, Any] | None:
        value = self.get(entity_id).get(field)
        return value if isinstance(value, dict) else None

    def linked_objects(self, entity_id: str) -> tuple[dict[str, Any], ...]:
        row = self.get(entity_id)
        linked: set[str] = set()
        for key, value in row.items():
            if not (key.endswith("_id") or key.endswith("_ids") or key.endswith("_reference") or key.endswith("_references") or key.endswith("_candidates")):
                continue
            values = value if isinstance(value, list) else [value]
            linked.update(str(item) for item in values if item is not None and str(item) in self._objects)
        return tuple(self._objects[item] for item in sorted(linked))

    def performances_containing_turn(self, turn_id: str) -> tuple[dict[str, Any], ...]:
        return tuple(row for row in self.list("performances") if turn_id in (row.get("dialogue_turn_references") or []))

    def shots_intersecting_performance(self, performance_id: str) -> tuple[dict[str, Any], ...]:
        performance = self.get(performance_id)
        return self.overlapping("shots", float(performance["start"]), float(performance["end"]))

    def moments_containing_interval(self, start: float, end: float) -> tuple[dict[str, Any], ...]:
        return tuple(row for row in self.list("cinematic_moments") if float(row["start"]) <= start and float(row["end"]) >= end)

    def editorial_observations_for_placement(self, placement_id: str) -> tuple[dict[str, Any], ...]:
        return tuple(row for row in self.list("editorial_observations") if row.get("referenced_placement_id") == placement_id)

    def capability_status(self, capability: str) -> dict[str, Any]:
        return self._model["capabilities"][capability]

    @staticmethod
    def _collection_name(object_type: str) -> str:
        aliases = {
            "shot": "shots", "transition": "transitions", "speech": "speech_passages",
            "speaker": "speaker_clusters", "turn": "dialogue_turns", "performance": "performances",
            "moment": "cinematic_moments", "editorial": "editorial_observations",
            "artifact": "source_artifacts",
        }
        collection = aliases.get(object_type, object_type)
        if collection not in ENTITY_COLLECTIONS:
            raise KeyError(f"Unknown FilmModel object type: {object_type}")
        return collection
