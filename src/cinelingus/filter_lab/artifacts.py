from __future__ import annotations

from pathlib import Path
from typing import Any

from cinelingus.util import read_json, utc_now, write_json


def materialize_required_artifacts(
    *, pipeline: Any, required_artifacts: tuple[str, ...], output_dir: Path, schedule: dict[str, Any]
) -> dict[str, Path]:
    """Create standardized filter-facing views only when the selected filter requires them."""
    paths: dict[str, Path] = {}
    required = set(required_artifacts)
    if "dialogue_events" in required:
        path = pipeline.source.cache_dir / "dialogue_events.json"
        if path.exists():
            paths["dialogue_events"] = path
    if "performances" in required:
        source_path = pipeline.source.cache_dir / "performance.json"
        destination_path = pipeline.destination.cache_dir / "performance.json"
        available = [path for path in (source_path, destination_path) if path.exists()]
        if available:
            output_path = output_dir / "performances.json"
            write_json(output_path, {
                "schema_version": "1.0", "creation_timestamp": utc_now(),
                "roles": {"source": read_json(source_path) if source_path.exists() else None, "destination": read_json(destination_path) if destination_path.exists() else None},
            })
            paths["performances"] = output_path
    if "speakers" in required:
        speaker_path = pipeline.destination.cache_dir / "speaker_map.json"
        if speaker_path.exists():
            speaker_map = read_json(speaker_path)
            output_path = output_dir / "speakers.json"
            write_json(output_path, {
                "schema_version": "1.0", "creation_timestamp": utc_now(),
                "media_hash": speaker_map.get("media_hash"), "requested_backend": speaker_map.get("requested_backend"),
                "actual_backend": speaker_map.get("actual_backend"), "speaker_count": speaker_map.get("speaker_count", 0),
                "speakers": [
                    {
                        "speaker_id": item.get("speaker_id"), "total_speaking_time": item.get("total_duration", 0.0),
                        "event_count": item.get("event_count", 0), "dialogue_diversity": item.get("dialogue_diversity"),
                        "confidence": item.get("confidence"), "first_seen": item.get("first_seen"), "last_seen": item.get("last_seen"),
                    }
                    for item in speaker_map.get("speakers", [])
                ],
            })
            paths["speakers"] = output_path
    if "scenes" in required:
        performance_path = pipeline.destination.cache_dir / "performance.json"
        if performance_path.exists():
            performance = read_json(performance_path)
            output_path = output_dir / "scenes.json"
            write_json(output_path, {
                "schema_version": "1.0", "creation_timestamp": utc_now(),
                "media_hash": performance.get("media_hash"),
                "scenes": [
                    {
                        "scene_id": item.get("id"), "start": item.get("start"), "end": item.get("end"),
                        "duration": item.get("duration"), "participating_speakers": sorted(set(item.get("speaker_sequence", []))),
                        "dialogue_density": item.get("dialogue_density"), "conversational_transitions": max(0, len(item.get("speaker_sequence", [])) - 1),
                    }
                    for item in performance.get("performances", [])
                ],
            })
            paths["scenes"] = output_path
    if "shots" in required:
        path = pipeline.destination.cache_dir / "shots.json"
        if path.exists():
            paths["shots"] = path
    if "speaker_graph" in required and schedule.get("speaker_graph"):
        output_path = output_dir / "speaker_graph.json"
        write_json(output_path, {"schema_version": "1.0", "creation_timestamp": utc_now(), "speaker_graph": schedule["speaker_graph"]})
        paths["speaker_graph"] = output_path
    multiworld = schedule.get("multiworld_artifacts", {})
    for artifact_name in ("film_inspections", "shared_timeline", "world_model"):
        if artifact_name in required and artifact_name in multiworld:
            output_path = output_dir / f"{artifact_name}.json"
            write_json(output_path, {
                "schema_version": "1.0",
                "creation_timestamp": utc_now(),
                artifact_name: multiworld[artifact_name],
            })
            paths[artifact_name] = output_path
    return paths
