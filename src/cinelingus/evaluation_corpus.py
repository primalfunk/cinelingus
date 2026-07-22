from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Callable

from . import __version__
from .cache import media_hash
from .tools import ToolError, ffprobe_json
from .util import read_json, stable_hash, utc_now, write_json


VIDEO_EXTENSIONS = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".webm"}
ANIMATION_HINTS = (
    "cartoon", "death note", "gummi", "little bear", "mega man", "mulan", "olive",
    "speed racer", "magic schoolbus", "little mermaid", "sword in the stone", "wizard of oz",
    "wallace and gromit", "fox and", "hey duggee",
)
MIXED_HINTS = ("fraggle", "muppet", "faerie tale", "fairie tale")


def build_corpus_manifest(
    *,
    source_root: Path,
    output_path: Path,
    inventory_cache_path: Path,
    pipeline_cache_root: Path | None = None,
    max_files: int | None = None,
    refresh: bool = False,
    probe_media: Callable[[Path], dict[str, Any]] = ffprobe_json,
    hash_media: Callable[[Path], str] = media_hash,
) -> dict[str, Any]:
    """Inventory a read-only media corpus while writing only to workspace-owned paths."""
    source_root = source_root.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    inventory_cache_path = inventory_cache_path.expanduser().resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"Movie corpus does not exist: {source_root}")
    _require_outside_source(source_root, output_path)
    _require_outside_source(source_root, inventory_cache_path)
    files = sorted((path for path in source_root.rglob("*") if path.is_file()), key=lambda path: path.relative_to(source_root).as_posix().casefold())
    if max_files is not None:
        files = files[: max(0, int(max_files))]
    cache = read_json(inventory_cache_path) if inventory_cache_path.exists() and not refresh else {"entries": {}}
    cached_entries = dict(cache.get("entries") or {})
    next_cache: dict[str, Any] = {}
    media_rows: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    for path in files:
        relative = path.relative_to(source_root).as_posix()
        stat = path.stat()
        if path.suffix.casefold() not in VIDEO_EXTENSIONS:
            exclusions.append({
                "path": relative, "extension": path.suffix.casefold(),
                "reason": "unsupported_non_video_media", "size_bytes": stat.st_size,
            })
            continue
        fingerprint = {"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        cached = dict(cached_entries.get(relative) or {})
        if cached.get("fingerprint") == fingerprint and cached.get("media_hash") and cached.get("probe"):
            digest = str(cached["media_hash"])
            probe = dict(cached["probe"])
            inventory_source = "inventory_cache"
        else:
            try:
                digest = hash_media(path)
                probe = probe_media(path)
            except (OSError, ValueError, ToolError) as exc:
                exclusions.append({
                    "path": relative, "extension": path.suffix.casefold(),
                    "reason": "ingestion_or_probe_failure", "detail": f"{type(exc).__name__}: {exc}",
                    "size_bytes": stat.st_size,
                })
                continue
            inventory_source = "fresh_probe"
        next_cache[relative] = {"fingerprint": fingerprint, "media_hash": digest, "probe": probe}
        media_rows.append(_manifest_media_row(
            path=path, relative=relative, digest=digest, probe=probe,
            pipeline_cache_root=pipeline_cache_root, inventory_source=inventory_source,
        ))
    write_json(inventory_cache_path, {
        "schema_version": "1.0", "source_root": str(source_root),
        "updated_timestamp": utc_now(), "entries": next_cache,
    })
    strata_counts = Counter(stratum for row in media_rows for stratum in row["evaluation_strata"])
    manifest = {
        "schema_version": "1.0",
        "manifest_version": "movie_corpus_manifest_v1",
        "tool_version": __version__,
        "creation_timestamp": utc_now(),
        "source_root": str(source_root),
        "source_policy": "read_only_no_sidecars",
        "deterministic_order": "relative_path_casefold_ascending",
        "compatible_file_count": len(media_rows),
        "excluded_file_count": len(exclusions),
        "total_compatible_duration_seconds": round(sum(float(row["duration_seconds"]) for row in media_rows), 3),
        "total_compatible_size_bytes": sum(int(row["size_bytes"]) for row in media_rows),
        "strata_counts": dict(sorted(strata_counts.items())),
        "media": media_rows,
        "exclusions": exclusions,
        "selection_contract": {
            "stable_media_id": "sha256_content_prefix_16",
            "required_reproduction_inputs": [
                "manifest_version", "media_ids", "pairing_rules", "configuration", "seed", "tool_version",
            ],
        },
    }
    write_json(output_path, manifest)
    return manifest


def build_evaluation_plan(
    *,
    manifest_path: Path,
    output_path: Path,
    tier: str,
    seed: int = 1,
    max_files: int | None = None,
    max_pairings: int | None = None,
    max_total_source_duration: float | None = None,
    max_rendered_duration: float | None = None,
    max_disk_bytes: int | None = None,
    max_runtime_seconds: float | None = None,
) -> dict[str, Any]:
    if tier not in {"smoke", "standard", "extended"}:
        raise ValueError(f"Unsupported corpus tier: {tier}")
    manifest = read_json(manifest_path)
    defaults = {
        "smoke": {"files": 4, "pairings": 3, "source_duration": 4 * 60 * 60, "rendered_duration": 12 * 60},
        "standard": {"files": 12, "pairings": 8, "source_duration": 14 * 60 * 60, "rendered_duration": 45 * 60},
        "extended": {"files": len(manifest.get("media", [])), "pairings": 24, "source_duration": float("inf"), "rendered_duration": 4 * 60 * 60},
    }[tier]
    limits = {
        "maximum_files": int(max_files if max_files is not None else defaults["files"]),
        "maximum_pairings": int(max_pairings if max_pairings is not None else defaults["pairings"]),
        "maximum_total_source_duration_seconds": float(max_total_source_duration if max_total_source_duration is not None else defaults["source_duration"]),
        "maximum_rendered_duration_seconds": float(max_rendered_duration if max_rendered_duration is not None else defaults["rendered_duration"]),
        "maximum_disk_bytes": int(max_disk_bytes) if max_disk_bytes is not None else None,
        "maximum_runtime_seconds": float(max_runtime_seconds) if max_runtime_seconds is not None else None,
    }
    ordered = sorted(
        manifest.get("media", []),
        key=lambda row: (stable_hash([seed, row.get("media_id")]), str(row.get("media_id"))),
    )
    selected = _stratified_selection(ordered, limits["maximum_files"], limits["maximum_total_source_duration_seconds"])
    pairings = _purposeful_pairings(selected, limits["maximum_pairings"])
    plan = {
        "schema_version": "1.0",
        "plan_version": "movie_corpus_evaluation_plan_v1",
        "tool_version": __version__,
        "creation_timestamp": utc_now(),
        "manifest": str(manifest_path.resolve()),
        "manifest_version": manifest.get("manifest_version"),
        "tier": tier,
        "seed": int(seed),
        "dry_run": True,
        "limits": limits,
        "selected_file_count": len(selected),
        "selected_pairing_count": len(pairings),
        "selected_source_duration_seconds": round(sum(float(row.get("duration_seconds", 0.0)) for row in selected), 3),
        "estimated_analysis_cached_count": sum(1 for row in selected if row.get("analysis", {}).get("available")),
        "expected_output_count": len(pairings),
        "selected_media": [
            {key: row.get(key) for key in ("media_id", "filename", "duration_seconds", "content_type", "duration_class", "evaluation_strata")}
            for row in selected
        ],
        "pairings": pairings,
        "execution_policy": {
            "source_files_read_only": True,
            "write_sidecars_to_source": False,
            "cache_reuse": True,
            "resume_required": True,
            "excerpt_policy": "analysis_selected_bounded_excerpt_when_available_otherwise_plan_only",
        },
    }
    write_json(output_path, plan)
    return plan


def build_excerpt_plan(
    *, manifest_path: Path, cache_root: Path, output_path: Path, tier: str,
    seed: int = 1, max_excerpts: int | None = None,
    max_total_duration: float | None = None,
) -> dict[str, Any]:
    """Select bounded, deterministic excerpts from existing local analysis evidence."""
    if tier not in {"smoke", "standard", "extended"}:
        raise ValueError(f"Unsupported corpus tier: {tier}")
    manifest = read_json(manifest_path)
    defaults = {
        "smoke": (7, 3 * 60.0), "standard": (18, 10 * 60.0), "extended": (48, 30 * 60.0),
    }[tier]
    limit = int(max_excerpts if max_excerpts is not None else defaults[0])
    duration_limit = float(max_total_duration if max_total_duration is not None else defaults[1])
    candidates: list[dict[str, Any]] = []
    for media in manifest.get("media", []):
        if not (media.get("analysis") or {}).get("available"):
            continue
        candidates.extend(_excerpt_candidates(media, cache_root))
    ordered = sorted(
        candidates,
        key=lambda row: (
            stable_hash([seed, row["category"], row["media_id"], row["start"], row["end"]]),
            row["media_id"], row["start"], row["category"],
        ),
    )
    selected: list[dict[str, Any]] = []
    used_regions: dict[str, list[tuple[float, float]]] = {}
    elapsed = 0.0
    categories = sorted({str(row["category"]) for row in ordered})
    for category in categories:
        candidate = next((row for row in ordered if row["category"] == category and _excerpt_is_available(row, used_regions)), None)
        if candidate and len(selected) < limit and elapsed + candidate["duration"] <= duration_limit:
            selected.append(candidate); elapsed += candidate["duration"]
            used_regions.setdefault(candidate["media_id"], []).append((candidate["start"], candidate["end"]))
    for candidate in ordered:
        if len(selected) >= limit:
            break
        if elapsed + candidate["duration"] > duration_limit or not _excerpt_is_available(candidate, used_regions):
            continue
        selected.append(candidate); elapsed += candidate["duration"]
        used_regions.setdefault(candidate["media_id"], []).append((candidate["start"], candidate["end"]))
    selected = sorted(selected, key=lambda row: (row["media_id"], row["start"], row["category"]))
    for index, row in enumerate(selected, start=1):
        row["excerpt_id"] = f"excerpt_{index:04d}_{stable_hash([row['media_id'], row['start'], row['end'], row['category']])[:8]}"
    counts = Counter(str(row["category"]) for row in selected)
    plan = {
        "schema_version": "1.0", "plan_version": "evidence_excerpt_plan_v1",
        "tool_version": __version__, "creation_timestamp": utc_now(),
        "manifest": str(manifest_path.resolve()), "manifest_version": manifest.get("manifest_version"),
        "tier": tier, "seed": int(seed), "dry_run": True,
        "cache_root": str(cache_root.resolve()),
        "limits": {"maximum_excerpts": limit, "maximum_total_duration_seconds": duration_limit},
        "candidate_count": len(candidates), "selected_excerpt_count": len(selected),
        "selected_total_duration_seconds": round(elapsed, 3),
        "category_counts": dict(sorted(counts.items())), "excerpts": selected,
        "execution_policy": {
            "source_files_read_only": True, "analysis_evidence_required": True,
            "exact_boundaries_required": True, "resume_required": True,
        },
    }
    write_json(output_path, plan)
    return plan


def _excerpt_candidates(media: dict[str, Any], cache_root: Path) -> list[dict[str, Any]]:
    root = cache_root.resolve() / str(media.get("media_hash"))
    if not root.is_dir():
        return []
    artifacts = sorted(root.rglob("*.json"), key=lambda path: path.as_posix().casefold())
    timeline = _first_json(artifacts, "filtered_timeline.json") or _first_json(artifacts, "timeline.json") or {}
    windows = list(timeline.get("windows") or [])
    performance = _first_json(artifacts, "performance.json") or {}
    performances = list(performance.get("performances") or [])
    shots_data = _first_json(artifacts, "shots.json") or {}
    transitions = list(shots_data.get("transitions") or [])
    duration = float(media.get("duration_seconds", 0.0) or 0.0)
    signature = str((media.get("analysis") or {}).get("analysis_cache_signature") or "")
    rows: list[dict[str, Any]] = []

    def add(category: str, start: float, end: float, reason: str, evidence: dict[str, Any]) -> None:
        start = max(0.0, min(duration, float(start)))
        end = max(start, min(duration, float(end)))
        if end - start < 2.0:
            return
        bounded_end = min(end, start + 30.0)
        rows.append({
            "media_id": media["media_id"], "filename": media.get("filename"),
            "source_path": media.get("source_path"), "content_type": media.get("content_type"),
            "duration_class": media.get("duration_class"), "category": category,
            "start": round(start, 3), "end": round(bounded_end, 3),
            "duration": round(bounded_end - start, 3), "selection_reason": reason,
            "analysis_signature": signature,
            "evidence": evidence,
            "expected_failure_modes": _excerpt_expected_failures(category),
        })

    if performances:
        dense = max(performances, key=lambda row: (float(row.get("dialogue_density", 0.0) or 0.0), float(row.get("duration", 0.0) or 0.0)))
        add("dense_dialogue", float(dense.get("start", 0.0)), float(dense.get("end", 0.0)), "highest analyzed performance dialogue density", _performance_evidence(dense))
        exchanges = [
            row for row in performances
            if int(row.get("estimated_turn_count", 0) or 0) >= 2
            and int(row.get("estimated_speaker_count", 0) or 0) >= 2
        ]
        if exchanges:
            rapid = max(
                exchanges,
                key=lambda row: (
                    float(row.get("estimated_turn_count", 0.0) or 0.0)
                    / max(1.0, float(row.get("duration", 0.0) or 0.0)),
                    float(row.get("estimated_turn_count", 0.0) or 0.0),
                ),
            )
            add(
                "rapid_speaker_exchange",
                float(rapid.get("start", 0.0)) - 4.0,
                float(rapid.get("end", 0.0)) + 8.0,
                "highest qualified multi-speaker turn rate with transcription handles",
                _performance_evidence(rapid),
            )
        monologues = [row for row in performances if int(row.get("estimated_speaker_count", 0) or 0) <= 1]
        if monologues:
            mono = max(monologues, key=lambda row: float(row.get("duration", 0.0) or 0.0))
            add("long_monologue", float(mono.get("start", 0.0)), float(mono.get("end", 0.0)), "longest single-speaker analyzed performance", _performance_evidence(mono))
    if windows:
        fragmented = max(
            windows,
            key=lambda row: (1.0 if float(row.get("duration", 0.0) or 0.0) <= 1.5 else 0.0, -float(row.get("duration", 0.0) or 0.0)),
        )
        add("short_fragmented_lines", float(fragmented.get("start", 0.0)) - 4.0, float(fragmented.get("end", 0.0)) + 8.0, "short analyzed speech window with contextual handles", _window_evidence(fragmented))
        gaps = []
        ordered_windows = sorted(windows, key=lambda row: float(row.get("start", 0.0) or 0.0))
        for left, right in zip(ordered_windows, ordered_windows[1:]):
            gap_start, gap_end = float(left.get("end", 0.0) or 0.0), float(right.get("start", 0.0) or 0.0)
            if gap_end - gap_start >= 4.0:
                gaps.append((gap_end - gap_start, gap_start, gap_end))
        if gaps:
            gap, gap_start, gap_end = max(gaps)
            add(
                "quiet_room_tone",
                max(gap_start, gap_end - 5.0),
                min(duration, gap_end + 15.0),
                "quiet-to-dialogue boundary after the longest analyzed non-speech gap",
                {"gap_seconds": round(gap, 3), "dialogue_resume": round(gap_end, 3)},
            )
    if transitions and windows:
        matches = []
        for transition in transitions:
            center = (float(transition.get("start", 0.0) or 0.0) + float(transition.get("end", 0.0) or 0.0)) / 2.0
            distance = min((abs(center - float(row.get("start", 0.0) or 0.0)) for row in windows), default=999999.0)
            matches.append((distance, center, transition))
        distance, center, transition = min(matches, key=lambda row: (row[0], row[1]))
        add("transition_near_dialogue", center - 7.5, center + 7.5, "closest detected transition to analyzed speech", {"transition_id": transition.get("id"), "transition_kind": transition.get("kind"), "speech_distance_seconds": round(distance, 3)})
    if media.get("content_type") == "animation" and performances:
        dense = max(performances, key=lambda row: float(row.get("dialogue_density", 0.0) or 0.0))
        add(
            "animation_dialogue",
            float(dense.get("start", 0.0)) - 4.0,
            float(dense.get("end", 0.0)) + 8.0,
            "animation-specific dialogue evidence with transcription handles",
            _performance_evidence(dense),
        )
    return rows


def _performance_evidence(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "performance_id": row.get("id"), "dialogue_density": row.get("dialogue_density"),
        "performance_duration": row.get("duration"),
        "turn_count": row.get("estimated_turn_count"), "speaker_count": row.get("estimated_speaker_count"),
        "performance_type": row.get("performance_type"), "interruptions_detected": row.get("interruptions_detected"),
    }


def _window_evidence(row: dict[str, Any]) -> dict[str, Any]:
    return {"window_id": row.get("id"), "duration": row.get("duration"), "confidence": row.get("confidence")}


def _excerpt_expected_failures(category: str) -> list[str]:
    return {
        "dense_dialogue": ["masking", "residual_dialogue", "low_rendered_coverage"],
        "rapid_speaker_exchange": ["speaker_mismatch", "performance_mismatch"],
        "long_monologue": ["duration_failure", "incomplete_sentence"],
        "short_fragmented_lines": ["mid_word_cut", "transition_artifact"],
        "quiet_room_tone": ["residual_dialogue", "masking"],
        "transition_near_dialogue": ["transition_artifact", "incomplete_sentence"],
        "animation_dialogue": ["visual_mismatch", "speaker_mismatch"],
    }.get(category, [])


def _excerpt_is_available(row: dict[str, Any], used: dict[str, list[tuple[float, float]]], overlap_ratio: float = 0.5) -> bool:
    duration = max(0.001, float(row["duration"]))
    for start, end in used.get(str(row["media_id"]), []):
        overlap = max(0.0, min(float(row["end"]), end) - max(float(row["start"]), start))
        if overlap / duration > overlap_ratio:
            return False
    return True


def _manifest_media_row(
    *, path: Path, relative: str, digest: str, probe: dict[str, Any],
    pipeline_cache_root: Path | None, inventory_source: str,
) -> dict[str, Any]:
    streams = list(probe.get("streams") or [])
    fmt = dict(probe.get("format") or {})
    videos = [row for row in streams if row.get("codec_type") == "video"]
    audios = [row for row in streams if row.get("codec_type") == "audio"]
    video = videos[0] if videos else {}
    duration = _number(fmt.get("duration"))
    width, height = video.get("width"), video.get("height")
    content_type, classification_method = _content_type(path.name)
    analysis = _analysis_status(digest, pipeline_cache_root, duration)
    duration_class = "short" if duration <= 15 * 60 else "episode" if duration <= 65 * 60 else "feature"
    strata = [
        {"animation": "animation_or_cartoons", "live_action": "live_action", "mixed_or_uncertain": "mixed_or_uncertain"}[content_type],
        f"{duration_class}_form",
    ]
    occupancy = analysis.get("dialogue_occupancy")
    if occupancy is not None:
        strata.append("dialogue_heavy" if occupancy >= 0.5 else "sparse_dialogue" if occupancy <= 0.15 else "moderate_dialogue")
    if not audios or not videos or duration <= 0:
        strata.append("technically_difficult")
    return {
        "media_id": f"media_{digest[:16]}",
        "filename": path.name,
        "relative_path": relative,
        "source_path": str(path.resolve()),
        "media_hash": digest,
        "size_bytes": path.stat().st_size,
        "duration_seconds": round(duration, 3),
        "resolution": f"{width}x{height}" if width and height else None,
        "frame_rate": _rate(video.get("avg_frame_rate")),
        "video_codec": video.get("codec_name"),
        "audio_stream_count": len(audios),
        "audio_streams": [
            {
                "codec": row.get("codec_name"), "sample_rate": _integer(row.get("sample_rate")),
                "channels": _integer(row.get("channels")), "channel_layout": row.get("channel_layout"),
                "bit_rate": _integer(row.get("bit_rate")),
            }
            for row in audios
        ],
        "content_type": content_type,
        "content_type_method": classification_method,
        "duration_class": duration_class,
        "evaluation_strata": sorted(set(strata)),
        "analysis": analysis,
        "basic_audio_quality": {
            "audio_present": bool(audios),
            "maximum_channels": max((_integer(row.get("channels")) or 0 for row in audios), default=0),
            "maximum_sample_rate": max((_integer(row.get("sample_rate")) or 0 for row in audios), default=0),
            "codec_set": sorted({str(row.get("codec_name")) for row in audios if row.get("codec_name")}),
        },
        "known_ingestion_problems": [],
        "inventory_source": inventory_source,
        "benchmark_roles": [],
    }


def _analysis_status(digest: str, pipeline_cache_root: Path | None, duration: float) -> dict[str, Any]:
    roots = []
    if pipeline_cache_root is not None:
        candidate = pipeline_cache_root.resolve() / digest
        if candidate.is_dir():
            roots.append(candidate)
    artifacts = sorted(path for root in roots for path in root.rglob("*.json") if path.is_file())
    names = sorted({path.name for path in artifacts})
    dialogue_rows = _first_rows(artifacts, ("filtered_dialogue_events.json", "dialogue_events.json", "filtered_timeline.json", "timeline.json"))
    occupancy = None
    if dialogue_rows is not None and duration > 0:
        occupancy = round(min(1.0, sum(max(0.0, _number(row.get("end")) - _number(row.get("start"))) for row in dialogue_rows) / duration), 4)
    shots = _first_json(artifacts, "shots.json")
    shot_count = len(shots.get("shots", [])) if shots is not None else None
    return {
        "available": bool(artifacts),
        "artifact_names": names,
        "analysis_cache_signature": stable_hash([(str(path), path.stat().st_size, path.stat().st_mtime_ns) for path in artifacts]) if artifacts else None,
        "dialogue_occupancy": occupancy,
        "detected_shot_count": shot_count,
    }


def _first_json(paths: list[Path], name: str) -> dict[str, Any] | None:
    path = next((row for row in paths if row.name == name), None)
    if path is None:
        return None
    try:
        return read_json(path)
    except (OSError, ValueError):
        return None


def _first_rows(paths: list[Path], names: tuple[str, ...]) -> list[dict[str, Any]] | None:
    for name in names:
        data = _first_json(paths, name)
        if data is not None:
            return list(data.get("events") or data.get("windows") or [])
    return None


def _stratified_selection(rows: list[dict[str, Any]], maximum: int, duration_limit: float) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    used: set[str] = set()
    elapsed = 0.0
    desired = ["animation_or_cartoons", "live_action", "short_form", "episode_form", "feature_form", "technically_difficult", "dialogue_heavy", "sparse_dialogue"]
    for stratum in desired:
        candidate = next((row for row in rows if stratum in row.get("evaluation_strata", []) and row["media_id"] not in used), None)
        if candidate is None:
            continue
        duration = float(candidate.get("duration_seconds", 0.0))
        if len(selected) < maximum and elapsed + duration <= duration_limit:
            selected.append(candidate); used.add(candidate["media_id"]); elapsed += duration
    for candidate in rows:
        duration = float(candidate.get("duration_seconds", 0.0))
        if len(selected) >= maximum:
            break
        if candidate["media_id"] not in used and elapsed + duration <= duration_limit:
            selected.append(candidate); used.add(candidate["media_id"]); elapsed += duration
    return selected


def _purposeful_pairings(rows: list[dict[str, Any]], maximum: int) -> list[dict[str, Any]]:
    if len(rows) < 2:
        return []
    pairs = []
    seen: set[tuple[str, str]] = set()
    usage: Counter[str] = Counter()
    goals = (
        ("animation_to_animation", lambda a, b: a["content_type"] == b["content_type"] == "animation"),
        ("live_action_to_live_action", lambda a, b: a["content_type"] == b["content_type"] == "live_action"),
        ("animation_to_live_action", lambda a, b: {a["content_type"], b["content_type"]} == {"animation", "live_action"}),
        ("intentional_duration_mismatch", lambda a, b: max(a["duration_seconds"], b["duration_seconds"]) >= 1.8 * max(1.0, min(a["duration_seconds"], b["duration_seconds"]))),
        ("similar_duration_dialogue_exchange", lambda a, b: abs(a["duration_seconds"] - b["duration_seconds"]) / max(a["duration_seconds"], b["duration_seconds"], 1.0) <= 0.2),
    )
    all_pairs = [(left, right) for left_index, left in enumerate(rows) for right in rows[left_index + 1:]]

    def unused_matches(predicate: Callable[[dict[str, Any], dict[str, Any]], bool]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
        matches = [
            (left, right) for left, right in all_pairs
            if (left["media_id"], right["media_id"]) not in seen and predicate(left, right)
        ]
        return sorted(
            matches,
            key=lambda pair: (
                max(usage[pair[0]["media_id"]], usage[pair[1]["media_id"]]),
                usage[pair[0]["media_id"]] + usage[pair[1]["media_id"]],
                pair[0]["media_id"], pair[1]["media_id"],
            ),
        )

    def append_pair(left: dict[str, Any], right: dict[str, Any], purpose: str) -> None:
        seen.add((left["media_id"], right["media_id"]))
        usage.update((left["media_id"], right["media_id"]))
        pairs.append(_pairing(len(pairs) + 1, left, right, purpose))

    for goal, predicate in goals:
        matches = unused_matches(predicate)
        if matches:
            append_pair(*matches[0], goal)
        if len(pairs) >= maximum:
            return pairs
    while len(pairs) < maximum:
        matches = unused_matches(lambda _left, _right: True)
        if not matches:
            break
        append_pair(*matches[0], "general_cross_stratum_regression")
    return pairs


def _pairing(index: int, source: dict[str, Any], destination: dict[str, Any], purpose: str) -> dict[str, Any]:
    return {
        "pairing_id": f"pair_{index:03d}", "purpose": purpose,
        "source_media_id": source["media_id"], "destination_media_id": destination["media_id"],
        "scope": {"mode": "analysis_selected_excerpt", "status": "pending_analysis_evidence"},
        "expected_failure_modes": _expected_failures(purpose),
    }


def _expected_failures(purpose: str) -> list[str]:
    return {
        "intentional_duration_mismatch": ["duration_failure", "low_rendered_coverage"],
        "animation_to_animation": ["visual_mismatch", "speaker_mismatch"],
        "animation_to_live_action": ["visual_mismatch", "performance_mismatch"],
        "similar_duration_dialogue_exchange": ["speaker_mismatch", "performance_mismatch", "transition_artifact"],
        "live_action_to_live_action": ["residual_dialogue", "masking", "speaker_mismatch"],
    }.get(purpose, ["incomplete_sentence", "masking", "performance_mismatch"])


def _content_type(filename: str) -> tuple[str, str]:
    lowered = filename.casefold()
    if any(value in lowered for value in ANIMATION_HINTS):
        return "animation", "local_filename_hint_low_confidence"
    if any(value in lowered for value in MIXED_HINTS):
        return "mixed_or_uncertain", "local_filename_hint_low_confidence"
    return "live_action", "provisional_non_animation_filename_classification"


def _require_outside_source(source_root: Path, path: Path) -> None:
    if path == source_root or source_root in path.parents:
        raise ValueError(f"Corpus artifacts must not be written inside the source folder: {path}")


def _number(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _integer(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _rate(value: Any) -> float | None:
    try:
        numerator, denominator = str(value).split("/", 1)
        return round(float(numerator) / float(denominator), 6) if float(denominator) else None
    except (TypeError, ValueError):
        return None
