from __future__ import annotations

from pathlib import Path
from typing import Any
from copy import deepcopy
import tempfile
import wave

from .audio_provenance import analyze_wav_intervals
from .tools import ffprobe_json, run


def render_schedule_regions_over_original_audio(
    *,
    original_media: Path,
    schedule: dict[str, Any],
    regions: list[dict[str, Any]],
    duration: float,
    output_path: Path,
    sample_rate: int,
    channels: int,
    target_lufs: float,
    fade_duration: float,
    duck_db: float = -28.0,
    suppression_mode: str = "hard_mute",
    suppression_fade_duration: float = 0.05,
    background_reconstruction: str = "neighboring_non_speech_with_adaptive_crossfades",
    guard_seconds: float = 0.15,
) -> dict[str, Any]:
    """Patch only repaired timeline regions into an existing PCM WAV."""
    if not output_path.exists():
        raise FileNotFoundError(f"Incremental editorial rendering requires an existing WAV: {output_path}")
    merged_regions = _merge_duck_regions([
        {
            "start": max(0.0, float(row.get("start", 0.0) or 0.0) - guard_seconds),
            "duration": (
                min(float(duration), float(row.get("end", 0.0) or 0.0) + guard_seconds)
                - max(0.0, float(row.get("start", 0.0) or 0.0) - guard_seconds)
            ),
        }
        for row in regions
        if float(row.get("end", 0.0) or 0.0) > float(row.get("start", 0.0) or 0.0)
    ])
    normalized = [
        {
            "start": float(row["start"]),
            "end": float(row["start"]) + float(row["duration"]),
        }
        for row in merged_regions
    ]
    if not normalized:
        return {"strategy": "incremental_region_patch_v1", "region_count": 0, "regions": []}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="cinelingus_editorial_", dir=str(output_path.parent)) as temp_name:
        temp_dir = Path(temp_name)
        current = output_path
        for index, region in enumerate(normalized, start=1):
            start, end = float(region["start"]), float(region["end"])
            span = end - start
            carrier = temp_dir / f"carrier_{index:03d}.wav"
            patch = temp_dir / f"patch_{index:03d}.wav"
            merged = temp_dir / f"merged_{index:03d}.wav"
            run([
                "ffmpeg", "-y", "-ss", f"{start:.6f}", "-t", f"{span:.6f}",
                "-i", str(original_media), "-vn", "-ar", str(sample_rate), "-ac", str(channels),
                "-c:a", "pcm_s16le", str(carrier),
            ])
            localized = _localized_schedule(schedule, start=start, end=end)
            render_schedule_over_original_audio(
                original_media=carrier,
                schedule=localized,
                duration=span,
                output_path=patch,
                sample_rate=sample_rate,
                channels=channels,
                target_lufs=target_lufs,
                fade_duration=fade_duration,
                mute_regions=_mapping_suppression_regions(localized),
                duck_db=duck_db,
                suppression_mode=suppression_mode,
                suppression_fade_duration=suppression_fade_duration,
                background_reconstruction=background_reconstruction,
            )
            _replace_wav_region(
                source=current, patch=patch, output=merged, start=start, end=end,
                duration=duration, sample_rate=sample_rate, channels=channels,
            )
            current = merged
        current.replace(output_path)
    report = {
        "strategy": "incremental_region_patch_v1",
        "region_count": len(normalized),
        "guard_seconds": round(float(guard_seconds), 3),
        "regions": normalized,
        "full_timeline_rerendered": False,
    }
    schedule["editorial_incremental_render_report"] = report
    return report


def _localized_schedule(schedule: dict[str, Any], *, start: float, end: float) -> dict[str, Any]:
    localized = deepcopy(schedule)
    localized["mappings"] = []
    for mapping in schedule.get("mappings", []):
        item_start = float(mapping.get("destination_timestamp", 0.0) or 0.0)
        item_end = item_start + float(mapping.get("planned_render_duration", 0.0) or 0.0)
        if min(end, item_end) <= max(start, item_start):
            continue
        item = deepcopy(mapping)
        overlap_start = max(start, item_start)
        overlap_end = min(end, item_end)
        overlap_duration = max(0.0, overlap_end - overlap_start)
        stretch = max(0.001, float(item.get("stretch_factor", 1.0) or 1.0))
        elapsed_source = max(0.0, overlap_start - item_start) / stretch
        item["clip_trim_start"] = float(item.get("clip_trim_start", 0.0) or 0.0) + elapsed_source
        item["clip_trim_duration"] = overlap_duration / stretch
        item["planned_render_duration"] = overlap_duration
        item["destination_timestamp"] = overlap_start - start
        item["alignment_slot_start"] = (
            float(item["alignment_slot_start"]) - start
            if item.get("alignment_slot_start") is not None else item["destination_timestamp"]
        )
        item["alignment_slot_end"] = (
            float(item["alignment_slot_end"]) - start
            if item.get("alignment_slot_end") is not None else item["destination_timestamp"] + float(item.get("planned_render_duration", 0.0) or 0.0)
        )
        for operation in item.get("render_operations", []):
            if operation.get("operation") == "delay":
                operation["seconds"] = round(item["destination_timestamp"], 3)
        localized["mappings"].append(item)
    for key in ("destination_speech_regions", "residue_correction_regions"):
        localized[key] = [
            {
                **row,
                "start": max(0.0, float(row.get("start", 0.0) or 0.0) - start),
                "end": min(end, float(row.get("end", 0.0) or 0.0)) - start,
            }
            for row in schedule.get(key, [])
            if min(end, float(row.get("end", 0.0) or 0.0)) > max(start, float(row.get("start", 0.0) or 0.0))
        ]
    return localized


def _mapping_suppression_regions(schedule: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand only speech regions touched by an explicit local repair control."""
    mappings = [row for row in schedule.get("mappings", []) if row.get("enabled", True)]
    expanded = []
    for region in schedule.get("destination_speech_regions", []):
        start = float(region.get("start", 0.0) or 0.0)
        end = float(region.get("end", start) or start)
        touching = []
        for mapping in mappings:
            mapping_start = float(mapping.get("destination_timestamp", 0.0) or 0.0)
            mapping_end = mapping_start + float(mapping.get("planned_render_duration", 0.0) or 0.0)
            if min(end, mapping_end) > max(start, mapping_start):
                touching.append(mapping)
        leading = max((
            float(row.get("suppression_leading_padding", row.get("suppression_padding", 0.0)) or 0.0)
            for row in touching
        ), default=0.0)
        trailing = max((
            float(row.get("suppression_trailing_padding", row.get("suppression_padding", 0.0)) or 0.0)
            for row in touching
        ), default=0.0)
        expanded.append({**region, "start": max(0.0, start - leading), "end": end + trailing})
    return expanded


def _replace_wav_region(
    *, source: Path, patch: Path, output: Path, start: float, end: float,
    duration: float, sample_rate: int, channels: int,
) -> None:
    filters, labels = [], []
    if start > 0.0005:
        filters.append(f"[0:a]atrim=start=0:end={start:.6f},asetpts=PTS-STARTPTS[pre]")
        labels.append("[pre]")
    filters.append(f"[1:a]atrim=start=0:end={end - start:.6f},asetpts=PTS-STARTPTS[mid]")
    labels.append("[mid]")
    if end < duration - 0.0005:
        filters.append(f"[0:a]atrim=start={end:.6f}:end={duration:.6f},asetpts=PTS-STARTPTS[post]")
        labels.append("[post]")
    filters.append(f"{''.join(labels)}concat=n={len(labels)}:v=0:a=1[out]")
    run([
        "ffmpeg", "-y", "-i", str(source), "-i", str(patch),
        "-filter_complex", ";".join(filters), "-map", "[out]",
        "-ar", str(sample_rate), "-ac", str(channels), "-c:a", "pcm_s16le", str(output),
    ])


def render_montage_visual(*, input_video: Path, selected_moments: list[dict], output_path: Path) -> None:
    """Render exact plan boundaries with their source soundtrack as one continuous reel."""
    if not selected_moments:
        raise ValueError("Montage rendering requires at least one selected moment.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    filters = []
    labels = []
    for index, moment in enumerate(selected_moments):
        boundary = moment.get("visual_boundary")
        if not isinstance(boundary, dict):
            raise ValueError(f"Montage moment {moment.get('id')} is missing its visual_boundary contract.")
        start = float(boundary["start"])
        end = float(boundary["end"])
        if end <= start:
            raise ValueError(f"Montage moment {moment.get('id')} has invalid visual boundaries.")
        video_label = f"v{index}"
        audio_label = f"a{index}"
        filters.append(f"[0:v]trim=start={start:.3f}:end={end:.3f},setpts=PTS-STARTPTS[{video_label}]")
        filters.append(f"[0:a]atrim=start={start:.3f}:end={end:.3f},asetpts=PTS-STARTPTS[{audio_label}]")
        labels.append(f"[{video_label}][{audio_label}]")
    filters.append(f"{''.join(labels)}concat=n={len(labels)}:v=1:a=1[outv][outa]")
    run([
        "ffmpeg", "-y", "-i", str(input_video),
        "-filter_complex", ";".join(filters),
        "-map", "[outv]", "-map", "[outa]", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-c:a", "aac", "-b:a", "192k",
        str(output_path),
    ])


def render_multi_source_montage(*, selected_moments: list[dict], output_path: Path) -> None:
    """Render ordered picture and soundtrack segments from distinct complete media sources."""
    if not selected_moments:
        raise ValueError("Multi-source montage rendering requires at least one selected moment.")
    first_path = Path(str(selected_moments[0].get("source_path") or ""))
    if not first_path.exists():
        raise ValueError(f"Multi-source montage input does not exist: {first_path}")
    probe = ffprobe_json(first_path)
    video_stream = next((row for row in probe.get("streams", []) if row.get("codec_type") == "video"), None)
    if video_stream is None:
        raise ValueError(f"Multi-source montage input has no video stream: {first_path}")
    width = max(2, int(video_stream.get("width") or 1280))
    height = max(2, int(video_stream.get("height") or 720))
    width -= width % 2
    height -= height % 2
    args = ["ffmpeg", "-y"]
    filters: list[str] = []
    labels: list[str] = []
    for index, moment in enumerate(selected_moments):
        source_path = Path(str(moment.get("source_path") or ""))
        if not source_path.exists():
            raise ValueError(f"Multi-source montage input does not exist: {source_path}")
        boundary = moment.get("visual_boundary") or {
            "start": moment.get("source_start"), "end": moment.get("source_end")
        }
        if not isinstance(boundary, dict) or boundary.get("start") is None or boundary.get("end") is None:
            raise ValueError(f"Multi-source montage moment {moment.get('id')} has no source boundary.")
        start, end = float(boundary["start"]), float(boundary["end"])
        if end <= start:
            raise ValueError(f"Multi-source montage moment {moment.get('id')} has invalid source boundaries.")
        args.extend(["-i", str(source_path)])
        filters.append(
            f"[{index}:v]trim=start={start:.3f}:end={end:.3f},setpts=PTS-STARTPTS,"
            f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=30,format=yuv420p[v{index}]"
        )
        filters.append(
            f"[{index}:a]atrim=start={start:.3f}:end={end:.3f},asetpts=PTS-STARTPTS,"
            f"aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo[a{index}]"
        )
        labels.append(f"[v{index}][a{index}]")
    filters.append(f"{''.join(labels)}concat=n={len(selected_moments)}:v=1:a=1[outv][outa]")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    args.extend([
        "-filter_complex", ";".join(filters), "-map", "[outv]", "-map", "[outa]",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-c:a", "aac", "-b:a", "192k",
        str(output_path),
    ])
    run(args)


def _chunks(items: list[dict], size: int) -> list[list[dict]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def render_dialogue_wav(
    *,
    schedule: dict,
    duration: float,
    output_path: Path,
    sample_rate: int,
    channels: int,
    target_lufs: float,
    fade_duration: float,
    batch_size: int = 40,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    base = output_path.parent / "_silence_base.wav"
    run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"anullsrc=channel_layout={'stereo' if channels == 2 else 'mono'}:sample_rate={sample_rate}",
            "-t",
            f"{duration:.3f}",
            str(base),
        ]
    )
    current = base
    temp_files = [base]
    enabled_mappings = [mapping for mapping in schedule["mappings"] if mapping.get("enabled", True)]
    for batch_index, batch in enumerate(_chunks(enabled_mappings, batch_size), start=1):
        next_path = output_path.parent / f"_mix_batch_{batch_index:04d}.wav"
        args = ["ffmpeg", "-y", "-i", str(current)]
        filter_parts = []
        mix_inputs = ["[0:a]"]
        for item_index, mapping in enumerate(batch, start=1):
            args.extend(["-i", str(Path(mapping["clip_path"]))])
            delay_ms = int(round(mapping["destination_timestamp"] * 1000))
            trim_start = float(mapping.get("clip_trim_start", 0.0))
            trim_duration = mapping.get("clip_trim_duration")
            filters = []
            rendered_duration = None
            if trim_duration is not None:
                trim_duration_float = float(trim_duration)
                trim_end = trim_start + trim_duration_float
                filters.append(f"atrim=start={trim_start:.3f}:end={trim_end:.3f}")
                filters.append("asetpts=PTS-STARTPTS")
                rendered_duration = trim_duration_float
            stretch_factor = float(mapping["stretch_factor"])
            if abs(stretch_factor - 1.0) > 0.001:
                filters.append(f"atempo={1.0 / stretch_factor:.4f}")
                if rendered_duration is not None:
                    rendered_duration *= stretch_factor
            filters.append(f"loudnorm=I={target_lufs:.1f}:LRA=11:TP=-1.5")
            filters.extend(_mapping_audio_filters(mapping))
            clip_fade = _effective_fade_duration(
                float(mapping.get("fade_duration", fade_duration) or 0.0), rendered_duration,
            )
            if clip_fade > 0:
                filters.append(f"afade=t=in:st=0:d={clip_fade:.3f}")
                if rendered_duration is not None:
                    fade_start = max(0.0, rendered_duration - clip_fade)
                    filters.append(f"afade=t=out:st={fade_start:.3f}:d={clip_fade:.3f}")
            filters.append(f"adelay={delay_ms}:all=1")
            label = f"clip{item_index}"
            filter_parts.append(f"[{item_index}:a]{','.join(filters)}[{label}]")
            mix_inputs.append(f"[{label}]")
        filter_parts.append(
            f"{''.join(mix_inputs)}amix=inputs={len(mix_inputs)}:duration=first:dropout_transition=0:normalize=0,alimiter=limit=0.95[out]"
        )
        args.extend(
            [
                "-filter_complex",
                ";".join(filter_parts),
                "-map",
                "[out]",
                "-ar",
                str(sample_rate),
                "-ac",
                str(channels),
                str(next_path),
            ]
        )
        run(args)
        if current != base:
            temp_files.append(current)
        current = next_path
    if current != output_path:
        if output_path.exists():
            output_path.unlink()
        current.replace(output_path)
    for temp in temp_files:
        if temp.exists() and temp != output_path:
            temp.unlink()



def render_schedule_over_original_audio(
    *,
    original_media: Path,
    schedule: dict,
    duration: float,
    output_path: Path,
    sample_rate: int,
    channels: int,
    target_lufs: float,
    fade_duration: float,
    mute_regions: list[dict] | None = None,
    duck_db: float = -28.0,
    suppression_mode: str = "duck",
    suppression_fade_duration: float = 0.05,
    background_reconstruction: str = "none",
    mute_batch_size: int = 40,
    mix_batch_size: int = 40,
) -> None:
    if suppression_mode not in {"duck", "hard_mute"}:
        raise ValueError(f"Unknown source-bed suppression mode: {suppression_mode}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    base = output_path.parent / "_self_shuffle_original_base.wav"
    run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(original_media),
            "-vn",
            "-t",
            f"{duration:.3f}",
            "-ar",
            str(sample_rate),
            "-ac",
            str(channels),
            str(base),
        ]
    )
    current = base
    temp_files = [base]
    enabled_mappings = [mapping for mapping in schedule["mappings"] if mapping.get("enabled", True)]
    requested_mute_regions = mute_regions if mute_regions is not None else enabled_mappings
    regions_to_mute = _audio_active_duck_regions(requested_mute_regions)
    suppression_report = {
        "strategy": "hard_carrier_speech_suppression_v1" if suppression_mode == "hard_mute" else "clip_activity_exact_v1",
        "requested_region_count": len(requested_mute_regions),
        "rendered_region_count": len(regions_to_mute),
        "duck_db": None if suppression_mode == "hard_mute" else round(float(duck_db), 3),
        "suppression_mode": suppression_mode,
        "suppression_floor": "DIGITAL_SILENCE" if suppression_mode == "hard_mute" else f"{float(duck_db):.1f}dB",
        "edge_fade_seconds": round(float(suppression_fade_duration), 3) if suppression_mode == "hard_mute" else 0.0,
        "residual_speech_test": "NOT_ACOUSTICALLY_MEASURED",
    }
    schedule["audio_ducking"] = suppression_report
    if suppression_mode == "hard_mute":
        schedule["audio_suppression"] = dict(suppression_report)

    for batch_index, batch in enumerate(_chunks(regions_to_mute, mute_batch_size), start=1):
        next_path = output_path.parent / f"_self_shuffle_mute_batch_{batch_index:04d}.wav"
        filters = []
        for region in batch:
            start, end = _mute_region_bounds(region)
            if end <= start:
                continue
            if suppression_mode == "hard_mute":
                filters.append(_hard_suppression_filter(
                    start=start,
                    end=end,
                    total_duration=duration,
                    fade_duration=suppression_fade_duration,
                ))
            else:
                filters.append(f"volume=enable='between(t,{start:.3f},{end:.3f})':volume={duck_db:.1f}dB")
        if not filters:
            continue
        run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(current),
                "-af",
                ",".join(filters),
                "-ar",
                str(sample_rate),
                "-ac",
                str(channels),
                str(next_path),
            ]
        )
        if current != base:
            temp_files.append(current)
        current = next_path

    if suppression_mode == "hard_mute" and background_reconstruction == "neighboring_non_speech_with_adaptive_crossfades":
        reconstruction_plan = _neighboring_ambience_plan(
            target_regions=regions_to_mute,
            protected_speech_regions=(
                list(schedule.get("destination_speech_regions", []))
                + list(schedule.get("residue_correction_regions", []))
            ) or regions_to_mute,
            duration=duration,
        )
        if reconstruction_plan:
            next_path = output_path.parent / "_self_shuffle_ambience_reconstruction.wav"
            _render_ambience_reconstruction(
                current=current,
                source=base,
                plan=reconstruction_plan,
                output_path=next_path,
                sample_rate=sample_rate,
                channels=channels,
                fade_duration=max(0.02, suppression_fade_duration),
            )
            if current != base:
                temp_files.append(current)
            current = next_path
        reconstruction_report = {
            "strategy": background_reconstruction,
            "requested_region_count": len(regions_to_mute),
            "reconstructed_region_count": len(reconstruction_plan),
            "silence_fallback_region_count": len(regions_to_mute) - len(reconstruction_plan),
            "sources": reconstruction_plan,
            "silence_fallback_targets": _unreconstructed_targets(regions_to_mute, reconstruction_plan),
            "residual_speech_test": "NOT_ACOUSTICALLY_MEASURED",
        }
    else:
        reconstruction_report = {
            "strategy": "none",
            "requested_region_count": len(regions_to_mute),
            "reconstructed_region_count": 0,
            "silence_fallback_region_count": len(regions_to_mute) if suppression_mode == "hard_mute" else 0,
            "sources": [],
            "silence_fallback_targets": [
                {"start": round(_mute_region_bounds(row)[0], 3), "end": round(_mute_region_bounds(row)[1], 3)}
                for row in regions_to_mute
            ],
            "residual_speech_test": "NOT_ACOUSTICALLY_MEASURED",
        }
    schedule["background_reconstruction_report"] = reconstruction_report
    schedule.setdefault("performance_summary", {})["voice_residue"] = "NOT_ACOUSTICALLY_MEASURED"
    schedule["performance_summary"]["ambience_reconstructed_regions"] = reconstruction_report["reconstructed_region_count"]
    schedule["performance_summary"]["ambience_silence_fallback_regions"] = reconstruction_report["silence_fallback_region_count"]
    schedule["performance_summary"]["suppression_contract"] = (
        "ORIGINAL_BED_ZEROED_IN_DECLARED_SPEECH_REGIONS"
        if suppression_mode == "hard_mute"
        else "DESTINATION_BED_DUCKED"
    )

    for batch_index, batch in enumerate(_chunks(enabled_mappings, mix_batch_size), start=1):
        next_path = output_path.parent / f"_self_shuffle_mix_batch_{batch_index:04d}.wav"
        args = ["ffmpeg", "-y", "-i", str(current)]
        filter_parts = []
        mix_inputs = ["[0:a]"]
        for item_index, mapping in enumerate(batch, start=1):
            args.extend(["-i", str(Path(mapping["clip_path"]))])
            delay_ms = int(round(mapping["destination_timestamp"] * 1000))
            trim_start = float(mapping.get("clip_trim_start", 0.0))
            trim_duration = mapping.get("clip_trim_duration")
            filters = []
            rendered_duration = None
            if trim_duration is not None:
                trim_duration_float = float(trim_duration)
                trim_end = trim_start + trim_duration_float
                filters.append(f"atrim=start={trim_start:.3f}:end={trim_end:.3f}")
                filters.append("asetpts=PTS-STARTPTS")
                rendered_duration = trim_duration_float
            stretch_factor = float(mapping["stretch_factor"])
            if abs(stretch_factor - 1.0) > 0.001:
                filters.append(f"atempo={1.0 / stretch_factor:.4f}")
                if rendered_duration is not None:
                    rendered_duration *= stretch_factor
            filters.append(f"loudnorm=I={target_lufs:.1f}:LRA=11:TP=-1.5")
            filters.extend(_mapping_audio_filters(mapping))
            clip_fade = _effective_fade_duration(
                float(mapping.get("fade_duration", fade_duration) or 0.0), rendered_duration,
            )
            if clip_fade > 0:
                filters.append(f"afade=t=in:st=0:d={clip_fade:.3f}")
                if rendered_duration is not None:
                    fade_start = max(0.0, rendered_duration - clip_fade)
                    filters.append(f"afade=t=out:st={fade_start:.3f}:d={clip_fade:.3f}")
            filters.append(f"adelay={delay_ms}:all=1")
            label = f"clip{item_index}"
            filter_parts.append(f"[{item_index}:a]{','.join(filters)}[{label}]")
            mix_inputs.append(f"[{label}]")
        filter_parts.append(
            f"{''.join(mix_inputs)}amix=inputs={len(mix_inputs)}:duration=first:dropout_transition=0:normalize=0,alimiter=limit=0.95[out]"
        )
        args.extend(
            [
                "-filter_complex",
                ";".join(filter_parts),
                "-map",
                "[out]",
                "-ar",
                str(sample_rate),
                "-ac",
                str(channels),
                str(next_path),
            ]
        )
        run(args)
        if current != base:
            temp_files.append(current)
        current = next_path

    if current != output_path:
        if output_path.exists():
            output_path.unlink()
        current.replace(output_path)
    for temp in temp_files:
        if temp.exists() and temp != output_path:
            temp.unlink()



def _mute_region_bounds(region: dict) -> tuple[float, float]:
    start = float(region.get("start", region.get("destination_timestamp", 0.0)) or 0.0)
    if region.get("clip_path") or region.get("clip_trim_duration") is not None:
        stretch = max(0.001, float(region.get("stretch_factor", 1.0) or 1.0))
        duration = region.get("clip_trim_duration")
        if duration is None:
            duration = float(region.get("planned_render_duration", region.get("duration", 0.0)) or 0.0) / stretch
        duration = float(duration or 0.0) * stretch
    else:
        duration = region.get("duration", region.get("planned_render_duration", 0.0))
    end = start + float(duration or 0.0)
    return start, end


def _audio_active_duck_regions(regions: list[dict]) -> list[dict[str, float]]:
    """Duck only while replacement audio is literally active, never through its padding."""
    expanded: list[dict[str, float]] = []
    for region in regions:
        clip_path = Path(str(region.get("clip_path", "")))
        if not region.get("clip_path") or not clip_path.exists():
            start, end = _mute_region_bounds(region)
            if end > start:
                expanded.append({"start": start, "duration": end - start})
            continue
        trim_start = max(0.0, float(region.get("clip_trim_start", 0.0) or 0.0))
        stretch = max(0.001, float(region.get("stretch_factor", 1.0) or 1.0))
        trim_duration = region.get("clip_trim_duration")
        if trim_duration is None:
            planned = float(region.get("planned_render_duration", region.get("duration", 0.0)) or 0.0)
            trim_duration = planned / stretch
        trim_end = trim_start + max(0.0, float(trim_duration or 0.0))
        try:
            measured = analyze_wav_intervals(clip_path, [{"id": "clip", "start": trim_start, "end": trim_end}])
        except (EOFError, OSError, ValueError, wave.Error):
            measured = []
        if not measured:
            start, end = _mute_region_bounds(region)
            if end > start:
                expanded.append({"start": start, "duration": end - start})
            continue
        destination_start = float(region.get("destination_timestamp", region.get("start", 0.0)) or 0.0)
        for active in measured[0].get("active_intervals", []):
            relative_start = max(0.0, float(active["start"]) - trim_start) * stretch
            relative_end = max(relative_start, float(active["end"]) - trim_start) * stretch
            if relative_end > relative_start:
                expanded.append({"start": destination_start + relative_start, "duration": relative_end - relative_start})
    return _merge_duck_regions(expanded)


def _merge_duck_regions(regions: list[dict[str, float]], *, gap: float = 0.05) -> list[dict[str, float]]:
    merged: list[dict[str, float]] = []
    for region in sorted(regions, key=lambda row: float(row["start"])):
        start = float(region["start"])
        end = start + float(region["duration"])
        if merged:
            prior_end = merged[-1]["start"] + merged[-1]["duration"]
            if start <= prior_end + gap:
                merged[-1]["duration"] = round(max(prior_end, end) - merged[-1]["start"], 3)
                continue
        merged.append({"start": round(start, 3), "duration": round(end - start, 3)})
    return merged


def _hard_suppression_filter(*, start: float, end: float, total_duration: float, fade_duration: float) -> str:
    """Return a click-resistant envelope that reaches literal silence for the requested region."""
    fade = max(0.0, float(fade_duration))
    fade_out_start = max(0.0, start - fade)
    fade_in_end = min(float(total_duration), end + fade)
    expression = "1"
    if fade_in_end > end:
        expression = f"if(lt(t,{fade_in_end:.3f}),(t-{end:.3f})/{fade_in_end - end:.3f},1)"
    expression = f"if(lte(t,{end:.3f}),0,{expression})"
    if start > fade_out_start:
        expression = f"if(lt(t,{start:.3f}),({start:.3f}-t)/{start - fade_out_start:.3f},{expression})"
    expression = f"if(lt(t,{fade_out_start:.3f}),1,{expression})"
    return f"volume='{expression}':eval=frame"


def _neighboring_ambience_plan(
    *,
    target_regions: list[dict[str, float]],
    protected_speech_regions: list[dict],
    duration: float,
    guard: float = 0.08,
    minimum_source_duration: float = 0.12,
    minimum_target_coverage: float = 0.6,
) -> list[dict[str, Any]]:
    """Score deterministic source beds from the complement of all detected speech."""
    blocked = []
    for region in protected_speech_regions:
        start, end = _mute_region_bounds(region)
        if end > start:
            blocked.append({
                "start": max(0.0, start - guard),
                "duration": min(float(duration), end + guard) - max(0.0, start - guard),
            })
    blocked = _merge_duck_regions(blocked, gap=guard)
    gaps: list[tuple[float, float]] = []
    cursor = 0.0
    for region in blocked:
        start, end = _mute_region_bounds(region)
        if start - cursor >= minimum_source_duration:
            gaps.append((cursor, start))
        cursor = max(cursor, end)
    if float(duration) - cursor >= minimum_source_duration:
        gaps.append((cursor, float(duration)))

    plan: list[dict[str, Any]] = []
    gap_use_counts: dict[tuple[float, float], int] = {}
    for target in target_regions:
        target_start, target_end = _mute_region_bounds(target)
        target_duration = target_end - target_start
        if target_duration <= 0.0 or not gaps:
            continue
        center = (target_start + target_end) / 2.0
        candidates = []
        for gap_start, gap_end in gaps:
            available = gap_end - gap_start
            reuse_count = gap_use_counts.get((gap_start, gap_end), 0)
            if reuse_count > 0:
                continue
            coverage = min(1.0, available / max(target_duration, minimum_source_duration))
            if coverage < max(0.0, min(1.0, minimum_target_coverage)):
                continue
            distance = 0.0 if gap_start <= center <= gap_end else min(abs(center - gap_start), abs(center - gap_end))
            proximity = 1.0 / (1.0 + distance)
            duration_fit = coverage
            preceding = gap_end <= target_start
            side_preference = 1.0 if preceding else 0.8
            score = proximity * 0.55 + duration_fit * 0.35 + side_preference * 0.10
            candidates.append({
                "gap_start": gap_start,
                "gap_end": gap_end,
                "score": score,
                "components": {
                    "proximity": round(proximity, 4),
                    "duration_fit": round(duration_fit, 4),
                    "side_preference": round(side_preference, 4),
                    "reuse_penalty": 0.0,
                },
            })
        if not candidates:
            continue
        selected = max(candidates, key=lambda row: (row["score"], -row["gap_start"]))
        source_start, source_end = float(selected["gap_start"]), float(selected["gap_end"])
        gap_use_counts[(source_start, source_end)] = gap_use_counts.get((source_start, source_end), 0) + 1
        available = source_end - source_start
        used = min(available, max(minimum_source_duration, target_duration))
        if source_end <= target_start:
            source_start = source_end - used
        elif source_start >= target_end:
            source_end = source_start + used
        else:
            source_start = max(source_start, min(center - used / 2.0, source_end - used))
            source_end = source_start + used
        plan.append({
            "target_start": round(target_start, 3),
            "target_end": round(target_end, 3),
            "target_duration": round(target_duration, 3),
            "source_start": round(source_start, 3),
            "source_end": round(source_end, 3),
            "source_duration": round(source_end - source_start, 3),
            "source_kind": "detected_non_speech_neighbor",
            "selection_score": round(float(selected["score"]), 4),
            "score_components": selected["components"],
            "candidate_count": len(candidates),
            "render_duration": round(source_end - source_start, 3),
            "unfilled_duration": round(max(0.0, target_duration - (source_end - source_start)), 3),
            "coverage_ratio": round(min(1.0, (source_end - source_start) / target_duration), 4),
            "loop_required": False,
        })
    return plan


def _unreconstructed_targets(
    targets: list[dict[str, float]],
    plan: list[dict[str, Any]],
) -> list[dict[str, float]]:
    reconstructed = {
        (round(float(row["target_start"]), 3), round(float(row["target_end"]), 3))
        for row in plan
    }
    missing = []
    for target in targets:
        start, end = _mute_region_bounds(target)
        if (round(start, 3), round(end, 3)) not in reconstructed:
            missing.append({"start": round(start, 3), "end": round(end, 3)})
    return missing


def _render_ambience_reconstruction(
    *,
    current: Path,
    source: Path,
    plan: list[dict[str, float | str]],
    output_path: Path,
    sample_rate: int,
    channels: int,
    fade_duration: float,
) -> None:
    args = ["ffmpeg", "-y", "-i", str(current)]
    for _ in plan:
        args.extend(["-i", str(source)])
    filters: list[str] = []
    mix_inputs = ["[0:a]"]
    for index, item in enumerate(plan, start=1):
        source_start = float(item["source_start"])
        source_end = float(item["source_end"])
        source_duration = max(0.001, source_end - source_start)
        target_duration = float(item["target_duration"])
        render_duration = min(target_duration, float(item.get("render_duration", target_duration)))
        target_start = float(item["target_start"])
        chain = [
            f"atrim=start={source_start:.3f}:end={source_end:.3f}",
            "asetpts=PTS-STARTPTS",
        ]
        chain.append(f"atrim=duration={render_duration:.3f}")
        edge = min(float(fade_duration), render_duration / 3.0)
        if edge > 0.0:
            chain.extend([
                f"afade=t=in:st=0:d={edge:.3f}",
                f"afade=t=out:st={max(0.0, render_duration - edge):.3f}:d={edge:.3f}",
            ])
        chain.append(f"adelay={int(round(target_start * 1000))}:all=1")
        label = f"ambience{index}"
        filters.append(f"[{index}:a]{','.join(chain)}[{label}]")
        mix_inputs.append(f"[{label}]")
    filters.append(
        f"{''.join(mix_inputs)}amix=inputs={len(mix_inputs)}:duration=first:dropout_transition=0:normalize=0,alimiter=limit=0.95[out]"
    )
    args.extend([
        "-filter_complex", ";".join(filters), "-map", "[out]",
        "-ar", str(sample_rate), "-ac", str(channels), str(output_path),
    ])
    run(args)


def _effective_fade_duration(configured: float, rendered_duration: float | None) -> float:
    if configured <= 0:
        return 0.0
    if rendered_duration is None:
        return configured
    return max(0.0, min(configured, rendered_duration / 3.0))


def _mapping_audio_filters(mapping: dict) -> list[str]:
    """Translate contract-visible mapping controls into bounded FFmpeg filters."""
    filters: list[str] = []
    highpass = mapping.get("highpass_hz")
    lowpass = mapping.get("lowpass_hz")
    gain = mapping.get("gain_db")
    if highpass is not None:
        filters.append(f"highpass=f={max(20.0, min(20000.0, float(highpass))):.1f}")
    if lowpass is not None:
        filters.append(f"lowpass=f={max(20.0, min(20000.0, float(lowpass))):.1f}")
    if gain is not None:
        filters.append(f"volume={max(-60.0, min(24.0, float(gain))):.2f}dB")
    return filters


def scheduled_audio_duration(schedule: dict, destination_duration: float) -> float:
    enabled = [mapping for mapping in schedule.get("mappings", []) if mapping.get("enabled", True)]
    if not enabled:
        return max(0.001, float(destination_duration))
    end_time = max(
        float(mapping.get("destination_timestamp", 0.0)) + float(mapping.get("planned_render_duration", mapping.get("clip_trim_duration", 0.0)))
        for mapping in enabled
    )
    return round(max(0.001, min(float(destination_duration), end_time)), 3)


def preview_bounds(mappings: list[dict], destination_duration: float, padding: float = 1.0) -> tuple[float, float]:
    if not mappings:
        raise ValueError("Preview needs at least one mapping.")
    start = min(float(mapping.get("destination_timestamp", 0.0)) for mapping in mappings)
    end = max(
        float(mapping.get("destination_timestamp", 0.0))
        + float(mapping.get("planned_render_duration", mapping.get("clip_trim_duration", 0.0)))
        for mapping in mappings
    )
    start = max(0.0, start - padding)
    end = min(float(destination_duration), end + padding)
    if end <= start:
        end = min(float(destination_duration), start + 0.5)
    return round(start, 3), round(max(end, start + 0.001), 3)


def build_preview_schedule(schedule: dict, mappings: list[dict], start_time: float) -> dict:
    preview_mappings = []
    for mapping in mappings:
        item = dict(mapping)
        item["enabled"] = True
        _rebase_destination_times(item, start_time)
        preview_mappings.append(item)
    preview = dict(schedule)
    preview["mappings"] = preview_mappings
    return preview


def _rebase_destination_times(mapping: dict, start_time: float) -> None:
    for field in (
        "destination_timestamp",
        "alignment_slot_start",
        "alignment_slot_end",
        "shot_start",
        "shot_end",
    ):
        if mapping.get(field) is None:
            continue
        mapping[field] = round(max(0.0, float(mapping.get(field, 0.0) or 0.0) - start_time), 3)


def extract_video_segment(
    *,
    input_video: Path,
    output_path: Path,
    start_time: float,
    duration: float,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            "ffmpeg",
            "-y",
            "-ss",
            f"{start_time:.3f}",
            "-t",
            f"{duration:.3f}",
            "-i",
            str(input_video),
            "-c",
            "copy",
            str(output_path),
        ]
    )


def concat_media_files(*, inputs: list[Path], output_path: Path, reencode: bool = False) -> None:
    if not inputs:
        raise ValueError("At least one input is required for media concat.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    list_path = output_path.parent / f"_{output_path.stem}_concat.txt"
    list_path.write_text("".join(f"file '{_ffconcat_path(path)}'\n" for path in inputs), encoding="utf-8")
    args = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(list_path)]
    if reencode:
        args.extend(["-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-c:a", "aac"])
    else:
        args.extend(["-c", "copy"])
    args.append(str(output_path))
    run(args)


def concat_wav_files(*, inputs: list[Path], output_path: Path, sample_rate: int, channels: int) -> None:
    if not inputs:
        raise ValueError("At least one input is required for WAV concat.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    list_path = output_path.parent / f"_{output_path.stem}_concat.txt"
    list_path.write_text("".join(f"file '{_ffconcat_path(path)}'\n" for path in inputs), encoding="utf-8")
    run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_path),
            "-ar",
            str(sample_rate),
            "-ac",
            str(channels),
            str(output_path),
        ]
    )


def _ffconcat_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/").replace("'", "'\\''")
def mux_video_segment(
    *,
    destination_video: Path,
    dialogue_wav: Path,
    output_path: Path,
    start_time: float,
    duration: float,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            "ffmpeg",
            "-y",
            "-ss",
            f"{start_time:.3f}",
            "-t",
            f"{duration:.3f}",
            "-i",
            str(destination_video),
            "-i",
            str(dialogue_wav),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            str(output_path),
        ]
    )

def mux_video(
    *,
    destination_video: Path,
    dialogue_wav: Path,
    output_path: Path,
    duration: float | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    args = [
        "ffmpeg",
        "-y",
        "-i",
        str(destination_video),
        "-i",
        str(dialogue_wav),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
    ]
    if duration is not None:
        if duration <= 0:
            raise ValueError("Mux duration must be positive.")
        # Stream-copy muxing can otherwise retain video packets through the next
        # GOP even when -shortest is present. The planner's audio-safe extent is
        # authoritative, so also impose it as an explicit output duration.
        args.extend(["-t", f"{duration:.3f}"])
    args.extend(["-shortest", str(output_path)])
    run(args)

