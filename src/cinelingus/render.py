from __future__ import annotations

from pathlib import Path
import wave

from .audio_provenance import analyze_wav_intervals
from .tools import run


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
            clip_fade = _effective_fade_duration(fade_duration, rendered_duration)
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
    mute_batch_size: int = 40,
    mix_batch_size: int = 40,
) -> None:
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
    schedule["audio_ducking"] = {
        "strategy": "clip_activity_exact_v1",
        "requested_region_count": len(requested_mute_regions),
        "rendered_region_count": len(regions_to_mute),
        "duck_db": round(float(duck_db), 3),
    }

    for batch_index, batch in enumerate(_chunks(regions_to_mute, mute_batch_size), start=1):
        next_path = output_path.parent / f"_self_shuffle_mute_batch_{batch_index:04d}.wav"
        filters = []
        for region in batch:
            start, end = _mute_region_bounds(region)
            if end <= start:
                continue
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
            clip_fade = _effective_fade_duration(fade_duration, rendered_duration)
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

