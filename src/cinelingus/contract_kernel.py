from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from .filter_lab.models import FilterDefinition
from .tools import ffprobe_json
from .util import read_json, stable_hash, utc_now, write_json
from .validation import _validate_object


CONTRACT_SCHEMA_VERSION = "1.0"
CONTRACT_KERNEL_VERSION = "contract_kernel_v1"
COMPLETE_MEDIA_SCOPE = "complete_media_files"
FULL_TIMELINE_POLICY = "FULL_SOURCE_TIMELINE_LIMITED_BY_REQUIRED_AUDIO"


@dataclass(frozen=True)
class StreamDescriptor:
    index: int
    kind: str
    codec: str | None
    duration: float | None
    start_time: float
    time_base: str | None = None
    frame_rate: float | None = None
    sample_rate: int | None = None
    channels: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MediaDescriptor:
    path: Path
    media_hash: str
    format_duration: float | None
    streams: tuple[StreamDescriptor, ...]
    probe_version: str = "ffprobe_stream_facts_v1"

    @property
    def video_streams(self) -> tuple[StreamDescriptor, ...]:
        return tuple(stream for stream in self.streams if stream.kind == "video")

    @property
    def audio_streams(self) -> tuple[StreamDescriptor, ...]:
        return tuple(stream for stream in self.streams if stream.kind == "audio")

    @property
    def primary_video_duration(self) -> float:
        if not self.video_streams:
            raise ValueError(f"Media has no video stream: {self.path}")
        duration = self.video_streams[0].duration or self.format_duration
        if duration is None or duration <= 0:
            raise ValueError(f"Media has no positive video duration: {self.path}")
        return duration

    @property
    def primary_audio_duration(self) -> float:
        if not self.audio_streams:
            raise ValueError(f"Media has no audio stream: {self.path}")
        duration = self.audio_streams[0].duration or self.format_duration
        if duration is None or duration <= 0:
            raise ValueError(f"Media has no positive audio duration: {self.path}")
        return duration

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "media_hash": self.media_hash,
            "format_duration": self.format_duration,
            "probe_version": self.probe_version,
            "streams": [stream.to_dict() for stream in self.streams],
        }

    @classmethod
    def from_probe(cls, *, path: Path, media_hash: str, probe: dict[str, Any]) -> "MediaDescriptor":
        streams = tuple(_stream_descriptor(index, stream) for index, stream in enumerate(probe.get("streams") or []))
        return cls(
            path=Path(path).expanduser().resolve(),
            media_hash=str(media_hash),
            format_duration=_positive_float((probe.get("format") or {}).get("duration")),
            streams=streams,
        )

    @classmethod
    def from_movie_artifact(cls, movie: dict[str, Any]) -> "MediaDescriptor":
        return cls.from_probe(
            path=Path(movie["path"]),
            media_hash=str(movie["media_hash"]),
            probe={"format": {"duration": movie.get("duration")}, "streams": movie.get("streams", [])},
        )


@dataclass(frozen=True)
class ResolvedExtent:
    start: float
    duration: float
    anchor_video_duration: float
    required_audio_durations: tuple[float, ...]
    authority: str = "shortest_required_stream"
    policy: str = FULL_TIMELINE_POLICY

    @property
    def curtailed(self) -> bool:
        return self.duration + 0.001 < self.anchor_video_duration

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": self.start,
            "output_duration": self.duration,
            "anchor_video_duration": self.anchor_video_duration,
            "required_audio_durations": list(self.required_audio_durations),
            "authority": self.authority,
            "policy": self.policy,
            "anchor_curtailed": self.curtailed,
        }


class OutputExtentResolver:
    """The sole authority for translating stream facts into an output boundary."""

    def resolve(self, media: Iterable[MediaDescriptor]) -> ResolvedExtent:
        rows = tuple(media)
        if not rows:
            raise ValueError("A run contract requires at least one media input.")
        anchor_duration = rows[0].primary_video_duration
        audio_rows = rows[1:] or rows[:1]
        audio_durations = tuple(row.primary_audio_duration for row in audio_rows)
        duration = round(min((anchor_duration, *audio_durations)), 3)
        if duration <= 0:
            raise ValueError("The canonical output extent is not positive.")
        return ResolvedExtent(
            start=0.0,
            duration=duration,
            anchor_video_duration=round(anchor_duration, 3),
            required_audio_durations=tuple(round(value, 3) for value in audio_durations),
        )


@dataclass(frozen=True)
class RunContract:
    contract_id: str
    creation_timestamp: str
    filter_id: str
    filter_version: str
    media: tuple[MediaDescriptor, ...]
    timeline: ResolvedExtent
    audio_policy: dict[str, Any]
    repetition_policy: dict[str, Any]
    analysis_requirements: dict[str, Any]
    filter_invariants: tuple[str, ...]
    acceptance: dict[str, Any]
    schema_version: str = CONTRACT_SCHEMA_VERSION
    kernel_version: str = CONTRACT_KERNEL_VERSION
    input_scope: str = COMPLETE_MEDIA_SCOPE

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kernel_version": self.kernel_version,
            "contract_id": self.contract_id,
            "creation_timestamp": self.creation_timestamp,
            "filter_id": self.filter_id,
            "filter_version": self.filter_version,
            "input_scope": self.input_scope,
            "media": [row.to_dict() for row in self.media],
            "timeline": self.timeline.to_dict(),
            "audio": dict(self.audio_policy),
            "repetition": dict(self.repetition_policy),
            "analysis": dict(self.analysis_requirements),
            "filter": {"invariants": list(self.filter_invariants)},
            "acceptance": dict(self.acceptance),
        }


def compile_run_contract(
    *,
    definition: FilterDefinition,
    media: Iterable[MediaDescriptor],
    resolver: OutputExtentResolver | None = None,
) -> RunContract:
    rows = tuple(media)
    definition.validate_film_count(len(rows))
    extent = (resolver or OutputExtentResolver()).resolve(rows)
    repetition_allowed = definition.id == "translation.echo"
    identity_requirement = "required" if definition.requires_speaker_identity else "optional"
    payload = {
        "kernel_version": CONTRACT_KERNEL_VERSION,
        "filter_id": definition.id,
        "filter_version": definition.version,
        "media_hashes": [row.media_hash for row in rows],
        "timeline": extent.to_dict(),
        "audio_policy": "continuous_source_bed_with_authored_substitutions",
        "repetition_policy": "filter_authorized" if repetition_allowed else "forbidden",
        "analysis_identity": identity_requirement,
    }
    return RunContract(
        contract_id=f"run_contract_{stable_hash(payload)[:24]}",
        creation_timestamp=utc_now(),
        filter_id=definition.id,
        filter_version=definition.version,
        media=rows,
        timeline=extent,
        audio_policy={
            "policy": "continuous_source_bed_with_authored_substitutions",
            "final_stream_count": 1,
            "original_anchor_stream_retained": False,
        },
        repetition_policy={
            "policy": "filter_authorized" if repetition_allowed else "forbidden",
            "authorization_basis": "FILTER_CONTRACT:translation.echo" if repetition_allowed else None,
            "exhaustion_behavior": "leave_remaining_windows_unmodified",
        },
        analysis_requirements={
            "transcription": "required",
            "speaker_identity": identity_requirement,
            "accepted_speaker_quality": ["direct"] if definition.requires_speaker_identity else ["direct", "inferred", "weak"],
        },
        filter_invariants=tuple(definition.quality_requirements),
        acceptance={
            "duration_tolerance_seconds": 0.05,
            "video_packet_tolerance_seconds": 0.25,
            "provenance_required": True,
            "audio_activity_required": True,
            "required_video_stream_count": 1,
            "required_audio_stream_count": 1,
        },
    )


def probe_media_descriptors(
    paths: Iterable[Path],
    *,
    media_hashes: Iterable[str] | None = None,
) -> tuple[MediaDescriptor, ...]:
    path_rows = tuple(Path(path) for path in paths)
    hashes = tuple(media_hashes or (stable_hash(str(path.resolve())) for path in path_rows))
    if len(hashes) != len(path_rows):
        raise ValueError("Media hashes and paths must have matching lengths.")
    return tuple(
        MediaDescriptor.from_probe(path=path, media_hash=media_hash, probe=ffprobe_json(path))
        for path, media_hash in zip(path_rows, hashes)
    )


def write_run_contract(contract: RunContract, output_path: Path, schemas_dir: Path) -> Path:
    data = contract.to_dict()
    _validate_object(data, read_json(schemas_dir / "run_contract.schema.json"), str(output_path))
    write_json(output_path, data)
    return output_path


def _stream_descriptor(index: int, stream: dict[str, Any]) -> StreamDescriptor:
    return StreamDescriptor(
        index=int(stream.get("index", index)),
        kind=str(stream.get("codec_type") or "unknown"),
        codec=str(stream["codec_name"]) if stream.get("codec_name") else None,
        duration=_positive_float(stream.get("duration")),
        start_time=_positive_or_zero(stream.get("start_time")),
        time_base=str(stream["time_base"]) if stream.get("time_base") else None,
        frame_rate=_rate(stream.get("avg_frame_rate")),
        sample_rate=int(stream["sample_rate"]) if stream.get("sample_rate") else None,
        channels=int(stream["channels"]) if stream.get("channels") else None,
    )


def _positive_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _positive_or_zero(value: Any) -> float:
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return 0.0


def _rate(value: Any) -> float | None:
    if not value or value == "0/0":
        return None
    if isinstance(value, (int, float)):
        return float(value)
    numerator, separator, denominator = str(value).partition("/")
    try:
        return float(numerator) / float(denominator) if separator else float(numerator)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
