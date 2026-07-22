from __future__ import annotations

import contextlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import queue
import threading
import time
import traceback
from typing import Any
from collections.abc import Mapping

from PySide6.QtCore import QObject, QTimer, Signal

from .cinematic_filters import FILTER_DISPLAY_NAMES
from .config import AppConfig, load_config
from .filter_lab.public_catalog import OperatingMode, PublicApparatusEntry, default_public_apparatus_catalog
from .filter_lab.recipe import FilterRecipe, load_recipe, save_recipe
from .filter_lab.registry import default_filter_registry
from .pipeline import Pipeline
from .operator_language import stage_key_for_diagnostic
from .reliable_inputs import default_input_directory, preflight_media_inputs
from .run_guard import exclusive_output_run, verify_filter_execution


QUALITY_LABELS = ("Glimpse", "Study", "Divination")
QUALITY_MODES = {"Glimpse": "fast_preview", "Study": "balanced", "Divination": "quality"}
MATCHING_LABELS = tuple(FILTER_DISPLAY_NAMES.values())
MATCHING_IDS = {label: key for key, label in FILTER_DISPLAY_NAMES.items()}
REALITY_LABELS = {"One Film": OperatingMode.SOLITARY, "Several Films": OperatingMode.MULTIWORLD}
STAGE_SEQUENCE = (
    ("inspect", "CATALOG"),
    ("source_dialogue", "VOICE"),
    ("destination_speech", "IDENTITY"),
    ("performances", "PERFORMANCE"),
    ("schedule", "ASSEMBLY"),
    ("render_audio", "RENDER"),
    ("finalize", "REVIEW"),
)
STAGE_PROGRESS_FLOORS = {
    "inspect": 6.0,
    "source_dialogue": 16.0,
    "destination_speech": 30.0,
    "performances": 46.0,
    "schedule": 60.0,
    "render_audio": 74.0,
    "finalize": 94.0,
}
PIPELINE_STAGE_PRESENTATIONS = {
    "multiworld:inspect_films": ("inspect", "CATALOGUING SELECTED MATERIAL"),
    "multiworld:create_shared_timeline": ("schedule", "ALIGNING FILM TIMELINES"),
    "multiworld:construct_world_model": ("performances", "CONSTRUCTING THE SHARED WORLD"),
    "multiworld:apply_cinematic_law": ("schedule", "APPLYING THE CINEMATIC LAW"),
    "multiworld:generate_replacement_decisions": ("schedule", "ARRANGING REPLACEMENT DECISIONS"),
    "multiworld:review": ("schedule", "REVIEWING THE ARRANGEMENT"),
    "multiworld:render": ("finalize", "EXAMINING THE FINISHED ARTIFACT"),
    "runtime:render_audio": ("render_audio", "RECONSTRUCTING THE SOUNDTRACK"),
    "runtime:render_video": ("render_audio", "ASSEMBLING PICTURE AND SOUND"),
    "runtime:finalize": ("finalize", "EXAMINING THE FINISHED ARTIFACT"),
}


def _quality_summary_from_problem_report(problem_report: Any) -> dict[str, Any]:
    """Normalize current and legacy report shapes without crashing the UI."""
    if not isinstance(problem_report, Mapping):
        return {}
    try:
        problem_count = int(problem_report.get("problem_count") or 0)
    except (TypeError, ValueError):
        problem_count = 0
    result: dict[str, Any] = {"problem_count": problem_count}
    summary = problem_report.get("summary")
    if isinstance(summary, Mapping):
        result.update(dict(summary))
    return result


def format_clock_duration(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:d}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"


def stage_sequence_key(stage_key: str | None) -> str | None:
    aliases = {"clips": "source_dialogue", "render_video": "render_audio"}
    normalized = aliases.get(str(stage_key or ""), str(stage_key or ""))
    return normalized if any(key == normalized for key, _label in STAGE_SEQUENCE) else None


@dataclass
class EngineState:
    reality: str = "Several Films"
    discipline: str = "Alchemical Engine"
    apparatus: str = "Transposition"
    films: list[Path | None] = field(default_factory=list)
    output_dir: Path = Path("output")
    quality: str = "Study"
    matching: str = "Balanced"
    parameters: dict[str, Any] = field(default_factory=dict)
    running: bool = False
    completed: bool = False
    cancelling: bool = False
    overall_progress: float = 0.0
    stage_progress: float = 0.0
    active_stage_index: int = -1
    operation: str = "AWAITING MATERIAL"
    machine_state: str = "DORMANT"
    elapsed: str = "00:00"
    remaining: str = "CALCULATING..."
    completion_time: str = "CALCULATING..."
    summary: str = "NO RUN YET"
    last_output: Path | None = None
    journal: list[str] = field(default_factory=list)
    performance_summary: dict[str, Any] = field(default_factory=dict)


class QueueWriter:
    def __init__(self, output: "queue.Queue[tuple[str, Any]]") -> None:
        self.output = output

    def write(self, text: str) -> int:
        if text:
            self.output.put(("log", text))
        return len(text)

    def flush(self) -> None:
        return None


class QtEngineController(QObject):
    """Toolkit-independent production run state exposed through Qt signals."""

    changed = Signal()
    error = Signal(str, str)
    notice = Signal(str)
    run_finished = Signal(str)

    def __init__(self, root: Path | None = None, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.root = (root or Path.cwd()).resolve()
        self.base_config = load_config(self.root)
        self.registry = default_filter_registry()
        self.catalog = default_public_apparatus_catalog()
        self.state = EngineState(output_dir=self.base_config.output_dir)
        self._events: "queue.Queue[tuple[str, Any]]" = queue.Queue()
        self._worker: threading.Thread | None = None
        self._cancel_requested = False
        self._run_started_at: float | None = None
        self._last_activity_at: float | None = None
        self._configure_default()
        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(100)
        self._poll_timer.timeout.connect(self._drain_events)
        self._poll_timer.start()
        self._heartbeat = QTimer(self)
        self._heartbeat.setInterval(1000)
        self._heartbeat.timeout.connect(self._update_heartbeat)
        self._heartbeat.start()

    @property
    def worker_active(self) -> bool:
        return bool(self._worker and self._worker.is_alive())

    @property
    def entry(self) -> PublicApparatusEntry:
        return self.catalog.resolve(
            self.state.apparatus,
            operating_mode=REALITY_LABELS[self.state.reality],
            discipline=self.state.discipline,
        )

    @property
    def definition(self):
        return self.registry.get(self.entry.internal_id)

    @property
    def whisper_model(self) -> str:
        mode = QUALITY_MODES[self.state.quality]
        return str(self.base_config.quality_modes.get(mode, {}).get("whisper_model", self.base_config.whisper_model))

    def realities(self) -> tuple[str, ...]:
        return tuple(REALITY_LABELS)

    def disciplines(self, reality: str | None = None) -> tuple[str, ...]:
        mode = REALITY_LABELS[reality or self.state.reality]
        available = self.catalog.entries(operating_mode=mode, primary_only=True)
        ids = {entry.discipline for entry in available}
        return tuple(row.name for row in self.catalog.disciplines() if row.id in ids)

    def apparatuses(self, reality: str | None = None, discipline: str | None = None) -> tuple[PublicApparatusEntry, ...]:
        mode = REALITY_LABELS[reality or self.state.reality]
        selected_discipline = discipline or self.state.discipline
        return self.catalog.entries(operating_mode=mode, discipline=selected_discipline, primary_only=True)

    def _configure_default(self) -> None:
        entries = self.apparatuses()
        preferred = next((entry for entry in entries if entry.public_name == "Transposition"), None)
        if preferred is None:
            preferred = next((entry for entry in entries if entry.invokable), entries[0])
        self.state.apparatus = preferred.public_name
        self.state.parameters = self.registry.get(preferred.internal_id).parameter_defaults
        self.state.quality = next(
            (label for label, mode in QUALITY_MODES.items() if mode == self.base_config.transcription_mode),
            "Study",
        )
        self.state.matching = FILTER_DISPLAY_NAMES.get(self.base_config.cinematic_filter, "Balanced")
        self._resize_film_slots(preferred.minimum_films)

    def _resize_film_slots(self, count: int) -> None:
        count = max(1, count)
        current = list(self.state.films)
        self.state.films = (current + [None] * count)[:count]

    def configure(
        self,
        *,
        reality: str,
        discipline: str,
        apparatus: str,
        films: list[Path | None],
        output_dir: Path,
        quality: str,
        matching: str,
        parameters: dict[str, Any] | None = None,
    ) -> None:
        if self.worker_active:
            raise ValueError("The apparatus cannot be reconfigured during an invocation.")
        mode = REALITY_LABELS[reality]
        entry = self.catalog.resolve(apparatus, operating_mode=mode, discipline=discipline)
        definition = self.registry.get(entry.internal_id)
        selected_count = max(definition.minimum_films, len(films))
        if definition.maximum_films is not None:
            selected_count = min(selected_count, definition.maximum_films)
        entry.validate_film_count(selected_count)
        self.state.reality = reality
        self.state.discipline = discipline
        self.state.apparatus = entry.public_name
        self.state.films = (list(films) + [None] * selected_count)[:selected_count]
        self.state.output_dir = Path(output_dir).expanduser()
        self.state.quality = quality if quality in QUALITY_LABELS else "Study"
        self.state.matching = matching if matching in MATCHING_LABELS else "Balanced"
        self.state.parameters = definition.normalize_parameters(parameters)
        self.state.completed = False
        self.state.operation = "READY TO INVOKE" if all(self.state.films) else "AWAITING MATERIAL"
        self.state.machine_state = "READY" if all(self.state.films) else "DORMANT"
        self._append_journal(f"Configured {entry.public_name} / {self.state.quality}")
        self.changed.emit()

    def admit_film(self, path: Path, index: int | None = None) -> None:
        if self.worker_active:
            return
        if index is None:
            index = next((i for i, value in enumerate(self.state.films) if value is None), 0)
        if index >= len(self.state.films):
            maximum = self.definition.maximum_films
            if maximum is not None and len(self.state.films) >= maximum:
                raise ValueError(f"{self.entry.public_name} accepts at most {maximum} films.")
            self.state.films.append(None)
        self.state.films[index] = Path(path)
        self.state.operation = "READY TO INVOKE" if all(self.state.films) else "AWAITING MATERIAL"
        self.state.machine_state = "READY" if all(self.state.films) else "DORMANT"
        self.changed.emit()

    def cycle_quality(self, delta: int = 1) -> None:
        index = QUALITY_LABELS.index(self.state.quality)
        self.state.quality = QUALITY_LABELS[(index + delta) % len(QUALITY_LABELS)]
        self.changed.emit()

    def cycle_matching(self, delta: int = 1) -> None:
        index = MATCHING_LABELS.index(self.state.matching)
        self.state.matching = MATCHING_LABELS[(index + delta) % len(MATCHING_LABELS)]
        self.changed.emit()

    def selected_config(self) -> AppConfig:
        self.entry.require_invokable()
        films = [Path(path).expanduser().resolve() for path in self.state.films if path is not None]
        self.entry.validate_film_count(len(films))
        self.definition.validate_film_count(len(films))
        if self.definition.supported_output_forms != ("full_length",):
            raise ValueError(f"{self.definition.name} does not declare the required full-length output contract.")
        for index, film in enumerate(films):
            if not film.exists():
                label = "Anchor Film" if index == 0 else f"Film {chr(65 + index)}"
                raise ValueError(f"{label} does not exist: {film}")
        if self.definition.is_multiworld and len({str(path).casefold() for path in films}) != len(films):
            raise ValueError("Choose distinct films for a Multiworld run.")
        return self.base_config.with_films(films).with_overrides(
            mode=QUALITY_MODES[self.state.quality],
            output_dir=self.state.output_dir.resolve(),
            cinematic_filter=MATCHING_IDS[self.state.matching],
        )

    def save_recipe(self, path: Path) -> Path:
        config = self.selected_config()
        roles = {"films": [str(item) for item in config.films]}
        recipe = FilterRecipe.create(
            self.definition.id,
            input_media_roles=roles,
            parameters=self.state.parameters,
            output_settings={
                "form": "full_length",
                "output_directory": str(config.output_dir),
                "quality": self.state.quality,
                "matching": self.state.matching,
            },
            random_seed=int(self.state.parameters.get("seed", 1)),
            target_duration=None,
            requested_analysis_backends={
                "diarization": config.speaker_diarization_backend,
                "transcription": config.whisper_model,
            },
        )
        saved = save_recipe(recipe, Path(path))
        self._append_journal(f"Recipe saved: {saved}")
        self.notice.emit(f"Recipe saved: {saved.name}")
        return saved

    def load_recipe(self, path: Path) -> tuple[str, ...]:
        if self.worker_active:
            raise ValueError("A recipe cannot be loaded during an invocation.")
        loaded = load_recipe(Path(path), registry=self.registry)
        recipe = loaded.recipe
        entry = self.catalog.get(recipe.filter_id)
        reality = "One Film" if entry.operating_mode == OperatingMode.SOLITARY else "Several Films"
        discipline = self.catalog.discipline(entry.discipline).name
        roles = recipe.input_media_roles
        if "films" in roles:
            films = [Path(item) for item in roles["films"]]
        elif "film" in roles:
            films = [Path(str(roles["film"]))]
        else:
            films = [
                Path(str(roles[key]))
                for key in ("destination_video", "source_dialogue")
                if roles.get(key)
            ]
        output_dir = Path(str(recipe.output_settings.get("output_directory") or self.state.output_dir))
        quality = str(recipe.output_settings.get("quality") or self.state.quality)
        matching = str(recipe.output_settings.get("matching") or self.state.matching)
        self.configure(
            reality=reality,
            discipline=discipline,
            apparatus=entry.public_name,
            films=films,
            output_dir=output_dir,
            quality=quality,
            matching=matching,
            parameters=recipe.parameters,
        )
        notes = (*loaded.migrations, *loaded.warnings)
        self._append_journal(f"Recipe loaded: {path}")
        for note in notes:
            self._append_journal(note)
        return notes

    def invoke(self) -> None:
        if self.worker_active:
            return
        try:
            config = self.selected_config()
            report = preflight_media_inputs(config.films, output_dir=config.output_dir)
            parameters = self.definition.normalize_parameters(self.state.parameters)
        except (OSError, ValueError) as exc:
            self.error.emit("Material could not be verified", str(exc))
            return
        self._cancel_requested = False
        self._run_started_at = time.time()
        self._last_activity_at = self._run_started_at
        self.state.running = True
        self.state.completed = False
        self.state.cancelling = False
        self.state.overall_progress = 0.02
        self.state.stage_progress = 0.0
        self.state.active_stage_index = -1
        self.state.operation = "PREPARING INVOCATION"
        self.state.machine_state = "ACTIVE"
        self.state.summary = f"INPUT CONTRACT READY · {report['predicted_output_duration']:.1f} SECONDS"
        self._append_journal(f"Invocation initiated: {self.entry.public_name}")
        self.changed.emit()
        self._worker = threading.Thread(
            target=self._run_pipeline,
            args=(config, self.definition.id, parameters),
            name="cinelingus-qt-pipeline",
            daemon=True,
        )
        self._worker.start()

    def cancel(self) -> None:
        if not self.worker_active:
            return
        self._cancel_requested = True
        self.state.cancelling = True
        self.state.machine_state = "CANCELLING"
        self.state.operation = "STOPPING AT SAFE BOUNDARY"
        self._append_journal("Cancellation requested")
        self.changed.emit()

    def _run_pipeline(self, config: AppConfig, filter_id: str, parameters: dict[str, Any]) -> None:
        writer = QueueWriter(self._events)
        try:
            with contextlib.redirect_stdout(writer), contextlib.redirect_stderr(writer):
                with exclusive_output_run(config.output_dir, filter_id) as run_lease:
                    pipeline = Pipeline(
                        config,
                        cancel_check=lambda: self._cancel_requested,
                        stage_callback=lambda stage: self._events.put(("stage", stage)),
                    )
                    result = pipeline.execute_configuration(filter_id, force=False, parameters=parameters)
                    output = Path(result.outputs["video"])
                    combined = {**result.outputs, **result.artifacts}
                    evidence = [
                        Path(value)
                        for key, value in combined.items()
                        if key in {
                            "filter_acceptance", "filter_recipe", "montage_render_acceptance",
                            "alteration_acceptance", "configuration_outcome",
                        }
                    ]
                    verify_filter_execution(
                        run_lease,
                        requested_filter_id=filter_id,
                        evidence_paths=evidence,
                        output=output,
                    )
            schedule_path = pipeline.destination.cache_dir / "replacement_schedule.json"
            performance_summary = {}
            quality_summary = {}
            if schedule_path.exists():
                try:
                    performance_summary = json.loads(schedule_path.read_text(encoding="utf-8")).get("performance_summary", {})
                except (OSError, ValueError):
                    performance_summary = {}
            problem_path = pipeline.config.output_dir / "problem_regions.json"
            if problem_path.exists():
                try:
                    problem_report = json.loads(problem_path.read_text(encoding="utf-8"))
                    quality_summary = _quality_summary_from_problem_report(problem_report)
                except (OSError, TypeError, ValueError):
                    quality_summary = {}
            self._events.put(("complete", {
                "output": output,
                "performance_summary": performance_summary,
                "quality_summary": quality_summary,
            }))
        except Exception:
            if self._cancel_requested:
                self._events.put(("cancelled", None))
            else:
                self._events.put(("failed", traceback.format_exc()))

    def _drain_events(self) -> None:
        dirty = False
        while True:
            try:
                event, payload = self._events.get_nowait()
            except queue.Empty:
                break
            dirty = True
            self._last_activity_at = time.time()
            if event == "log":
                text = str(payload)
                for line in text.splitlines():
                    if line.strip():
                        self._append_journal(line.strip())
                        stage_key = stage_key_for_diagnostic(line)
                        if stage_key:
                            self._apply_stage(stage_key)
            elif event == "stage":
                self._apply_stage(str(payload))
            elif event == "complete":
                if isinstance(payload, dict):
                    output = Path(payload["output"])
                    performance_summary = dict(payload.get("performance_summary") or {})
                    quality_summary = dict(payload.get("quality_summary") or {})
                else:
                    output = Path(payload)
                    performance_summary = {}
                    quality_summary = {}
                self.state.running = False
                self.state.completed = True
                self.state.cancelling = False
                self.state.overall_progress = 1.0
                self.state.stage_progress = 1.0
                self.state.remaining = "00:00"
                self.state.completion_time = datetime.now().strftime("%I:%M %p").lstrip("0")
                self.state.active_stage_index = len(STAGE_SEQUENCE)
                self.state.operation = "ARTIFACT READY FOR REVIEW"
                self.state.machine_state = "COMPLETE"
                self.state.summary = f"ARCHIVED · {output.name.upper()}"
                self.state.performance_summary = performance_summary
                if performance_summary:
                    total = int(performance_summary.get("destination_performance_count", 0) or 0)
                    coupled = int(performance_summary.get("performance_couplings", 0) or 0)
                    adapted = int(performance_summary.get("adapted_performances", 0) or 0)
                    turns = int(performance_summary.get("turn_sequence_matches", 0) or 0)
                    fallback = int(performance_summary.get("linewise_fallbacks", 0) or 0)
                    preserved = int(performance_summary.get("preserved_original_regions", 0) or 0)
                    suppressed = int(performance_summary.get("suppressed_unreplaced_regions", 0) or 0)
                    reconstructed = int(performance_summary.get("ambience_reconstructed_regions", 0) or 0)
                    silence_fallback = int(performance_summary.get("ambience_silence_fallback_regions", 0) or 0)
                    residue = str(performance_summary.get("voice_residue") or "NOT_MEASURED")
                    residue_label = "RESIDUE NOT TESTED" if "NOT_" in residue else f"RESIDUE {residue.replace('_', ' ')}"
                    unmatched_label = (
                        f"SUPPRESSED {suppressed}"
                        if "suppressed_unreplaced_regions" in performance_summary
                        else f"PRESERVED {preserved}"
                    )
                    self.state.summary = (
                        f"COUPLINGS {coupled}/{total} · ADAPTED {adapted} · TURN SEQUENCES {turns} · "
                        f"LINEWISE {fallback} · {unmatched_label} · AMBIENCE {reconstructed}/{reconstructed + silence_fallback} · "
                        f"{residue_label}"
                    )
                    review_count = int(quality_summary.get("problem_count", 0) or 0)
                    if review_count:
                        self.state.summary += f" · REVIEW {review_count}"
                self.state.last_output = output
                self._append_journal(f"Artifact archived: {output}")
                self.run_finished.emit(str(output))
            elif event == "cancelled":
                self.state.running = False
                self.state.cancelling = False
                self.state.machine_state = "CANCELLED"
                self.state.operation = "INVOCATION CANCELLED"
                self.state.summary = "RUN CANCELLED AT SAFE BOUNDARY"
            elif event == "failed":
                self.state.running = False
                self.state.cancelling = False
                self.state.machine_state = "FAULT"
                self.state.operation = "INVOCATION INTERRUPTED"
                self.state.summary = "TECHNICAL RECORD REQUIRES ATTENTION"
                self._append_journal(str(payload))
                self.error.emit("Invocation failed", str(payload))
        if dirty:
            self.changed.emit()

    def _apply_stage(self, raw: str) -> None:
        if not raw:
            return
        presentation = PIPELINE_STAGE_PRESENTATIONS.get(raw)
        if presentation:
            raw, operation = presentation
        else:
            operation = ""
        normalized = raw.removeprefix("diarization:")
        key = stage_sequence_key(normalized.split(":", 1)[0])
        if key is None:
            if raw.startswith("diarization:"):
                key = "destination_speech"
            else:
                return
        index = next(i for i, (stage_key, _label) in enumerate(STAGE_SEQUENCE) if stage_key == key)
        if index < self.state.active_stage_index:
            return
        self.state.active_stage_index = index
        self.state.overall_progress = max(self.state.overall_progress, STAGE_PROGRESS_FLOORS[key] / 100.0)
        self.state.stage_progress = 0.1
        self.state.operation = operation or STAGE_SEQUENCE[index][1]
        self.state.machine_state = "ACTIVE"
        self._append_journal(f"Stage: {self.state.operation}")

    def _update_heartbeat(self) -> None:
        if not self.state.running or self._run_started_at is None:
            return
        now = time.time()
        elapsed = now - self._run_started_at
        self.state.elapsed = format_clock_duration(elapsed)
        if self.state.overall_progress > 0.02:
            total = elapsed / self.state.overall_progress
            remaining = max(0.0, total - elapsed)
            self.state.remaining = format_clock_duration(remaining)
            self.state.completion_time = (datetime.now() + timedelta(seconds=remaining)).strftime("%I:%M %p").lstrip("0")
        else:
            self.state.remaining = "CALCULATING..."
            self.state.completion_time = "CALCULATING..."
        idle = now - (self._last_activity_at or now)
        self.state.stage_progress = min(0.95, max(self.state.stage_progress, 0.1 + idle / 900.0))
        self.changed.emit()

    def _append_journal(self, text: str) -> None:
        self.state.journal.extend(line for line in text.splitlines() if line.strip())
        if len(self.state.journal) > 500:
            self.state.journal = self.state.journal[-500:]

    def technical_record(self) -> str:
        return "\n".join(self.state.journal) or "No technical events recorded."

    def default_media_directory(self) -> Path:
        return default_input_directory()
