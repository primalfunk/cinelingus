from __future__ import annotations

import contextlib
import os
import queue
import re
import threading
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from .cinematic_filters import FILTER_DISPLAY_NAMES
from .build_info import format_build_identification
from .config import load_config
from .filter_lab.presentation import input_field_ids, parameter_help
from .filter_lab.registry import default_filter_registry
from .filter_lab.gui_controller import (
    current_filter_definition,
    load_recipe_dialog,
    save_recipe_dialog,
    selected_filter_parameters,
    sync_filter_family,
    sync_filter_mode,
)
from .mutations import MUTATION_DISPLAY_NAMES
from .pipeline import Pipeline
from .operator_language import (
    MODE_DESCRIPTIONS,
    MODE_GLYPHS,
    TRANSPOSITION,
    contains_traceback,
    display_mode_name,
    internal_mode_name,
    operator_message_for_log,
    stage_key_for_diagnostic,
    stage_message,
)
from .progress import ProgressState, format_progress_status
from .run_timing import completed_stage_text, estimate_overall_remaining
from .speakers import diarization_setup_status
from .ui_definitions import setting_definition
from .review import (
    PERFORMANCE_REVIEW_FILTERS,
    REVIEW_FILTERS,
    REVIEW_LABELS,
    apply_performance_review_label,
    apply_review_label,
    filtered_mapping_indices,
    filtered_performance_rows,
    performance_mapping_indices,
    performance_review_row_values,
    performance_review_summary,
    review_row_values,
    review_summary,
    write_review_notes,
)
from .util import read_json, write_json
from .whisper_backend import whisper_runtime
from .validation import validate_artifact

VIDEO_TYPES = [
    ("Video files", "*.mp4 *.mkv *.mov *.avi *.mpg *.mpeg"),
    ("All files", "*.*"),
]


FILTER_REGISTRY = default_filter_registry()
FILTER_FAMILY_DISPLAY_NAMES = {family.name: family.id for family in FILTER_REGISTRY.families()}
FILTER_DEFINITIONS_BY_NAME = {definition.name: definition for definition in FILTER_REGISTRY.definitions()}
TRANSFORMATION_CHOICES = ["Transposition", "Self Shuffle", "Echo", "Drift", "Contagion", "Possession", "Foreshadow", "Bloom"]
TRANSFORMATION_MUTATIONS = {
    display_mode_name(definition.name): definition.implementation_key
    for definition in FILTER_REGISTRY.definitions(implemented_only=True)
    if definition.implementation_key != "movie_masher"
}
SINGLE_FILM_TRANSFORMATIONS = set(TRANSFORMATION_CHOICES) - {TRANSPOSITION}
WORKFLOW_CHOICES = ["Best Short Remix", "Full Movie Remix"]
REMIX_PREFERENCES = {"Balanced": "balanced", "Best realism": "realism", "Funniest result": "funniest"}
TARGET_LENGTHS = {"2 minutes": 120.0, "3 minutes": 180.0, "5 minutes": 300.0}

QUALITY_PRESETS = {
    "Preview": "fast_preview",
    "Balanced": "balanced",
    "Precision": "quality",
}
QUALITY_PRESET_DESCRIPTIONS = {
    "Preview": "A faster examination for exploratory work",
    "Balanced": "A measured balance of speed and fidelity",
    "Precision": "A more exacting examination for final work",
}
HIGHLIGHT_BUCKET_LABELS = {
    "most_convincing": "Most Convincing",
    "funniest": "Funniest",
    "most_awkward": "Most Awkward",
    "most_improved_matches": "Most Improved",
    "needs_attention": "Needs Attention",
}
HIGHLIGHT_BUCKETS = tuple(HIGHLIGHT_BUCKET_LABELS)
HIGHLIGHT_BUCKET_BY_LABEL = {label: key for key, label in HIGHLIGHT_BUCKET_LABELS.items()}
STAGE_LABELS = {key: stage_message(key).title for key in (
    "inspect", "source_dialogue", "clips", "destination_speech", "performances", "schedule", "render_audio", "render_video", "finalize"
)}
STAGE_SEQUENCE = (
    ("inspect", "Specimens catalogued"),
    ("source_dialogue", "Spoken record isolated"),
    ("destination_speech", "Recurring voices examined"),
    ("performances", "Performances observed"),
    ("schedule", "Experiment arranged"),
    ("render_audio", "Reconstruction"),
    ("finalize", "Artifact examined"),
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


def stage_sequence_key(stage_key: str | None) -> str | None:
    aliases = {"clips": "source_dialogue", "render_video": "render_audio"}
    normalized = aliases.get(str(stage_key or ""), str(stage_key or ""))
    return normalized if any(key == normalized for key, _label in STAGE_SEQUENCE) else None


def diarization_chunk_progress(stage: str | None) -> tuple[int, int] | None:
    match = re.search(r"_chunk_(\d+)_of_(\d+)$", str(stage or ""))
    if not match:
        return None
    completed, total = int(match.group(1)), int(match.group(2))
    if completed < 0 or total <= 0:
        return None
    return min(completed, total), total


def single_film_input_needs_explicit_choice(
    transformation: str,
    selected_path: Path,
    default_path: Path,
    *,
    selected_by_user: bool,
) -> bool:
    if transformation not in SINGLE_FILM_TRANSFORMATIONS:
        return False
    if selected_by_user:
        return False
    try:
        return selected_path.resolve() == default_path.resolve()
    except OSError:
        return selected_path == default_path


def required_input_fields(transformation: str) -> tuple[str, ...]:
    return input_field_ids(FILTER_REGISTRY.get(internal_mode_name(transformation)))


def quality_preset_mode(label: str) -> str:
    legacy = {"Fast Preview": "Preview", "High Accuracy": "Precision"}
    return QUALITY_PRESETS.get(legacy.get(label, label), "balanced")


def quality_preset_label(mode: str) -> str:
    for label, value in QUALITY_PRESETS.items():
        if value == mode:
            return label
    return "Balanced"


def target_length_seconds(label: str) -> float:
    return TARGET_LENGTHS.get(label, 180.0)


def workflow_uses_target_length(workflow: str) -> bool:
    return workflow == "Best Short Remix"


def remix_preference_id(label: str) -> str:
    return REMIX_PREFERENCES.get(label, "balanced")


def stage_key_for_log_line(line: str) -> str | None:
    return stage_key_for_diagnostic(line)


def plain_status_for_log_line(line: str) -> str | None:
    key = stage_key_for_log_line(line)
    return STAGE_LABELS.get(key) if key else None


def quality_detail(label: str) -> str:
    mode = quality_preset_mode(label)
    visible_label = quality_preset_label(mode)
    purpose = QUALITY_PRESET_DESCRIPTIONS.get(visible_label, QUALITY_PRESET_DESCRIPTIONS["Balanced"])
    extra = " This examination may require substantially more time." if mode == "quality" else ""
    return f"{purpose}.{extra}"


def quality_runtime_warning(label: str, runtime: dict) -> str | None:
    mode = quality_preset_mode(label)
    if not runtime.get("available"):
        return "The transcription instrument is unavailable. Review the Technical Record before beginning."
    if mode == "quality" and not runtime.get("cuda_available"):
        return (
            "Precision fidelity will continue without accelerated examination and may require substantially more time. "
            "Balanced fidelity is recommended for routine work."
        )
    return None


def speaker_diarization_detail(config) -> str:
    if not getattr(config, "enable_speaker_awareness", True):
        return "Recurring-voice examination is disabled."
    backend = getattr(config, "speaker_diarization_backend", "heuristic")
    if backend == "heuristic":
        return "Recurring voices will be estimated from temporal structure."
    status = diarization_setup_status(backend=backend)
    if status.get("available"):
        return "Recurring-voice examination is available."
    return "Recurring voices will be estimated by an alternate method."


def run_truth_summary(
    *,
    transformation: str,
    destination: Path,
    source: Path,
    output_dir: Path,
    quality: str,
    matching_style: str,
    workflow: str = "Best Short Remix",
) -> str:
    mode = display_mode_name(transformation)
    if mode == TRANSPOSITION:
        media = f"Destination film: {compact_path(destination)} | Source film: {compact_path(source)}"
    else:
        media = f"Film: {compact_path(destination)}"
    return (
        f"Experiment: {mode} | {media} | Archive: {compact_path(output_dir)} | "
        f"Form: {workflow} | Fidelity: {quality} | Previous observations are reused only when material and settings match."
    )


def completed_run_truth_summary(output: Path, output_dir: Path, transformation: str) -> str:
    mode = display_mode_name(transformation)
    short_report = output.parent / "output_report.json"
    with contextlib.suppress(Exception):
        data = read_json(short_report)
        outputs = data.get("outputs") or {}
        video = outputs.get("video") or output
        audio = outputs.get("audio")
        selected = data.get("selection_summary") or {}
        source_bits = [f"Output: {compact_path(Path(video))}"]
        provenance = data.get("audio_provenance") or {}
        provenance_inputs = provenance.get("inputs") or {}
        if provenance_inputs.get("destination_video"):
            source_bits.append(f"Visual: {compact_path(Path(provenance_inputs.get('destination_video')))}")
        if provenance_inputs.get("source_dialogue"):
            source_bits.append(f"Dialogue: {compact_path(Path(provenance_inputs.get('source_dialogue')))}")
        if audio:
            source_bits.append(f"Audio: {compact_path(Path(audio))}")
        if provenance.get("status"):
            source_bits.append(f"Audio check: {provenance.get('status')}")
        if selected.get("vignette_count"):
            source_bits.append(f"Vignettes: {selected.get('vignette_count')}")
        if selected.get("candidate_id"):
            source_bits.append(f"Candidate: {selected.get('candidate_id')}")
        return f"Last completed experiment: {mode} | " + " | ".join(source_bits)
    if mode != TRANSPOSITION:
        report = output.parent / "mutation_report.json"
        with contextlib.suppress(Exception):
            data = read_json(report)
            name = display_mode_name((data.get("mutation_filter") or {}).get("display_name") or mode)
            source = data.get("source_film") or data.get("source_path") or "unknown"
            video = ((data.get("outputs") or {}).get("video") or output)
            return f"Last completed experiment: {name} | Film: {compact_path(Path(source))} | Artifact: {compact_path(Path(video))}"
    report = output_dir / "run_report.json"
    with contextlib.suppress(Exception):
        data = read_json(report)
        inputs = data.get("inputs") or {}
        destination = ((inputs.get("destination_video") or {}).get("path") or "unknown")
        source = ((inputs.get("source_dialogue") or {}).get("path") or "unknown")
        video = ((data.get("outputs") or {}).get("video_path") or output)
        return (
            f"Last completed experiment: Transposition | Destination: {compact_path(Path(destination))} | "
            f"Source: {compact_path(Path(source))} | Artifact: {compact_path(Path(video))}"
        )
    return f"Last completed experiment: {mode} | Artifact: {compact_path(output)}"


def compact_path(path: Path | str, max_chars: int = 78) -> str:
    value = str(path)
    if len(value) <= max_chars:
        return value
    path_obj = Path(value)
    parent = path_obj.parent.name
    compact = f"...\\{parent}\\{path_obj.name}" if parent else f"...\\{path_obj.name}"
    if len(compact) <= max_chars:
        return compact
    return f"...{compact[-max_chars + 3:]}"


def finished_output_folder(output: Path | str | None, fallback_output_dir: Path | str) -> Path:
    if output:
        path = Path(output).expanduser()
        return path if path.is_dir() else path.parent
    return Path(fallback_output_dir).expanduser()


EMBLEM_RELATIVE_PATH = Path("assets") / "cinelingus_emblem.png"
EMBLEM_VARIANT_RELATIVE_PATHS = {
    True: Path("assets") / "cinelingus_emblem_header.png",
    False: Path("assets") / "cinelingus_emblem_hero.png",
}


def emblem_asset_path(root: Path) -> Path:
    return Path(root) / EMBLEM_RELATIVE_PATH


def emblem_variant_asset_path(root: Path, *, compact: bool) -> Path:
    return Path(root) / EMBLEM_VARIANT_RELATIVE_PATHS[compact]


def responsive_layout(width: int) -> str:
    return "wide" if int(width or 0) >= 980 else "compact"


def accessible_control_labels() -> dict[str, str]:
    return {
        "begin": "Start Cinelingus processing job",
        "cancel": "Cancel Cinelingus processing job",
        "source": "Choose source film",
        "destination": "Choose destination film",
        "technical_record": "Show exact technical processing log",
    }


class EmblemMark(tk.Canvas):
    def __init__(self, parent, *, root_dir: Path, compact: bool = False, **kwargs) -> None:
        size = 76 if compact else 260
        background = kwargs.pop("background", "#0d1014")
        super().__init__(parent, width=size, height=size, background=background, highlightthickness=0, **kwargs)
        self.root_dir = Path(root_dir)
        self.compact = compact
        self.image_asset: tk.PhotoImage | None = None
        self.loaded_asset_path: Path | None = None
        self.bind("<Configure>", self._redraw)
        self._load_asset()

    @property
    def uses_fallback(self) -> bool:
        return self.image_asset is None

    def _load_asset(self) -> None:
        variant_path = emblem_variant_asset_path(self.root_dir, compact=self.compact)
        path = variant_path if variant_path.exists() else emblem_asset_path(self.root_dir)
        if not path.exists():
            return
        try:
            image = tk.PhotoImage(file=str(path))
            if path == variant_path:
                self.image_asset = image
            else:
                target = 68 if self.compact else 240
                factor = max(1, (max(image.width(), image.height()) + target - 1) // target)
                self.image_asset = image.subsample(factor, factor)
            self.loaded_asset_path = path
        except tk.TclError:
            self.image_asset = None
            self.loaded_asset_path = None

    def _redraw(self, _event=None) -> None:
        self.delete("all")
        width = max(1, self.winfo_width())
        height = max(1, self.winfo_height())
        cx, cy = width / 2, height / 2
        if self.image_asset is not None:
            self.create_image(cx, cy, image=self.image_asset)
            return
        radius = max(16, min(width, height) * (0.34 if self.compact else 0.36))
        gold, cyan = "#b99b5e", "#83d8e8"
        self.create_oval(cx - radius, cy - radius, cx + radius, cy + radius, outline=gold, width=2)
        self.create_oval(cx - radius * 0.72, cy - radius * 0.72, cx + radius * 0.72, cy + radius * 0.72, outline="#66583d", width=1)
        self.create_arc(cx - radius * 0.5, cy - radius * 0.5, cx + radius * 0.5, cy + radius * 0.5, start=42, extent=276, outline=cyan, width=max(2, int(radius / 18)), style="arc")
        self.create_line(cx, cy - radius * 0.92, cx, cy - radius * 0.64, fill=gold, width=2)
        self.create_line(cx, cy + radius * 0.64, cx, cy + radius * 0.92, fill=gold, width=2)
        self.create_text(cx, cy, text="C", fill="#e8e1d2", font=("Georgia", max(14, int(radius * 0.72)), "bold"))
        if not self.compact:
            self.create_text(cx, cy + radius + 30, text="APPARATUS LINGUARUM", fill=gold, font=("Georgia", 9))


class QueueWriter:
    def __init__(self, output_queue: "queue.Queue[str]") -> None:
        self.output_queue = output_queue

    def write(self, text: str) -> int:
        if text:
            self.output_queue.put(text)
        return len(text)

    def flush(self) -> None:
        pass


class MovieMasherApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Cinelingus - Wingtip Studio Laboratory")
        self.geometry("1120x780")
        self.minsize(760, 620)

        self.root_dir = Path.cwd()
        self.base_config = load_config(self.root_dir)
        self.output_queue: "queue.Queue[str]" = queue.Queue()
        self.worker: threading.Thread | None = None
        self.last_output: Path | None = None
        self.console_log_path: Path | None = None
        self.run_started_at: float | None = None
        self.last_console_activity_at: float | None = None
        self.last_heartbeat_console_at: float | None = None
        self.last_journal_event_ids: set[str] = set()
        self.last_plain_stage: str = "Ready"
        self.diarization_active_stage: str | None = None
        self.active_stage_key: str | None = None
        self.active_stage_started_at: float | None = None
        self.completed_stage_durations: dict[str, float] = {}
        self.furthest_stage_index = -1
        self.cancel_requested = False
        self.wizard_step = 1

        self.destination_var = tk.StringVar(value=str(self.base_config.destination_video))
        self.source_var = tk.StringVar(value=str(self.base_config.source_dialogue))
        self.output_var = tk.StringVar(value=str(self.base_config.output_dir))
        self.status_var = tk.StringVar(value="Ready")
        self.output_path_var = tk.StringVar(value="")
        self.problem_summary_var = tk.StringVar(value="No run yet")
        self.current_truth_var = tk.StringVar(value="")
        self.last_truth_var = tk.StringVar(value="No experiment has yet been archived.")
        self.preview_path_var = tk.StringVar(value="")
        self.filter_labels = {display: key for key, display in FILTER_DISPLAY_NAMES.items()}
        self.filter_var = tk.StringVar(value=FILTER_DISPLAY_NAMES.get(self.base_config.cinematic_filter, "Balanced"))
        self.quality_var = tk.StringVar(value=quality_preset_label(self.base_config.transcription_mode))
        self.input_guidance_var = tk.StringVar(value="Choose the source and destination films required by this experiment.")
        self.speaker_detail_var = tk.StringVar(value=speaker_diarization_detail(self.base_config))
        self.stage_var = tk.StringVar(value="Awaiting material")
        self.current_operation_var = tk.StringVar(value="No operation in progress")
        self.live_elapsed_var = tk.StringVar(value="00:00")
        self.live_idle_var = tk.StringVar(value="00:00")
        self.live_eta_var = tk.StringVar(value="Calculating...")
        self.progress_percent_var = tk.StringVar(value="0%")
        self.specimen_var = tk.StringVar(value="No specimen selected")
        self.overall_eta_var = tk.StringVar(value="Estimated remaining: Calculating...")
        self.mode_description_var = tk.StringVar(value=MODE_DESCRIPTIONS[TRANSPOSITION])
        self.completion_summary_var = tk.StringVar(value="")
        self.setting_definition_var = tk.StringVar(value="")
        self.overall_progress_var = tk.DoubleVar(value=0.0)
        self.stage_progress_var = tk.DoubleVar(value=0.0)
        self.mode_var = tk.StringVar(value=TRANSPOSITION)
        self.family_var = tk.StringVar(value="Translation")
        self.filter_detail_var = tk.StringVar(value="")
        self.relationship_var = tk.StringVar(value="")
        self.filter_parameter_vars: dict[str, tk.Variable] = {}
        self.advanced_filter_controls_var = tk.BooleanVar(value=False)
        self.technical_record_var = tk.BooleanVar(value=False)
        self.workflow_var = tk.StringVar(value="Best Short Remix")
        self.target_length_var = tk.StringVar(value="3 minutes")
        self.remix_preference_var = tk.StringVar(value="Balanced")
        self.destination_selected_by_user = False
        self.source_selected_by_user = False
        self.mutation_labels = {display: key for key, display in MUTATION_DISPLAY_NAMES.items()}
        self.mutation_var = tk.StringVar(value=MUTATION_DISPLAY_NAMES.get("echo", "Echo"))

        self._build_ui()
        self._bind_truth_refresh()
        self._sync_mode_fields()
        self._show_wizard_step(1)
        self.bind("<Configure>", self._on_window_resize)
        self.after(100, self._drain_output_queue)
        self.after(1000, self._refresh_run_heartbeat)

    def _build_legacy_ui(self) -> None:
        self._configure_style()
        self.columnconfigure(0, weight=1)
        self.rowconfigure(3, weight=1)

        header = ttk.Frame(self, padding=(18, 16, 18, 8), style="App.TFrame")
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)
        ttk.Label(header, text="Cinelingus", style="Title.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(header, text="Filter Laboratory — transform dialogue, performance, identity, and time.", style="Subtitle.TLabel").grid(row=1, column=0, sticky="w", pady=(2, 0))
        ttk.Label(header, textvariable=self.status_var, style="Status.TLabel").grid(row=0, column=1, rowspan=2, sticky="e")

        input_frame = ttk.LabelFrame(self, text="1. Choose Material And Filter", padding=14)
        input_frame.grid(row=1, column=0, sticky="ew", padx=18, pady=(8, 8))
        input_frame.columnconfigure(1, weight=1)
        self.choice_frame = input_frame

        ttk.Label(input_frame, text="Filter family", style="Field.TLabel").grid(row=0, column=0, sticky="w", pady=5)
        family_box = ttk.Combobox(input_frame, textvariable=self.family_var, values=list(FILTER_FAMILY_DISPLAY_NAMES), state="readonly", width=24)
        family_box.grid(row=0, column=1, sticky="w", padx=10, pady=5)
        family_box.bind("<<ComboboxSelected>>", lambda _event: sync_filter_family(self))
        ttk.Label(input_frame, text="Filter", style="Field.TLabel").grid(row=1, column=0, sticky="w", pady=5)
        self.mode_box = ttk.Combobox(input_frame, textvariable=self.mode_var, values=TRANSFORMATION_CHOICES, state="readonly", width=24)
        self.mode_box.grid(row=1, column=1, sticky="w", padx=10, pady=5)
        self.mode_box.bind("<<ComboboxSelected>>", lambda _event: self._sync_mode_fields())
        ttk.Label(input_frame, textvariable=self.filter_detail_var, style="Hint.TLabel", wraplength=720).grid(row=2, column=0, columnspan=3, sticky="w", pady=(6, 4))
        ttk.Label(input_frame, textvariable=self.relationship_var, style="Section.TLabel", wraplength=760).grid(row=3, column=0, columnspan=3, sticky="w", pady=(2, 8))
        self.destination_widgets = self._path_row(input_frame, 4, "Destination video", self.destination_var, self._choose_destination)
        self.source_widgets = self._path_row(input_frame, 5, "Source dialogue", self.source_var, self._choose_source)
        self.output_widgets = self._path_row(input_frame, 6, "Output folder", self.output_var, self._choose_output_dir)
        self.continue_button = ttk.Button(input_frame, text="Continue to behavior", command=self._continue_to_quality, style="Primary.TButton")
        self.continue_button.grid(row=7, column=1, sticky="w", padx=10, pady=(12, 2))

        quality_frame = ttk.LabelFrame(self, text="2. Choose Quality", padding=14)
        quality_frame.grid(row=2, column=0, sticky="ew", padx=18, pady=8)
        quality_frame.columnconfigure(1, weight=1)
        self.quality_frame = quality_frame
        ttk.Label(quality_frame, text="Quality preset", style="Field.TLabel").grid(row=0, column=0, sticky="w", pady=5)
        quality_box = ttk.Combobox(
            quality_frame,
            textvariable=self.quality_var,
            values=list(QUALITY_PRESETS),
            state="readonly",
            width=24,
        )
        quality_box.grid(row=0, column=1, sticky="w", padx=10, pady=5)
        quality_box.bind("<<ComboboxSelected>>", lambda _event: self._sync_quality_detail())
        self.quality_detail_var = tk.StringVar(value=quality_detail(self.quality_var.get()))
        ttk.Label(quality_frame, textvariable=self.quality_detail_var, style="Hint.TLabel").grid(row=0, column=2, sticky="w", padx=(12, 0))
        ttk.Label(quality_frame, text="Output mode", style="Field.TLabel").grid(row=1, column=0, sticky="w", pady=5)
        ttk.Combobox(quality_frame, textvariable=self.workflow_var, values=WORKFLOW_CHOICES, state="readonly", width=24).grid(row=1, column=1, sticky="w", padx=10, pady=5)
        ttk.Label(quality_frame, text="Best Short Remix finds the strongest few minutes instead of trying to remix the whole movie.", style="Hint.TLabel").grid(row=1, column=2, sticky="w", padx=(12, 0))
        ttk.Label(quality_frame, text="Target length", style="Field.TLabel").grid(row=2, column=0, sticky="w", pady=5)
        ttk.Combobox(quality_frame, textvariable=self.target_length_var, values=list(TARGET_LENGTHS), state="readonly", width=24).grid(row=2, column=1, sticky="w", padx=10, pady=5)
        ttk.Label(quality_frame, text="Preference", style="Field.TLabel").grid(row=3, column=0, sticky="w", pady=5)
        ttk.Combobox(quality_frame, textvariable=self.remix_preference_var, values=list(REMIX_PREFERENCES), state="readonly", width=24).grid(row=3, column=1, sticky="w", padx=10, pady=5)
        ttk.Label(quality_frame, text="Matching style", style="Field.TLabel").grid(row=4, column=0, sticky="w", pady=5)
        ttk.Combobox(
            quality_frame,
            textvariable=self.filter_var,
            values=list(self.filter_labels),
            state="readonly",
            width=24,
        ).grid(row=4, column=1, sticky="w", padx=10, pady=5)
        self.filter_controls_frame = ttk.LabelFrame(quality_frame, text="Filter behavior", padding=(10, 8))
        self.filter_controls_frame.grid(row=5, column=0, columnspan=3, sticky="ew", pady=(10, 4))
        ttk.Label(quality_frame, textvariable=self.input_guidance_var, style="Hint.TLabel").grid(row=6, column=1, columnspan=2, sticky="w", padx=10, pady=(4, 0))
        ttk.Label(quality_frame, textvariable=self.speaker_detail_var, style="Hint.TLabel").grid(row=7, column=1, columnspan=2, sticky="w", padx=10, pady=(4, 0))
        ttk.Label(quality_frame, text="What these settings do", style="Section.TLabel").grid(row=8, column=0, sticky="nw", pady=(12, 0))
        ttk.Label(quality_frame, textvariable=self.setting_definition_var, style="Hint.TLabel", wraplength=680).grid(row=8, column=1, columnspan=2, sticky="w", padx=10, pady=(12, 0))
        quality_actions = ttk.Frame(quality_frame)
        quality_actions.grid(row=9, column=1, columnspan=2, sticky="w", padx=10, pady=(16, 2))
        ttk.Button(quality_actions, text="Make New Selections", command=lambda: self._show_wizard_step(1)).grid(row=0, column=0)
        ttk.Button(quality_actions, text="Save Recipe", command=lambda: save_recipe_dialog(self)).grid(row=0, column=1, padx=(10, 0))
        ttk.Button(quality_actions, text="Load Recipe", command=lambda: load_recipe_dialog(self)).grid(row=0, column=2, padx=(10, 0))
        self.start_button = ttk.Button(quality_actions, text="RUN", command=self._start_run, style="Primary.TButton")
        self.start_button.grid(row=0, column=3, padx=(10, 0))

        run_frame = ttk.LabelFrame(self, text="3. Run And Review", padding=14)
        run_frame.grid(row=3, column=0, sticky="nsew", padx=18, pady=8)
        run_frame.columnconfigure(0, weight=1)
        self.run_frame = run_frame
        run_frame.rowconfigure(3, weight=1)

        actions = ttk.Frame(run_frame)
        actions.grid(row=0, column=0, sticky="ew")
        actions.columnconfigure(6, weight=1)
        self.cancel_button = ttk.Button(actions, text="Cancel", command=self._cancel_run)
        self.cancel_button.grid(row=0, column=0, sticky="w")
        self.open_button = ttk.Button(actions, text="Open Output", command=self._open_output_folder)
        self.open_button.grid(row=0, column=1, sticky="w", padx=(8, 0))
        self.performance_review_button = ttk.Button(actions, text="Review Performances", command=self._open_performance_review)
        self.performance_review_button.grid(row=0, column=2, sticky="w", padx=(8, 0))
        self.review_button = ttk.Button(actions, text="Review Schedule", command=self._open_review)
        self.review_button.grid(row=0, column=3, sticky="w", padx=(8, 0))
        self.highlight_review_button = ttk.Button(actions, text="Review Highlights", command=self._open_highlight_review)
        self.highlight_review_button.grid(row=0, column=4, sticky="w", padx=(8, 0))

        truth_frame = ttk.LabelFrame(run_frame, text="Run Truth", padding=(10, 8))
        truth_frame.grid(row=1, column=0, sticky="ew", pady=(14, 8))
        truth_frame.columnconfigure(0, weight=1)
        ttk.Label(truth_frame, textvariable=self.current_truth_var, style="Hint.TLabel", wraplength=840).grid(row=0, column=0, sticky="ew")

        progress_frame = ttk.Frame(run_frame)
        progress_frame.grid(row=2, column=0, sticky="ew", pady=(8, 10))
        progress_frame.columnconfigure(1, weight=1)
        ttk.Label(progress_frame, text="Overall", style="Field.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Progressbar(progress_frame, variable=self.overall_progress_var, maximum=100).grid(row=0, column=1, sticky="ew", padx=(10, 0))
        ttk.Label(progress_frame, text="Current stage", style="Field.TLabel").grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Progressbar(progress_frame, variable=self.stage_progress_var, maximum=100).grid(row=1, column=1, sticky="ew", padx=(10, 0), pady=(6, 0))
        ttk.Label(progress_frame, textvariable=self.stage_var, style="Hint.TLabel").grid(row=2, column=1, sticky="w", padx=(10, 0), pady=(6, 0))
        ttk.Label(progress_frame, textvariable=self.overall_eta_var, style="Hint.TLabel").grid(row=3, column=1, sticky="w", padx=(10, 0), pady=(4, 0))

        body = ttk.PanedWindow(run_frame, orient="horizontal")
        body.grid(row=3, column=0, sticky="nsew")

        stage_panel = ttk.Frame(body, padding=(0, 0, 10, 0))
        stage_panel.columnconfigure(0, weight=1)
        ttk.Label(stage_panel, text="Progress", style="Section.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 8))
        self.stage_step_vars: dict[str, tk.StringVar] = {}
        for index, (key, label) in enumerate(STAGE_SEQUENCE, start=1):
            var = tk.StringVar(value=f"[ ] {label}")
            self.stage_step_vars[key] = var
            ttk.Label(stage_panel, textvariable=var, style="Step.TLabel").grid(row=index, column=0, sticky="w", pady=2)
        body.add(stage_panel, weight=1)

        console_panel = ttk.Frame(body)
        console_panel.columnconfigure(0, weight=1)
        console_panel.rowconfigure(1, weight=1)
        ttk.Label(console_panel, text="Details", style="Section.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 8))
        self.console = tk.Text(console_panel, wrap="word", height=12, state="disabled", relief="flat", padx=10, pady=8)
        self.console.grid(row=1, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(console_panel, command=self.console.yview)
        scrollbar.grid(row=1, column=1, sticky="ns")
        self.console.configure(yscrollcommand=scrollbar.set)
        body.add(console_panel, weight=3)

        output_frame = ttk.LabelFrame(self, text="Finished Work", padding=14)
        output_frame.grid(row=4, column=0, sticky="ew", padx=18, pady=(8, 18))
        output_frame.columnconfigure(1, weight=1)
        self.finished_frame = output_frame
        ttk.Label(output_frame, text="Movie", style="Field.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Entry(output_frame, textvariable=self.output_path_var, state="readonly").grid(
            row=0, column=1, sticky="ew", padx=(10, 0)
        )
        ttk.Button(output_frame, text="Open Movie", command=self._open_finished_movie).grid(row=0, column=2, padx=(8, 0))
        ttk.Button(output_frame, text="Problem Previews", command=self._open_problem_previews).grid(row=0, column=3, padx=(8, 0))
        ttk.Label(output_frame, textvariable=self.problem_summary_var, style="Hint.TLabel", wraplength=760).grid(row=1, column=1, columnspan=3, sticky="w", padx=(10, 0), pady=(8, 0))
        ttk.Label(output_frame, textvariable=self.last_truth_var, style="Hint.TLabel", wraplength=760).grid(row=2, column=1, columnspan=3, sticky="w", padx=(10, 0), pady=(4, 0))

    def _build_ui(self) -> None:
        self._configure_style()
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        self._build_brand_header()
        self.content_host = ttk.Frame(self, padding=(18, 8, 18, 18), style="App.TFrame")
        self.content_host.grid(row=1, column=0, sticky="nsew")
        self.content_host.columnconfigure(0, weight=1)
        self.content_host.rowconfigure(0, weight=1)
        self._build_setup_view()
        self._build_active_view()
        self._build_results_view()

    def _build_brand_header(self) -> None:
        header = ttk.Frame(self, padding=(18, 12, 18, 10), style="App.TFrame")
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(1, weight=1)
        self.header_emblem = EmblemMark(header, root_dir=self.root_dir, compact=True)
        self.header_emblem.grid(row=0, column=0, rowspan=3, sticky="w", padx=(0, 12))
        ttk.Label(header, text="CINELINGUS", style="Title.TLabel").grid(row=0, column=1, sticky="w")
        ttk.Label(header, text="Apparatus Linguarum Cinematicarum", style="Plate.TLabel").grid(row=1, column=1, sticky="w")
        ttk.Label(header, text="Wingtip Studio Laboratory - Plate XII", style="Subtitle.TLabel").grid(row=2, column=1, sticky="w")
        ttk.Label(header, textvariable=self.status_var, style="Status.TLabel").grid(row=0, column=2, rowspan=3, sticky="e")

    def _build_setup_view(self) -> None:
        setup = ttk.Frame(self.content_host, style="App.TFrame")
        setup.grid(row=0, column=0, sticky="nsew")
        setup.columnconfigure(0, weight=1)
        setup.rowconfigure(0, weight=1)
        self.choice_frame = setup
        self.quality_frame = setup
        self.setup_body = ttk.Frame(setup, style="App.TFrame")
        self.setup_body.grid(row=0, column=0, sticky="nsew")
        self.setup_body.columnconfigure(0, weight=3)
        self.setup_body.columnconfigure(1, weight=2)
        self.setup_body.rowconfigure(0, weight=1)
        controls = ttk.Frame(self.setup_body, padding=(0, 0, 14, 0), style="App.TFrame")
        controls.grid(row=0, column=0, sticky="nsew")
        controls.columnconfigure(0, weight=1)
        self.setup_controls = controls
        ttk.Label(controls, text="EXPERIMENT", style="PlateHeading.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 8))

        experiment_panel = ttk.LabelFrame(controls, text="Mode", padding=(12, 10))
        experiment_panel.grid(row=1, column=0, sticky="ew")
        experiment_panel.columnconfigure((0, 1), weight=1)
        self.mode_box = ttk.Combobox(experiment_panel, textvariable=self.mode_var, values=TRANSFORMATION_CHOICES, state="readonly")
        self.mode_buttons: list[ttk.Radiobutton] = []
        for index, name in enumerate(TRANSFORMATION_CHOICES):
            button = ttk.Radiobutton(
                experiment_panel,
                text=f"{MODE_GLYPHS.get(name, '◇')}  {name}",
                value=name,
                variable=self.mode_var,
                command=self._sync_mode_fields,
                style="Experiment.TRadiobutton",
                takefocus=True,
            )
            button.grid(row=index // 2, column=index % 2, sticky="ew", padx=3, pady=3)
            self.mode_buttons.append(button)
        ttk.Label(experiment_panel, textvariable=self.mode_description_var, style="Hint.TLabel", wraplength=600).grid(row=4, column=0, columnspan=2, sticky="w", pady=(8, 0))
        ttk.Label(experiment_panel, textvariable=self.relationship_var, style="Instrument.TLabel", wraplength=600).grid(row=5, column=0, columnspan=2, sticky="w", pady=(4, 0))

        material_panel = ttk.LabelFrame(controls, text="Material", padding=(12, 10))
        material_panel.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        material_panel.columnconfigure(1, weight=1)
        self.source_widgets = self._path_row(material_panel, 0, "Source film", self.source_var, self._choose_source)
        self.destination_widgets = self._path_row(material_panel, 1, "Destination film", self.destination_var, self._choose_destination)
        ttk.Label(material_panel, textvariable=self.input_guidance_var, style="Hint.TLabel", wraplength=600).grid(row=2, column=0, columnspan=3, sticky="w", pady=(6, 0))

        fidelity_panel = ttk.LabelFrame(controls, text="Fidelity", padding=(12, 8))
        fidelity_panel.grid(row=3, column=0, sticky="ew", pady=(10, 0))
        for index, label in enumerate(QUALITY_PRESETS):
            ttk.Radiobutton(fidelity_panel, text=label, value=label, variable=self.quality_var, command=self._sync_quality_detail, takefocus=True).grid(row=0, column=index, sticky="w", padx=(0, 18))
        self.quality_detail_var = tk.StringVar(value=quality_detail(self.quality_var.get()))
        ttk.Label(fidelity_panel, textvariable=self.quality_detail_var, style="Hint.TLabel", wraplength=600).grid(row=1, column=0, columnspan=3, sticky="w", pady=(6, 0))

        ttk.Checkbutton(controls, text="Advanced settings", variable=self.advanced_filter_controls_var, command=self._toggle_advanced, takefocus=True).grid(row=4, column=0, sticky="w", pady=(10, 0))
        self._build_advanced_panel(controls)
        self.start_button = ttk.Button(controls, text="BEGIN EXPERIMENT", command=self._start_run, style="Primary.TButton", takefocus=True)
        self.start_button.grid(row=6, column=0, sticky="ew", pady=(14, 0), ipady=4)
        self.continue_button = self.start_button

        hero = ttk.Frame(self.setup_body, padding=16, style="Hero.TFrame")
        hero.grid(row=0, column=1, sticky="nsew")
        hero.columnconfigure(0, weight=1)
        hero.rowconfigure(0, weight=1)
        self.hero_panel = hero
        self.hero_emblem = EmblemMark(hero, root_dir=self.root_dir)
        self.hero_emblem.grid(row=0, column=0, sticky="nsew")
        ttk.Label(hero, text="A serious instrument\nfor an impossible science.", style="Hero.TLabel", justify="center").grid(row=1, column=0, pady=(8, 12))

    def _build_advanced_panel(self, controls: ttk.Frame) -> None:
        self.advanced_frame = ttk.LabelFrame(controls, text="Advanced", padding=(12, 10))
        self.advanced_frame.columnconfigure(1, weight=1)
        rows = (
            ("Output form", self.workflow_var, WORKFLOW_CHOICES),
            ("Target length", self.target_length_var, list(TARGET_LENGTHS)),
            ("Selection preference", self.remix_preference_var, list(REMIX_PREFERENCES)),
            ("Matching behavior", self.filter_var, list(self.filter_labels)),
        )
        for row, (label, variable, values) in enumerate(rows):
            label_widget = ttk.Label(self.advanced_frame, text=label, style="Field.TLabel")
            label_widget.grid(row=row, column=0, sticky="w", pady=3)
            choice_widget = ttk.Combobox(self.advanced_frame, textvariable=variable, values=values, state="readonly", width=22)
            choice_widget.grid(row=row, column=1, sticky="w", padx=8, pady=3)
            if label == "Target length":
                self.target_length_widgets = (label_widget, choice_widget)
        self.full_movie_length_note = ttk.Label(
            self.advanced_frame,
            text="Complete film duration is preserved; no target length applies.",
            style="Hint.TLabel",
        )
        self.output_widgets = self._path_row(self.advanced_frame, 4, "Archive folder", self.output_var, self._choose_output_dir)
        self.filter_controls_frame = ttk.LabelFrame(self.advanced_frame, text="Experiment parameters", padding=(10, 8))
        self.filter_controls_frame.grid(row=5, column=0, columnspan=3, sticky="ew", pady=(8, 4))
        ttk.Label(self.advanced_frame, textvariable=self.speaker_detail_var, style="Hint.TLabel", wraplength=600).grid(row=6, column=0, columnspan=3, sticky="w", pady=(5, 0))
        ttk.Label(self.advanced_frame, textvariable=self.setting_definition_var, style="Hint.TLabel", wraplength=600).grid(row=7, column=0, columnspan=3, sticky="w", pady=(5, 0))
        recipe_actions = ttk.Frame(self.advanced_frame)
        recipe_actions.grid(row=8, column=0, columnspan=3, sticky="w", pady=(8, 0))
        ttk.Button(recipe_actions, text="Save Recipe", command=lambda: save_recipe_dialog(self)).grid(row=0, column=0)
        ttk.Button(recipe_actions, text="Load Recipe", command=lambda: load_recipe_dialog(self)).grid(row=0, column=1, padx=(8, 0))

    def _build_active_view(self) -> None:
        run_frame = ttk.Frame(self.content_host, style="App.TFrame")
        run_frame.columnconfigure(0, weight=1)
        run_frame.rowconfigure(2, weight=1)
        self.run_frame = run_frame
        operation = ttk.LabelFrame(run_frame, text="CURRENT OPERATION", padding=(14, 10))
        operation.grid(row=0, column=0, sticky="ew")
        operation.columnconfigure(0, weight=1)
        ttk.Label(operation, textvariable=self.current_operation_var, style="Operation.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(operation, textvariable=self.progress_percent_var, style="Percent.TLabel").grid(row=0, column=1, sticky="e")
        ttk.Progressbar(operation, variable=self.overall_progress_var, maximum=100, style="Instrument.Horizontal.TProgressbar").grid(row=1, column=0, columnspan=2, sticky="ew", pady=(8, 8))
        metric_row = ttk.Frame(operation)
        metric_row.grid(row=2, column=0, columnspan=2, sticky="ew")
        for column in range(4):
            metric_row.columnconfigure(column, weight=1)
        self._metric(metric_row, 0, "Elapsed", self.live_elapsed_var)
        self._metric(metric_row, 1, "No new observations", self.live_idle_var)
        self._metric(metric_row, 2, "Estimated remaining", self.live_eta_var)
        self._metric(metric_row, 3, "Current specimen", self.specimen_var)

        stages = ttk.Frame(run_frame, padding=(0, 10, 0, 8), style="App.TFrame")
        stages.grid(row=1, column=0, sticky="ew")
        self.stage_step_vars: dict[str, tk.StringVar] = {}
        for index, (key, label) in enumerate(STAGE_SEQUENCE):
            stages.columnconfigure(index, weight=1)
            var = tk.StringVar(value=f"○ {label}")
            self.stage_step_vars[key] = var
            ttk.Label(stages, textvariable=var, style="Step.TLabel", anchor="center").grid(row=0, column=index, sticky="ew", padx=3)

        journal_panel = ttk.LabelFrame(run_frame, text="LABORATORY JOURNAL", padding=(10, 8))
        journal_panel.grid(row=2, column=0, sticky="nsew")
        journal_panel.columnconfigure(0, weight=1)
        journal_panel.rowconfigure(0, weight=1)
        self.journal = tk.Text(journal_panel, wrap="word", height=6, state="disabled", relief="flat", padx=12, pady=10, takefocus=True)
        self.journal.configure(background="#151a20", foreground="#e8e1d2", insertbackground="#83d8e8", selectbackground="#314451", font=("Segoe UI", 10))
        self.journal.grid(row=0, column=0, sticky="nsew")
        journal_scroll = ttk.Scrollbar(journal_panel, command=self.journal.yview)
        journal_scroll.grid(row=0, column=1, sticky="ns")
        self.journal.configure(yscrollcommand=journal_scroll.set)

        run_actions = ttk.Frame(run_frame, padding=(0, 10, 0, 0), style="App.TFrame")
        run_actions.grid(row=3, column=0, sticky="ew")
        self.cancel_button = ttk.Button(run_actions, text="Cancel Experiment", command=self._cancel_run, takefocus=True)
        self.cancel_button.grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(run_actions, text="Technical Record", variable=self.technical_record_var, command=self._toggle_technical_record, takefocus=True).grid(row=0, column=1, sticky="w", padx=(10, 0))
        self.open_button = ttk.Button(run_actions, text="Open Archive", command=self._open_output_folder)
        self.open_button.grid(row=0, column=2, sticky="w", padx=(10, 0))

        self.technical_frame = ttk.LabelFrame(run_frame, text="TECHNICAL RECORD", padding=(8, 6))
        self.technical_frame.columnconfigure(0, weight=1)
        self.technical_frame.rowconfigure(0, weight=1)
        self.console = tk.Text(self.technical_frame, wrap="word", height=5, state="disabled", relief="flat", padx=10, pady=8, takefocus=True)
        self.console.configure(background="#101419", foreground="#c8ced4", insertbackground="#83d8e8", selectbackground="#314451", font=("Segoe UI", 9))
        self.console.grid(row=0, column=0, sticky="nsew")
        technical_scroll = ttk.Scrollbar(self.technical_frame, command=self.console.yview)
        technical_scroll.grid(row=0, column=1, sticky="ns")
        self.console.configure(yscrollcommand=technical_scroll.set)

    def _metric(self, parent: ttk.Frame, column: int, label: str, variable: tk.StringVar) -> None:
        cell = ttk.Frame(parent)
        cell.grid(row=0, column=column, sticky="ew", padx=(0, 12))
        ttk.Label(cell, text=label.upper(), style="MetricLabel.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(cell, textvariable=variable, style="MetricValue.TLabel").grid(row=1, column=0, sticky="w")

    def _build_results_view(self) -> None:
        finished = ttk.Frame(self.content_host, style="App.TFrame", padding=(20, 10))
        finished.columnconfigure(0, weight=1)
        self.finished_frame = finished
        ttk.Label(finished, text="EXPERIMENT COMPLETE", style="Completion.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(finished, textvariable=self.completion_summary_var, style="Summary.TLabel", justify="left", wraplength=920).grid(row=1, column=0, sticky="ew", pady=(14, 14))
        artifact_panel = ttk.LabelFrame(finished, text="Result", padding=(12, 10))
        artifact_panel.grid(row=2, column=0, sticky="ew")
        artifact_panel.columnconfigure(0, weight=1)
        ttk.Entry(artifact_panel, textvariable=self.output_path_var, state="readonly").grid(row=0, column=0, sticky="ew")
        result_actions = ttk.Frame(finished, style="App.TFrame")
        result_actions.grid(row=3, column=0, sticky="w", pady=(14, 0))
        ttk.Button(result_actions, text="Open Artifact", command=self._open_finished_movie, style="Primary.TButton").grid(row=0, column=0)
        ttk.Button(result_actions, text="Open Folder", command=self._open_output_folder).grid(row=0, column=1, padx=(8, 0))
        ttk.Button(result_actions, text="View Technical Record", command=self._show_completed_technical_record).grid(row=0, column=2, padx=(8, 0))
        ttk.Button(result_actions, text="Begin Another Experiment", command=lambda: self._show_wizard_step(1)).grid(row=0, column=3, padx=(8, 0))
        review_actions = ttk.Frame(finished, style="App.TFrame")
        review_actions.grid(row=4, column=0, sticky="w", pady=(10, 0))
        self.performance_review_button = ttk.Button(review_actions, text="Review Performances", command=self._open_performance_review)
        self.performance_review_button.grid(row=0, column=0)
        self.review_button = ttk.Button(review_actions, text="Review Schedule", command=self._open_review)
        self.review_button.grid(row=0, column=1, padx=(8, 0))
        self.highlight_review_button = ttk.Button(review_actions, text="Review Highlights", command=self._open_highlight_review)
        self.highlight_review_button.grid(row=0, column=2, padx=(8, 0))
        ttk.Label(finished, textvariable=self.problem_summary_var, style="Hint.TLabel", wraplength=920).grid(row=5, column=0, sticky="w", pady=(12, 0))
        ttk.Label(finished, textvariable=self.last_truth_var, style="Hint.TLabel", wraplength=920).grid(row=6, column=0, sticky="w", pady=(4, 0))

    def _toggle_advanced(self) -> None:
        if self.advanced_filter_controls_var.get():
            self.advanced_frame.grid(row=5, column=0, sticky="ew", pady=(8, 0))
            self._refresh_setting_definitions()
        else:
            self.advanced_frame.grid_remove()

    def _toggle_technical_record(self) -> None:
        if self.technical_record_var.get():
            self.technical_frame.grid(row=4, column=0, sticky="nsew", pady=(10, 0))
        else:
            self.technical_frame.grid_remove()

    def _show_completed_technical_record(self) -> None:
        self._show_wizard_step(3)
        self.technical_record_var.set(True)
        self._toggle_technical_record()

    def _on_window_resize(self, event) -> None:
        if event.widget is not self or not hasattr(self, "hero_panel"):
            return
        variant = responsive_layout(event.width)
        if getattr(self, "_layout_variant", None) == variant:
            return
        self._layout_variant = variant
        if variant == "wide":
            self.setup_controls.grid_configure(column=0, columnspan=1, padx=(0, 14))
            self.hero_panel.grid(row=0, column=1, sticky="nsew")
        else:
            self.hero_panel.grid_remove()
            self.setup_controls.grid_configure(column=0, columnspan=2, padx=0)

    def _show_wizard_step(self, step: int) -> None:
        self.wizard_step = step
        for frame in {self.choice_frame, self.run_frame, self.finished_frame}:
            frame.grid_remove()
        active = {1: self.choice_frame, 2: self.choice_frame, 3: self.run_frame, 4: self.finished_frame}.get(step, self.choice_frame)
        active.grid(row=0, column=0, sticky="nsew")
        if step in {1, 2}:
            self._refresh_setting_definitions()
            self.status_var.set("Ready for experiment")
        elif step == 3:
            self.status_var.set("Experiment in progress")
        elif step == 4:
            self.status_var.set("Experiment complete")

    def _continue_to_quality(self) -> None:
        try:
            self._selected_config()
        except (FileNotFoundError, ValueError) as exc:
            messagebox.showerror("Selections incomplete", str(exc))
            return
        self._show_wizard_step(2)

    def _refresh_setting_definitions(self) -> None:
        definition = current_filter_definition(self)
        length_definition = (
            f"Length - {self.target_length_var.get()}: Selects the approximate finished reel duration; candidate selection may vary within the configured safe range."
            if workflow_uses_target_length(self.workflow_var.get())
            else "Length - Complete film: Full Movie Remix preserves the complete film duration, so no short-reel target is applied."
        )
        lines = (
            f"Filter - {definition.name}: {definition.operational_description}\n{parameter_help(definition)}",
            f"Quality - {self.quality_var.get()}: {setting_definition('quality', self.quality_var.get())}",
            f"Output - {self.workflow_var.get()}: {setting_definition('workflow', self.workflow_var.get())}",
            length_definition,
            f"Preference - {self.remix_preference_var.get()}: {setting_definition('preference', self.remix_preference_var.get())}",
            f"Matching - {self.filter_var.get()}: {setting_definition('matching', self.filter_var.get())}",
        )
        self.setting_definition_var.set("\n".join(lines))

    def _cancel_run(self) -> None:
        self.cancel_requested = True
        self.cancel_button.configure(state="disabled")
        self.status_var.set("Cancelling after the current operation...")
        self._append_console("Cancellation requested; stopping at the next safe pipeline boundary.\n")
    def _configure_style(self) -> None:
        style = ttk.Style(self)
        with contextlib.suppress(tk.TclError):
            style.theme_use("clam")
        background, panel, text, muted, accent, cyan = "#0d1014", "#151a20", "#e8e1d2", "#929aa3", "#b99b5e", "#83d8e8"
        self.configure(background=background)
        style.configure(".", background=panel, foreground=text, fieldbackground="#20262e", bordercolor="#3c4249", font=("Segoe UI", 10))
        style.configure("App.TFrame", background=background)
        style.configure("Hero.TFrame", background=background, bordercolor="#4c4433", relief="solid", borderwidth=1)
        style.configure("TFrame", background=panel)
        style.configure("TLabel", background=panel, foreground=text)
        style.configure("TLabelframe", background=panel, foreground=text, bordercolor="#4c4433", relief="solid", borderwidth=1)
        style.configure("TLabelframe.Label", background=panel, foreground=accent, font=("Georgia", 10, "bold"))
        style.configure("Title.TLabel", background=background, foreground=text, font=("Georgia", 23, "bold"))
        style.configure("Plate.TLabel", background=background, foreground=accent, font=("Georgia", 11, "italic"))
        style.configure("PlateHeading.TLabel", background=background, foreground=accent, font=("Georgia", 12, "bold"))
        style.configure("Subtitle.TLabel", background=background, foreground=muted, font=("Segoe UI", 10))
        style.configure("Status.TLabel", background=background, foreground=cyan, font=("Segoe UI", 10, "bold"))
        style.configure("Field.TLabel", foreground=text, font=("Segoe UI", 9, "bold"))
        style.configure("Hint.TLabel", foreground=muted, font=("Segoe UI", 9))
        style.configure("Instrument.TLabel", foreground=cyan, font=("Segoe UI", 9))
        style.configure("Section.TLabel", foreground=accent, font=("Segoe UI", 10, "bold"))
        style.configure("Step.TLabel", background=background, foreground="#b9c0c8", font=("Segoe UI", 8))
        style.configure("Operation.TLabel", foreground=cyan, font=("Georgia", 15, "bold"))
        style.configure("Percent.TLabel", foreground=cyan, font=("Segoe UI", 13, "bold"))
        style.configure("MetricLabel.TLabel", foreground=muted, font=("Segoe UI", 8, "bold"))
        style.configure("MetricValue.TLabel", foreground=text, font=("Segoe UI", 10))
        style.configure("Completion.TLabel", background=background, foreground=accent, font=("Georgia", 24, "bold"))
        style.configure("Summary.TLabel", background=background, foreground=text, font=("Segoe UI", 11))
        style.configure("Hero.TLabel", background=background, foreground=muted, font=("Georgia", 12, "italic"))
        style.configure("Experiment.TRadiobutton", padding=(9, 7), background="#1a2027", foreground=text, bordercolor="#4c4433", relief="solid")
        style.map("Experiment.TRadiobutton", background=[("selected", "#243039"), ("active", "#202830")], foreground=[("selected", cyan)], bordercolor=[("selected", cyan)], focuscolor=[("focus", cyan)])
        style.configure("Primary.TButton", background=accent, foreground="#111318", font=("Segoe UI", 10, "bold"))
        style.map("Primary.TButton", background=[("active", "#d0bb7d")])
        style.configure("Instrument.Horizontal.TProgressbar", background=cyan, troughcolor="#20262e", bordercolor="#3c4249", lightcolor=cyan, darkcolor=cyan)
        style.map("TCombobox", fieldbackground=[("readonly", "#20262e")], foreground=[("readonly", text)])
        style.map("TButton", focuscolor=[("focus", cyan)])

    def _sync_quality_detail(self) -> None:
        if hasattr(self, "quality_detail_var"):
            self.quality_detail_var.set(quality_detail(self.quality_var.get()))

    def _mark_stage(self, active_key: str | None, *, finished: bool = False) -> None:
        if not hasattr(self, "stage_step_vars"):
            return
        for index, (key, label) in enumerate(STAGE_SEQUENCE):
            if finished or key in self.completed_stage_durations or index < self.furthest_stage_index:
                self.stage_step_vars[key].set(f"✓ {label}")
            elif key == active_key:
                self.stage_step_vars[key].set(f"● {label}")
            else:
                self.stage_step_vars[key].set(f"○ {label}")

    def _set_overall_progress(self, value: float, *, allow_decrease: bool = False) -> float:
        bounded = max(0.0, min(100.0, float(value)))
        if not allow_decrease:
            bounded = max(float(self.overall_progress_var.get() or 0.0), bounded)
        self.overall_progress_var.set(bounded)
        self.progress_percent_var.set(f"{bounded:.0f}%")
        return bounded

    def _complete_active_stage(self, now: float | None = None) -> None:
        if self.active_stage_key and self.active_stage_started_at:
            finished_at = now or time.time()
            self.completed_stage_durations[self.active_stage_key] = max(0.0, finished_at - self.active_stage_started_at)
        self.active_stage_key = None
        self.active_stage_started_at = None

    def _update_overall_eta(self) -> None:
        elapsed = time.time() - self.run_started_at if self.run_started_at else 0.0
        remaining = estimate_overall_remaining(elapsed, float(self.overall_progress_var.get() or 0.0))
        text = "estimating..." if remaining is None else format_duration(remaining)
        self.overall_eta_var.set(f"Overall completion estimate: {text} | Total elapsed: {format_duration(elapsed)}")
    def _path_row(self, parent: ttk.Frame, row: int, label: str, variable: tk.StringVar, command) -> tuple[ttk.Widget, ttk.Widget, ttk.Widget]:
        label_widget = ttk.Label(parent, text=label)
        entry_widget = ttk.Entry(parent, textvariable=variable)
        button_widget = ttk.Button(parent, text="Browse", command=command)
        label_widget.grid(row=row, column=0, sticky="w", pady=4)
        entry_widget.grid(row=row, column=1, sticky="ew", padx=8, pady=4)
        button_widget.grid(row=row, column=2, sticky="e", pady=4)
        return label_widget, entry_widget, button_widget

    def _choose_destination(self) -> None:
        path = filedialog.askopenfilename(title="Select destination video", filetypes=VIDEO_TYPES)
        if path:
            self.destination_selected_by_user = True
            self.destination_var.set(path)

    def _choose_source(self) -> None:
        path = filedialog.askopenfilename(title="Select source dialogue", filetypes=VIDEO_TYPES)
        if path:
            self.source_selected_by_user = True
            self.source_var.set(path)

    def _choose_output_dir(self) -> None:
        path = filedialog.askdirectory(title="Select output folder")
        if path:
            self.output_var.set(path)

    def _bind_truth_refresh(self) -> None:
        for variable in (self.mode_var, self.workflow_var, self.target_length_var, self.remix_preference_var, self.destination_var, self.source_var, self.output_var, self.quality_var, self.filter_var):
            variable.trace_add("write", lambda *_args: self._refresh_truth_panel())
        self._refresh_truth_panel()

    def _sync_output_form_controls(self) -> None:
        if not hasattr(self, "target_length_widgets"):
            return
        if workflow_uses_target_length(self.workflow_var.get()):
            self.full_movie_length_note.grid_remove()
            self.target_length_widgets[0].grid(row=1, column=0, sticky="w", pady=3)
            self.target_length_widgets[1].grid(row=1, column=1, sticky="w", padx=8, pady=3)
        else:
            for widget in self.target_length_widgets:
                widget.grid_remove()
            self.full_movie_length_note.grid(row=1, column=0, columnspan=3, sticky="w", pady=3)

    def _refresh_truth_panel(self) -> None:
        self._sync_output_form_controls()
        self._refresh_setting_definitions()
        self.current_truth_var.set(
            run_truth_summary(
                transformation=self.mode_var.get(),
                destination=Path(self.destination_var.get()).expanduser(),
                source=Path(self.source_var.get()).expanduser(),
                output_dir=Path(self.output_var.get()).expanduser(),
                quality=self.quality_var.get(),
                matching_style=self.filter_var.get(),
                workflow=self.workflow_var.get(),
            )
        )

    def _start_run(self) -> None:
        if self.worker and self.worker.is_alive():
            return
        try:
            config = self._selected_config()
            filter_parameters = selected_filter_parameters(self)
        except ValueError as exc:
            messagebox.showerror("Invalid settings", str(exc))
            return
        self.console_log_path = config.output_dir / "gui_console.log"
        self.console_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.console_log_path.write_text("", encoding="utf-8")
        self._clear_text_widget(self.console)
        self._clear_text_widget(self.journal)
        self.last_journal_event_ids.clear()
        for build_line in format_build_identification(self.root_dir):
            self._append_console(build_line + "\n")
        speaker_status = diarization_setup_status(backend=config.speaker_diarization_backend)
        if config.speaker_diarization_backend == "pyannote" and not speaker_status.get("available"):
            warning = f"Speaker diarization warning: {speaker_status.get('reason')}; the run will fall back to timing-based speaker labels.\n"
            self._append_console(warning)
            self._consume_operator_text(warning)
        runtime_warning = quality_runtime_warning(self.quality_var.get(), whisper_runtime())
        if runtime_warning:
            if quality_preset_mode(self.quality_var.get()) == "quality":
                proceed = messagebox.askyesno("Precision may require more time", f"{runtime_warning}\n\nContinue with Precision fidelity?")
                if not proceed:
                    self.quality_var.set("Balanced")
                    self._sync_quality_detail()
                    self._refresh_truth_panel()
                    return
            self._append_console(f"Quality warning: {runtime_warning}\n")
            self._append_journal("Fidelity notice", runtime_warning, severity="warning", event_id="fidelity_warning")
        self.cancel_requested = False
        self.completed_stage_durations.clear()
        self.furthest_stage_index = -1
        self.active_stage_key = None
        self.active_stage_started_at = None
        self.cancel_button.configure(state="normal")
        self._show_wizard_step(3)
        self.run_started_at = time.time()
        self.last_console_activity_at = self.run_started_at
        self.last_heartbeat_console_at = self.run_started_at
        self._set_overall_progress(2.0, allow_decrease=True)
        self.stage_progress_var.set(0.0)
        self.stage_var.set("Preparing the experiment")
        self.current_operation_var.set("Preparing the experiment")
        self.live_elapsed_var.set("00:00")
        self.live_idle_var.set("00:00")
        self.live_eta_var.set("Calculating...")
        self.specimen_var.set(Path(self.destination_var.get()).name or "Selected material")
        self._set_running(True, "Experiment in progress")
        self._mark_stage(None)
        start_detail = f"Starting {self.workflow_var.get()} / {self.mode_var.get()} run with {self.quality_var.get()} quality...\n"
        self._append_console(start_detail)
        self._append_journal("Experiment initiated", f"{self.mode_var.get()} has begun with {self.quality_var.get().lower()} fidelity.", event_id="experiment_started")
        definition = current_filter_definition(self)
        mutation_id = definition.implementation_key or "movie_masher"
        self.worker = threading.Thread(target=self._run_pipeline, args=(config, False, self.mode_var.get(), mutation_id, self.workflow_var.get(), remix_preference_id(self.remix_preference_var.get()), filter_parameters), daemon=True)
        self.worker.start()


    def _selected_config(self):
        destination = Path(self.destination_var.get()).expanduser()
        source = Path(self.source_var.get()).expanduser()
        output = Path(self.output_var.get()).expanduser()
        definition = current_filter_definition(self)
        if not definition.implemented:
            raise ValueError(f"{definition.name} is in development and cannot be run.")
        if not destination.exists():
            raise ValueError(f"Destination video does not exist: {destination}")
        source_required = "source_dialogue" in definition.required_inputs
        if source_required and not source.exists():
            raise ValueError(f"Source dialogue does not exist: {source}")
        if not source_required:
            if single_film_input_needs_explicit_choice(
                self.mode_var.get(),
                destination,
                self.base_config.destination_video,
                selected_by_user=self.destination_selected_by_user,
            ):
                raise ValueError(f"Choose one source film for {self.mode_var.get()} before starting.")
            source = destination
        target = target_length_seconds(self.target_length_var.get())
        return self.base_config.with_overrides(
            mode=quality_preset_mode(self.quality_var.get()),
            destination_video=destination.resolve(),
            source_dialogue=source.resolve(),
            output_dir=output.resolve(),
            cinematic_filter=self.filter_labels.get(self.filter_var.get(), "balanced"),
            target_duration_seconds=target,
            minimum_duration_seconds=min(120.0, target),
            maximum_duration_seconds=max(300.0, target),
        )

    def _sync_mode_fields(self) -> None:
        display_name = display_mode_name(self.mode_var.get())
        if display_name != self.mode_var.get():
            self.mode_var.set(display_name)
        self.mode_description_var.set(MODE_DESCRIPTIONS.get(display_name, "A cinematic transformation experiment."))
        sync_filter_mode(self)

    def _run_pipeline(self, config, force: bool, app_mode: str = TRANSPOSITION, mutation_id: str = "echo", workflow: str = "Best Short Remix", preference: str = "balanced", filter_parameters: dict[str, Any] | None = None) -> None:
        writer = QueueWriter(self.output_queue)
        internal_app_mode = internal_mode_name(app_mode)
        try:
            with contextlib.redirect_stdout(writer), contextlib.redirect_stderr(writer):
                pipeline = Pipeline(config, cancel_check=lambda: self.cancel_requested, stage_callback=lambda stage: self.output_queue.put(f"__DIARIZATION_STAGE__{stage}\n"))
                if workflow == "Best Short Remix":
                    output = pipeline.run_best_short_remix(app_mode=internal_app_mode, mutation_id=mutation_id, preference=preference, filter_parameters=filter_parameters, force=force)["video"]
                elif internal_app_mode != "Movie Masher":
                    output = pipeline.run_mutation(mutation_id, force=force, parameters=filter_parameters)["video"]
                else:
                    output = pipeline.run_all(force=force)
            self.last_output = output
            self.output_queue.put(f"__OUTPUT__{output}\n")
            summary_dir = Path(output).parent
            summary = summarize_output_dir(summary_dir)
            actual_model, model_warning = summarize_whisper_model_used(config.output_dir, config.whisper_model)
            completion = completion_summary(
                output=Path(output),
                output_dir=config.output_dir,
                transformation=app_mode,
                quality_preset=quality_preset_label(config.transcription_mode),
                whisper_model=actual_model,
                model_warning=model_warning,
                started_at=self.run_started_at,
            )
            self.output_queue.put(f"__COMPLETION__{completion}\n")
            self.output_queue.put(f"Technical completion: transcription_model={actual_model}; output={output}; warning={model_warning or 'none'}\n")
            self.output_queue.put(f"__SUMMARY__{summary['message']}\n")
            if summary.get("preview_dir"):
                self.output_queue.put(f"__PREVIEWS__{summary['preview_dir']}\n")
            self.output_queue.put("Processing finished.\n")
            self.output_queue.put("__PROGRESS__100|100|Finished.\n")
            self.output_queue.put("__STATUS__Processing finished\n")
        except Exception:
            if self.cancel_requested:
                self.output_queue.put("__STATUS__Cancelled\n")
            else:
                self.output_queue.put(traceback.format_exc())
                self.output_queue.put("__STATUS__Failed\n")


    def _drain_output_queue(self) -> None:
        try:
            while True:
                text = self.output_queue.get_nowait()
                if text.startswith("__STATUS__"):
                    status = text.removeprefix("__STATUS__").strip()
                    self.status_var.set(status)
                    self._set_running(False, status)
                    if status == "Processing finished":
                        self.current_operation_var.set("Artifact archived")
                        self._append_journal("Artifact archived", "The resulting cinematic artifact has been archived.", event_id="completed")
                        self._signal_processing_finished()
                    elif status == "Failed":
                        self.current_operation_var.set("Experiment interrupted")
                        self._signal_processing_failed()
                    elif status == "Cancelled":
                        self._complete_active_stage()
                        self._show_wizard_step(2)
                        self.status_var.set("Run cancelled")
                    elif status == "Cancelled":
                        self._complete_active_stage()
                        self._show_wizard_step(2)
                        self.status_var.set("Run cancelled")
                elif text.startswith("__DIARIZATION_STAGE__"):
                    self.diarization_active_stage = text.removeprefix("__DIARIZATION_STAGE__").strip() or None
                    if self.diarization_active_stage:
                        event = stage_message('destination_speech')
                        self.current_operation_var.set(event.title)
                        self._append_journal(event.title, event.message, event_id=event.event_id)
                        self.last_console_activity_at = time.time()
                        chunk = diarization_chunk_progress(self.diarization_active_stage)
                        if chunk:
                            completed, total = chunk
                            fraction = completed / total
                            self.stage_progress_var.set(fraction * 100.0)
                            self._set_overall_progress(
                                STAGE_PROGRESS_FLOORS["destination_speech"]
                                + fraction * (STAGE_PROGRESS_FLOORS["performances"] - STAGE_PROGRESS_FLOORS["destination_speech"] - 1.0)
                            )
                            self._update_overall_eta()
                elif text.startswith("__OUTPUT__"):
                    output = text.removeprefix("__OUTPUT__").strip()
                    self.output_path_var.set(output)
                elif text.startswith("__RUNTRUTH__"):
                    self.last_truth_var.set(text.removeprefix("__RUNTRUTH__").strip())
                elif text.startswith("__SUMMARY__"):
                    summary_text = text.removeprefix("__SUMMARY__").strip()
                    current_summary = self.problem_summary_var.get().strip()
                    if current_summary.startswith("Transformation complete"):
                        self.problem_summary_var.set(f"{current_summary} {summary_text}")
                    else:
                        self.problem_summary_var.set(summary_text)
                elif text.startswith("__PREVIEWS__"):
                    self.preview_path_var.set(text.removeprefix("__PREVIEWS__").strip())
                elif text.startswith("__COMPLETION__"):
                    completion = text.removeprefix("__COMPLETION__").strip()
                    self.problem_summary_var.set(completion)
                    self.completion_summary_var.set(completion)
                elif text.startswith("__PROGRESS__"):
                    self._apply_progress_message(text.removeprefix("__PROGRESS__").strip())
                else:
                    self.last_console_activity_at = time.time()
                    self._append_console(text)
                    self._consume_operator_text(text)
                    self._update_plain_status_from_console(text)
        except queue.Empty:
            pass
        self.after(100, self._drain_output_queue)


    def _refresh_run_heartbeat(self) -> None:
        if self.worker and self.worker.is_alive() and self.run_started_at is not None:
            now = time.time()
            elapsed = now - self.run_started_at
            idle = now - (self.last_console_activity_at or self.run_started_at)
            stage_percent = heartbeat_stage_progress(elapsed)
            self.status_var.set("Experiment in progress")
            self.live_elapsed_var.set(format_clock_duration(elapsed))
            self.live_idle_var.set(format_clock_duration(idle))
            remaining = estimate_overall_remaining(elapsed, float(self.overall_progress_var.get() or 0.0))
            self.live_eta_var.set("Calculating..." if remaining is None else format_clock_duration(remaining))
            progress = ProgressState.start("active", self.current_operation_var.get(), total=100)
            progress.started_at = self.run_started_at
            progress.update(current=int(stage_percent))
            self.stage_progress_var.set(stage_percent)
            self._update_overall_eta()
        self.after(1000, self._refresh_run_heartbeat)

    def _apply_progress_message(self, payload: str) -> None:
        parts = payload.split("|", 2)
        if len(parts) != 3:
            return
        try:
            overall = float(parts[0])
            stage = float(parts[1])
        except ValueError:
            return
        message = parts[2]
        self._set_overall_progress(overall)
        bounded_stage = max(0.0, min(100.0, stage))
        self.stage_progress_var.set(bounded_stage)
        self._update_overall_eta()
        event = operator_message_for_log(message)
        visible_message = event.title if event else "Continuing the experiment"
        progress = ProgressState.start("reported", visible_message, total=100)
        if self.run_started_at is not None:
            progress.started_at = self.run_started_at
        progress.update(current=int(bounded_stage))
        self.stage_var.set(visible_message)
        self.current_operation_var.set(visible_message)
        if message == "Finished.":
            self._mark_stage(None, finished=True)

    def _update_plain_status_from_console(self, text: str) -> None:
        stage_key = stage_key_for_log_line(text)
        status = STAGE_LABELS.get(stage_key) if stage_key else None
        if not status or status == self.last_plain_stage:
            return
        event = stage_message(stage_key or "")
        self._append_journal(event.title, event.message, event_id=event.event_id)
        display_key = stage_sequence_key(stage_key)
        if display_key is None:
            self.last_plain_stage = status
            self.stage_var.set(status)
            self.current_operation_var.set(status)
            return
        stage_index = next(index for index, (key, _label) in enumerate(STAGE_SEQUENCE) if key == display_key)
        if stage_index < self.furthest_stage_index:
            return
        now = time.time()
        if stage_index > self.furthest_stage_index:
            self._complete_active_stage(now)
            self.active_stage_key = display_key
            self.active_stage_started_at = now
            self.furthest_stage_index = stage_index
        self.last_plain_stage = status
        self._mark_stage(display_key)
        self.stage_var.set(status)
        self.current_operation_var.set(status)
        floor = STAGE_PROGRESS_FLOORS[display_key]
        self._set_overall_progress(floor)
        self.stage_progress_var.set(15.0)
        self._update_overall_eta()
        self.status_var.set("Experiment in progress")

    def _signal_processing_finished(self) -> None:
        self.bell()
        output = self.output_path_var.get().strip()
        folder = finished_output_folder(output, self.output_var.get())
        self._complete_active_stage()
        self._mark_stage(None, finished=True)
        self._append_console(f"Processing finished. Output file: {output}\n")
        self.status_var.set("Processing finished")
        if not self.problem_summary_var.get().strip():
            self.problem_summary_var.set("Processing finished successfully.")
        self._show_wizard_step(4)

    def _open_finished_output_folder(self, folder: Path) -> None:
        try:
            folder.mkdir(parents=True, exist_ok=True)
            os.startfile(folder)
        except OSError as exc:
            self._append_console(f"Could not open output folder: {exc}\n")

    def _signal_processing_failed(self) -> None:
        self.bell()
        self._append_journal("Experiment interrupted", "The experiment did not complete. Review the Technical Record for details.", severity="error", event_id="experiment_failed")
        messagebox.showerror("Experiment interrupted", "The experiment did not complete. Review the Technical Record for details.")

    def _clear_text_widget(self, widget: tk.Text) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.configure(state="disabled")

    def _append_journal(self, title: str, message: str, *, severity: str = "info", event_id: str | None = None, force: bool = False) -> None:
        if event_id and event_id in self.last_journal_event_ids and not force:
            return
        if event_id:
            self.last_journal_event_ids.add(event_id)
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.journal.configure(state="normal")
        self.journal.tag_configure("time", foreground="#929aa3")
        self.journal.tag_configure("title", foreground="#b99b5e", font=("Segoe UI", 10, "bold"))
        self.journal.tag_configure("warning", foreground="#e0bd72")
        self.journal.tag_configure("error", foreground="#e58d86")
        self.journal.insert("end", f"{timestamp}  ", "time")
        self.journal.insert("end", f"{title}\n", severity if severity in {"warning", "error"} else "title")
        self.journal.insert("end", f"{message}\n\n")
        self.journal.see("end")
        self.journal.configure(state="disabled")

    def _consume_operator_text(self, text: str) -> None:
        if contains_traceback(text):
            return
        for line in text.splitlines():
            event = operator_message_for_log(line)
            if event is None:
                continue
            self.current_operation_var.set(event.title)
            if event.stage_key:
                self.stage_var.set(event.title)
            if event.journal:
                self._append_journal(event.title, event.message, severity=event.severity, event_id=event.event_id)

    def _append_console(self, text: str) -> None:
        self.console.configure(state="normal")
        self.console.insert("end", text)
        self.console.see("end")
        self.console.configure(state="disabled")
        if self.console_log_path is not None:
            try:
                with self.console_log_path.open("a", encoding="utf-8") as handle:
                    handle.write(text)
            except OSError:
                pass

    def _set_running(self, running: bool, status: str) -> None:
        self.status_var.set(status)
        state = "disabled" if running else "normal"
        self.start_button.configure(state=state)
        self.performance_review_button.configure(state=state)
        self.review_button.configure(state=state)
        self.highlight_review_button.configure(state=state)
        self.open_button.configure(state=state)
        self.cancel_button.configure(state="normal" if running else "disabled")


    def _open_highlight_review(self) -> None:
        if self.worker and self.worker.is_alive():
            return
        try:
            config = self._selected_config()
        except ValueError as exc:
            messagebox.showerror("Invalid settings", str(exc))
            return
        self.console_log_path = config.output_dir / "gui_console.log"
        self.console_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.console_log_path.write_text("", encoding="utf-8")
        for build_line in format_build_identification(self.root_dir):
            self._append_console(build_line + "\n")
        speaker_status = diarization_setup_status(backend=config.speaker_diarization_backend)
        if config.speaker_diarization_backend == "pyannote" and not speaker_status.get("available"):
            self._append_console(f"Speaker diarization warning: {speaker_status.get('reason')}; the run will fall back to timing-based speaker labels.\n")
        highlight_path = config.output_dir / "editorial_highlights.json"
        if not highlight_path.exists():
            messagebox.showerror("Highlights unavailable", "No editorial highlights are available yet. Run a transformation first.")
            return
        try:
            pipeline = Pipeline(config)
            schedule_path = pipeline.destination.cache_dir / "replacement_schedule.json"
            schedule = validate_artifact("replacement_schedule", schedule_path, pipeline.schemas_dir)
            highlights = validate_artifact("editorial_highlights", highlight_path, pipeline.schemas_dir)
        except Exception as exc:
            messagebox.showerror("Highlights unavailable", str(exc))
            return
        HighlightReviewWindow(self, config, schedule_path, schedule, highlights)


    def _open_performance_review(self) -> None:
        if self.worker and self.worker.is_alive():
            return
        try:
            config = self._selected_config()
        except ValueError as exc:
            messagebox.showerror("Invalid settings", str(exc))
            return
        self.console_log_path = config.output_dir / "gui_console.log"
        self.console_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.console_log_path.write_text("", encoding="utf-8")
        for build_line in format_build_identification(self.root_dir):
            self._append_console(build_line + "\n")
        speaker_status = diarization_setup_status(backend=config.speaker_diarization_backend)
        if config.speaker_diarization_backend == "pyannote" and not speaker_status.get("available"):
            self._append_console(f"Speaker diarization warning: {speaker_status.get('reason')}; the run will fall back to timing-based speaker labels.\n")
        try:
            pipeline = Pipeline(config)
            pipeline.schedule(force=False)
            schedule_path = pipeline.destination.cache_dir / "replacement_schedule.json"
            schedule = validate_artifact("replacement_schedule", schedule_path, pipeline.schemas_dir)
        except Exception as exc:
            messagebox.showerror("Review unavailable", str(exc))
            return
        PerformanceReviewWindow(self, config, schedule_path, schedule)

    def _open_review(self) -> None:
        if self.worker and self.worker.is_alive():
            return
        try:
            config = self._selected_config()
        except ValueError as exc:
            messagebox.showerror("Invalid settings", str(exc))
            return
        self.console_log_path = config.output_dir / "gui_console.log"
        self.console_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.console_log_path.write_text("", encoding="utf-8")
        for build_line in format_build_identification(self.root_dir):
            self._append_console(build_line + "\n")
        speaker_status = diarization_setup_status(backend=config.speaker_diarization_backend)
        if config.speaker_diarization_backend == "pyannote" and not speaker_status.get("available"):
            self._append_console(f"Speaker diarization warning: {speaker_status.get('reason')}; the run will fall back to timing-based speaker labels.\n")
        try:
            pipeline = Pipeline(config)
            pipeline.schedule(force=False)
            schedule_path = pipeline.destination.cache_dir / "replacement_schedule.json"
            schedule = validate_artifact("replacement_schedule", schedule_path, pipeline.schemas_dir)
        except Exception as exc:
            messagebox.showerror("Review unavailable", str(exc))
            return
        ScheduleReviewWindow(self, config, schedule_path, schedule)

    def _preview_from_review(self, config, indices: list[int], video: bool) -> None:
        if self.worker and self.worker.is_alive():
            return
        if not indices:
            messagebox.showerror("Preview unavailable", "Select at least one mapping to preview.")
            return
        label = "Rendering preview video" if video else "Rendering preview audio"
        self._set_running(True, label)
        self.worker = threading.Thread(target=self._run_review_preview, args=(config, indices, video), daemon=True)
        self.worker.start()

    def _run_review_preview(self, config, indices: list[int], video: bool) -> None:
        writer = QueueWriter(self.output_queue)
        try:
            with contextlib.redirect_stdout(writer), contextlib.redirect_stderr(writer):
                output = Pipeline(config).render_preview(indices, video=video)
            target = output.get("video") if video else output.get("audio")
            if target:
                self.output_queue.put(f"__OUTPUT__{target}\n")
            self.output_queue.put(f"Preview written: {target}\n")
            self.output_queue.put("__STATUS__Preview complete\n")
        except Exception:
            if self.cancel_requested:
                self.output_queue.put("__STATUS__Cancelled\n")
            else:
                self.output_queue.put(traceback.format_exc())
                self.output_queue.put("__STATUS__Failed\n")

    def _render_from_review(self, config, video: bool) -> None:
        if self.worker and self.worker.is_alive():
            return
        label = "Rendering video" if video else "Rendering audio"
        self._set_running(True, label)
        self.worker = threading.Thread(target=self._run_review_render, args=(config, video), daemon=True)
        self.worker.start()

    def _run_review_render(self, config, video: bool) -> None:
        writer = QueueWriter(self.output_queue)
        try:
            with contextlib.redirect_stdout(writer), contextlib.redirect_stderr(writer):
                pipeline = Pipeline(config, cancel_check=lambda: self.cancel_requested, stage_callback=lambda stage: self.output_queue.put(f"__DIARIZATION_STAGE__{stage}\n"))
                output = pipeline.render_video(force=True) if video else pipeline.render_audio(force=True)
                pipeline.generate_reports()
            if video:
                self.output_queue.put(f"__OUTPUT__{output}\n")
            self.output_queue.put("Processing finished.\n")
            self.output_queue.put("__STATUS__Processing finished\n")
        except Exception:
            if self.cancel_requested:
                self.output_queue.put("__STATUS__Cancelled\n")
            else:
                self.output_queue.put(traceback.format_exc())
                self.output_queue.put("__STATUS__Failed\n")

    def _open_finished_movie(self) -> None:
        output = self.output_path_var.get().strip()
        if not output:
            messagebox.showerror("Output unavailable", "No finished movie has been produced yet.")
            return
        path = Path(output).expanduser()
        if not path.exists():
            messagebox.showerror("Output unavailable", f"Finished movie not found: {path}")
            return
        os.startfile(path)

    def _open_problem_previews(self) -> None:
        path_text = self.preview_path_var.get().strip()
        path = Path(path_text).expanduser() if path_text else Path(self.output_var.get()).expanduser() / "previews" / "problem_regions"
        if not path.exists():
            messagebox.showerror("Previews unavailable", "No problem preview folder exists yet.")
            return
        os.startfile(path)

    def _open_output_folder(self) -> None:
        output = self.output_path_var.get().strip()
        path = finished_output_folder(output, self.output_var.get())
        path.mkdir(parents=True, exist_ok=True)
        os.startfile(path)


class HighlightReviewWindow(tk.Toplevel):
    def __init__(self, app: MovieMasherApp, config, schedule_path: Path, schedule: dict, highlights: dict) -> None:
        super().__init__(app)
        self.app = app
        self.config = config
        self.schedule_path = schedule_path
        self.schedule = schedule
        self.highlights = highlights
        self.bucket_var = tk.StringVar(value=HIGHLIGHT_BUCKET_LABELS["most_convincing"])
        self.rows_by_key: dict[str, dict] = {}
        self.title("Review Highlights")
        self.geometry("1160x560")
        self.minsize(900, 430)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        self._build_ui()
        self._load_rows()

    def _build_ui(self) -> None:
        columns = ("bucket", "performance", "start", "duration", "score", "label", "component", "status")
        self.tree = ttk.Treeview(self, columns=columns, show="headings", selectmode="extended")
        headings = {
            "bucket": "Bucket",
            "performance": "Performance",
            "start": "Start",
            "duration": "Duration",
            "score": "Score",
            "label": "Editorial Label",
            "component": "Why Listed",
            "status": "Review Status",
        }
        widths = {
            "bucket": 140,
            "performance": 120,
            "start": 76,
            "duration": 80,
            "score": 70,
            "label": 170,
            "component": 140,
            "status": 150,
        }
        for column in columns:
            self.tree.heading(column, text=headings[column])
            self.tree.column(column, width=widths[column], stretch=column == "label")
        self.tree.tag_configure("needs", background="#ffe7e7")
        self.tree.tag_configure("funny", background="#eef6ff")
        self.tree.tag_configure("convincing", background="#eaf7ed")
        self.tree.tag_configure("awkward", background="#fff3d6")
        self.tree.grid(row=0, column=0, sticky="nsew", padx=12, pady=(12, 8))
        scrollbar = ttk.Scrollbar(self, command=self.tree.yview)
        scrollbar.grid(row=0, column=1, sticky="ns", pady=(12, 8))
        self.tree.configure(yscrollcommand=scrollbar.set)

        controls = ttk.Frame(self, padding=(12, 0, 12, 12))
        controls.grid(row=1, column=0, columnspan=2, sticky="ew")
        controls.columnconfigure(8, weight=1)
        ttk.Label(controls, text="Show").grid(row=0, column=0, sticky="w")
        bucket_box = ttk.Combobox(
            controls,
            textvariable=self.bucket_var,
            values=["All Highlights", *HIGHLIGHT_BUCKET_LABELS.values()],
            state="readonly",
            width=22,
        )
        bucket_box.grid(row=0, column=1, sticky="w", padx=(6, 12))
        bucket_box.bind("<<ComboboxSelected>>", lambda _event: self._load_rows())
        ttk.Button(controls, text="Review Selected", command=self._open_performance_review).grid(row=0, column=2, padx=(8, 0))
        ttk.Button(controls, text="Preview Selected", command=self._preview_selected).grid(row=0, column=3, padx=(8, 0))
        ttk.Button(controls, text="Open Output", command=self.app._open_output_folder).grid(row=0, column=4, padx=(8, 0))
        ttk.Button(controls, text="Close", command=self.destroy).grid(row=0, column=5, padx=(8, 0))
        self.summary_var = tk.StringVar()
        ttk.Label(controls, textvariable=self.summary_var).grid(row=0, column=8, sticky="e")

    def _load_rows(self) -> None:
        self.tree.delete(*self.tree.get_children())
        selected_label = self.bucket_var.get()
        selected_bucket = HIGHLIGHT_BUCKET_BY_LABEL.get(selected_label)
        rows = highlight_rows(self.highlights, selected_bucket)
        self.rows_by_key = {}
        for index, row in enumerate(rows):
            key = f"{row.get('bucket')}:{row.get('performance_id')}:{index}"
            self.rows_by_key[key] = row
            self.tree.insert("", "end", iid=key, values=highlight_row_values(row), tags=highlight_row_tags(row))
        summary = self.highlights.get("summary", {})
        self.summary_var.set(
            f"{len(rows)} shown / {summary.get('evaluated_performances', 0)} evaluated / {summary.get('needs_review_count', 0)} need review"
        )

    def _selected_rows(self) -> list[dict]:
        return [self.rows_by_key[item] for item in self.tree.selection() if item in self.rows_by_key]

    def _selected_performance_ids(self) -> list[str]:
        performance_ids: list[str] = []
        for row in self._selected_rows():
            performance_id = str(row.get("performance_id") or "")
            if performance_id and performance_id not in performance_ids:
                performance_ids.append(performance_id)
        return performance_ids

    def _selected_mapping_indices(self) -> list[int]:
        indices: list[int] = []
        for row in self._selected_rows():
            for value in row.get("mapping_indices", []):
                try:
                    index = int(value)
                except (TypeError, ValueError):
                    continue
                if index not in indices:
                    indices.append(index)
        return indices

    def _open_performance_review(self) -> None:
        selected = self._selected_performance_ids()
        if not selected:
            messagebox.showerror("Review unavailable", "Select at least one highlight first.")
            return
        PerformanceReviewWindow(self.app, self.config, self.schedule_path, self.schedule, initial_performance_ids=selected)

    def _preview_selected(self) -> None:
        self.app._preview_from_review(self.config, self._selected_mapping_indices(), video=True)


class PerformanceReviewWindow(tk.Toplevel):
    def __init__(self, app: MovieMasherApp, config, schedule_path: Path, schedule: dict, initial_performance_ids: list[str] | None = None) -> None:
        super().__init__(app)
        self.app = app
        self.config = config
        self.schedule_path = schedule_path
        self.schedule = schedule
        self.initial_performance_ids = initial_performance_ids or []
        self.title("Review Performances")
        self.geometry("1240x620")
        self.minsize(980, 460)
        self.filter_var = tk.StringVar(value=PERFORMANCE_REVIEW_FILTERS[0])
        self.review_label_var = tk.StringVar(value=REVIEW_LABELS[0])
        self.rows_by_id: dict[str, dict] = {}
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        self._build_ui()
        self._load_rows()
        self.select_performances(self.initial_performance_ids)

    def _build_ui(self) -> None:
        columns = (
            "performance",
            "type",
            "start",
            "duration",
            "coverage",
            "mappings",
            "reuse",
            "score",
            "reviewed",
            "speaker_match",
            "source_speakers",
            "destination_speakers",
            "speaker_fallback",
            "labels",
            "reason",
            "transcript",
        )
        self.tree = ttk.Treeview(self, columns=columns, show="headings", selectmode="extended")
        headings = {
            "performance": "Performance",
            "type": "Type",
            "start": "Start",
            "duration": "Duration",
            "coverage": "Coverage",
            "mappings": "Mappings",
            "reuse": "Reuse",
            "score": "Avg Score",
            "reviewed": "Reviewed",
            "speaker_match": "Speaker Match",
            "source_speakers": "Source Speakers",
            "destination_speakers": "Dest Speakers",
            "speaker_fallback": "Speaker Fallback",
            "labels": "Labels",
            "reason": "Reason",
            "transcript": "Transcript Preview",
        }
        widths = {
            "performance": 110,
            "type": 100,
            "start": 76,
            "duration": 76,
            "coverage": 76,
            "mappings": 74,
            "reuse": 58,
            "score": 74,
            "reviewed": 72,
            "speaker_match": 92,
            "source_speakers": 130,
            "destination_speakers": 130,
            "speaker_fallback": 150,
            "labels": 150,
            "reason": 160,
            "transcript": 340,
        }
        for column in columns:
            self.tree.heading(column, text=headings[column])
            self.tree.column(column, width=widths[column], stretch=column == "transcript")
        self.tree.tag_configure("risk", background="#fff1d6")
        self.tree.grid(row=0, column=0, sticky="nsew", padx=12, pady=(12, 8))
        scrollbar = ttk.Scrollbar(self, command=self.tree.yview)
        scrollbar.grid(row=0, column=1, sticky="ns", pady=(12, 8))
        self.tree.configure(yscrollcommand=scrollbar.set)

        controls = ttk.Frame(self, padding=(12, 0, 12, 12))
        controls.grid(row=1, column=0, columnspan=2, sticky="ew")
        controls.columnconfigure(11, weight=1)
        ttk.Label(controls, text="Filter").grid(row=0, column=0, sticky="w")
        filter_box = ttk.Combobox(controls, textvariable=self.filter_var, values=PERFORMANCE_REVIEW_FILTERS, state="readonly", width=22)
        filter_box.grid(row=0, column=1, sticky="w", padx=(6, 12))
        filter_box.bind("<<ComboboxSelected>>", lambda _event: self._load_rows())
        ttk.Combobox(controls, textvariable=self.review_label_var, values=REVIEW_LABELS, state="readonly", width=18).grid(row=0, column=2, sticky="w")
        ttk.Button(controls, text="Mark", command=self._mark_selected).grid(row=0, column=3, padx=(8, 0))
        ttk.Button(controls, text="Preview Performance", command=self._preview_performance).grid(row=0, column=4, padx=(8, 0))
        ttk.Button(controls, text="Open Mapping Review", command=self._open_mapping_review).grid(row=0, column=5, padx=(8, 0))
        ttk.Button(controls, text="Save", command=self._save).grid(row=0, column=6, padx=(8, 0))
        ttk.Button(controls, text="Render Final", command=self._render_video).grid(row=0, column=7, padx=(8, 0))
        ttk.Button(controls, text="Close", command=self.destroy).grid(row=0, column=8, padx=(8, 0))
        self.summary_var = tk.StringVar()
        ttk.Label(controls, textvariable=self.summary_var).grid(row=0, column=11, sticky="e")

    def _load_rows(self) -> None:
        self.tree.delete(*self.tree.get_children())
        rows = filtered_performance_rows(self.schedule, self.filter_var.get())
        self.rows_by_id = {str(row["performance_id"]): row for row in rows}
        for row in rows:
            tags = ("risk",) if row.get("risky") else ()
            self.tree.insert("", "end", iid=str(row["performance_id"]), values=performance_review_row_values(row), tags=tags)
        self.summary_var.set(performance_review_summary(rows, len(self.tree.get_children())))

    def select_performances(self, performance_ids: list[str]) -> None:
        existing = [performance_id for performance_id in performance_ids if self.tree.exists(str(performance_id))]
        if existing:
            self.tree.selection_set(existing)
            self.tree.focus(existing[0])
            self.tree.see(existing[0])

    def _selected_performance_ids(self) -> list[str]:
        return [str(item) for item in self.tree.selection()]

    def _selected_mapping_indices(self) -> list[int]:
        return performance_mapping_indices(self.schedule, self._selected_performance_ids())

    def _mark_selected(self) -> None:
        selected = self._selected_performance_ids()
        if not selected:
            return
        apply_performance_review_label(self.schedule, selected, self.review_label_var.get())
        self._load_rows()

    def _save(self, *, show_message: bool = True) -> None:
        write_json(self.schedule_path, self.schedule)
        write_review_notes(self.schedule, self.schedule_path.with_name("review_notes.json"), schedule_path=self.schedule_path)
        if show_message:
            messagebox.showinfo("Schedule saved", f"Saved {self.schedule_path}")

    def _preview_performance(self) -> None:
        self._save(show_message=False)
        self.app._preview_from_review(self.config, self._selected_mapping_indices(), video=True)

    def _open_mapping_review(self) -> None:
        self._save(show_message=False)
        ScheduleReviewWindow(self.app, self.config, self.schedule_path, self.schedule)

    def _render_video(self) -> None:
        self._save()
        self.app._render_from_review(self.config, video=True)


class ScheduleReviewWindow(tk.Toplevel):
    def __init__(self, app: MovieMasherApp, config, schedule_path: Path, schedule: dict) -> None:
        super().__init__(app)
        self.app = app
        self.config = config
        self.schedule_path = schedule_path
        self.schedule = schedule
        self.title("Review Schedule")
        self.geometry("1240x620")
        self.minsize(980, 460)
        self.filter_var = tk.StringVar(value=REVIEW_FILTERS[0])
        self.review_label_var = tk.StringVar(value=REVIEW_LABELS[0])
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        self._build_ui()
        self._load_rows()

    def _build_ui(self) -> None:
        columns = (
            "enabled",
            "window",
            "clip",
            "start",
            "duration",
            "score",
            "shot",
            "visual",
            "cross",
            "overrun",
            "strategy",
            "source_speaker",
            "destination_speaker",
            "speaker_match",
            "speaker_fallback",
            "review",
            "transcript",
        )
        self.tree = ttk.Treeview(self, columns=columns, show="headings", selectmode="extended")
        headings = {
            "enabled": "On",
            "window": "Window",
            "clip": "Clip",
            "start": "Start",
            "duration": "Duration",
            "score": "Score",
            "shot": "Shot",
            "visual": "Visual Fit",
            "cross": "Cross",
            "overrun": "Overrun",
            "strategy": "Strategy",
            "source_speaker": "Source Speaker",
            "destination_speaker": "Dest Speaker",
            "speaker_match": "Speaker",
            "speaker_fallback": "Speaker Fallback",
            "review": "Review",
            "transcript": "Transcript",
        }
        widths = {
            "enabled": 48,
            "window": 90,
            "clip": 90,
            "start": 76,
            "duration": 76,
            "score": 64,
            "shot": 110,
            "visual": 78,
            "cross": 58,
            "overrun": 74,
            "strategy": 150,
            "source_speaker": 118,
            "destination_speaker": 112,
            "speaker_match": 78,
            "speaker_fallback": 150,
            "review": 130,
            "transcript": 310,
        }
        for column in columns:
            self.tree.heading(column, text=headings[column])
            self.tree.column(column, width=widths[column], stretch=column == "transcript")
        self.tree.tag_configure("risk", background="#fff1d6")
        self.tree.tag_configure("disabled", foreground="#777777")
        self.tree.grid(row=0, column=0, sticky="nsew", padx=12, pady=(12, 8))
        scrollbar = ttk.Scrollbar(self, command=self.tree.yview)
        scrollbar.grid(row=0, column=1, sticky="ns", pady=(12, 8))
        self.tree.configure(yscrollcommand=scrollbar.set)

        controls = ttk.Frame(self, padding=(12, 0, 12, 12))
        controls.grid(row=1, column=0, columnspan=2, sticky="ew")
        controls.columnconfigure(11, weight=1)
        ttk.Label(controls, text="Filter").grid(row=0, column=0, sticky="w")
        filter_box = ttk.Combobox(controls, textvariable=self.filter_var, values=REVIEW_FILTERS, state="readonly", width=22)
        filter_box.grid(row=0, column=1, sticky="w", padx=(6, 12))
        filter_box.bind("<<ComboboxSelected>>", lambda _event: self._load_rows())
        ttk.Combobox(controls, textvariable=self.review_label_var, values=REVIEW_LABELS, state="readonly", width=18).grid(row=0, column=2, sticky="w")
        ttk.Button(controls, text="Mark", command=self._mark_selected).grid(row=0, column=3, padx=(8, 0))
        ttk.Button(controls, text="Enable", command=lambda: self._set_selected_enabled(True)).grid(row=0, column=4, padx=(8, 0))
        ttk.Button(controls, text="Disable", command=lambda: self._set_selected_enabled(False)).grid(row=0, column=5, padx=(8, 0))
        ttk.Button(controls, text="Preview Region", command=self._preview_region_video).grid(row=0, column=6, padx=(8, 0))
        ttk.Button(controls, text="Save", command=self._save).grid(row=0, column=7, padx=(8, 0))
        ttk.Button(controls, text="Render Final", command=self._render_video).grid(row=0, column=8, padx=(8, 0))
        ttk.Button(controls, text="Close", command=self.destroy).grid(row=0, column=9, padx=(8, 0))
        self.summary_var = tk.StringVar()
        ttk.Label(controls, textvariable=self.summary_var).grid(row=0, column=11, sticky="e")

    def _load_rows(self) -> None:
        self.tree.delete(*self.tree.get_children())
        mappings = self.schedule.get("mappings", [])
        for index in filtered_mapping_indices(mappings, self.filter_var.get()):
            mapping = mappings[index]
            tags = []
            if not mapping.get("enabled", True):
                tags.append("disabled")
            if mapping.get("mapping_crosses_shot_boundary") or float(mapping.get("visual_fit_score") or 1.0) < 0.75 or mapping.get("review_label") not in {None, "unreviewed", "good"}:
                tags.append("risk")
            self.tree.insert("", "end", iid=str(index), values=self._row_values(mapping), tags=tuple(tags))
        self._update_summary()

    def _row_values(self, mapping: dict) -> tuple:
        return review_row_values(mapping)

    def _selected_indices(self) -> list[int]:
        return [int(item) for item in self.tree.selection()]


    def _mark_selected(self) -> None:
        selected = self._selected_indices()
        if not selected:
            return
        apply_review_label(self.schedule.get("mappings", []), selected, self.review_label_var.get())
        self._load_rows()

    def _set_selected_enabled(self, enabled: bool) -> None:
        for index in self._selected_indices():
            self.schedule["mappings"][index]["enabled"] = enabled
            if self.tree.exists(str(index)):
                self.tree.item(str(index), values=self._row_values(self.schedule["mappings"][index]))
        self._load_rows()

    def _preview_clip(self) -> None:
        selected = self._selected_indices()
        if not selected:
            return
        clip_path = Path(self.schedule["mappings"][selected[0]].get("clip_path", ""))
        if not clip_path.exists():
            messagebox.showerror("Preview unavailable", f"Clip file not found: {clip_path}")
            return
        os.startfile(clip_path)

    def _save(self, *, show_message: bool = True) -> None:
        write_json(self.schedule_path, self.schedule)
        write_review_notes(self.schedule, self.schedule_path.with_name("review_notes.json"), schedule_path=self.schedule_path)
        if show_message:
            messagebox.showinfo("Schedule saved", f"Saved {self.schedule_path}")

    def _preview_region_video(self) -> None:
        self._save(show_message=False)
        self.app._preview_from_review(self.config, self._selected_indices(), video=True)

    def _render_video(self) -> None:
        self._save()
        self.app._render_from_review(self.config, video=True)

    def _update_summary(self) -> None:
        mappings = self.schedule.get("mappings", [])
        self.summary_var.set(review_summary(mappings, len(self.tree.get_children())))

def highlight_rows(highlights: dict, bucket: str | None = None) -> list[dict]:
    bucket_names = [bucket] if bucket else list(HIGHLIGHT_BUCKETS)
    rows: list[dict] = []
    seen: set[tuple[str, str]] = set()
    highlight_groups = highlights.get("highlights", {}) if isinstance(highlights, dict) else {}
    for bucket_name in bucket_names:
        for item in highlight_groups.get(bucket_name, []) or []:
            performance_id = str(item.get("performance_id") or "")
            key = (bucket_name or "", performance_id)
            if key in seen:
                continue
            seen.add(key)
            row = dict(item)
            row["bucket"] = bucket_name
            row["bucket_label"] = HIGHLIGHT_BUCKET_LABELS.get(bucket_name or "", str(bucket_name or ""))
            rows.append(row)
    return rows


def highlight_row_values(row: dict) -> tuple:
    return (
        row.get("bucket_label", ""),
        row.get("performance_id", ""),
        _round_optional(row.get("start"), 3),
        _round_optional(row.get("duration"), 3),
        _round_optional(row.get("editorial_score"), 3),
        row.get("editorial_label", ""),
        row.get("component", ""),
        row.get("review_status", ""),
    )


def highlight_row_tags(row: dict) -> tuple[str, ...]:
    bucket = row.get("bucket")
    if bucket == "needs_attention":
        return ("needs",)
    if bucket == "funniest":
        return ("funny",)
    if bucket == "most_convincing":
        return ("convincing",)
    if bucket == "most_awkward":
        return ("awkward",)
    return ()


def _round_optional(value, digits: int):
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return ""


def format_duration(seconds: float | None) -> str:
    total = max(0, int(seconds or 0))
    minutes, sec = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {sec}s"
    if minutes:
        return f"{minutes}m {sec}s"
    return f"{sec}s"


def format_clock_duration(seconds: float | None) -> str:
    total = max(0, int(seconds or 0))
    hours, remainder = divmod(total, 3600)
    minutes, sec = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{sec:02d}" if hours else f"{minutes:02d}:{sec:02d}"


def heartbeat_stage_progress(elapsed_seconds: float) -> float:
    return 10.0 + (max(0.0, elapsed_seconds) % 80.0)


def heartbeat_stage_message(*, stage: str, idle_seconds: float, idle_threshold: float = 12.0) -> str:
    if idle_seconds >= idle_threshold:
        return f"{stage} No new observations for {format_clock_duration(idle_seconds)}."
    return stage


def should_emit_console_heartbeat(*, idle_seconds: float, now: float, last_heartbeat_at: float | None, idle_threshold: float = 30.0, interval: float = 30.0) -> bool:
    return False


def completion_summary(
    *,
    output: Path,
    output_dir: Path,
    transformation: str,
    quality_preset: str,
    whisper_model: str,
    started_at: float | None,
    model_warning: str | None = None,
) -> str:
    elapsed = time.time() - started_at if started_at else None
    mode = display_mode_name(transformation)
    duration = reported_output_duration(output, output_dir)
    duration_text = format_clock_duration(duration) if duration is not None else "Recorded in the Technical Record"
    message = (
        f"Mode: {mode}\n"
        f"Fidelity: {quality_preset}\n"
        f"Artifact: {output.name}\n"
        f"Final duration: {duration_text}\n"
        f"Observation time: {format_duration(elapsed)}"
    )
    if model_warning:
        message = f"{message}\n\nVocal analysis: An alternate method was used. Review the Technical Record for details."
    return message


def reported_output_duration(output: Path, output_dir: Path) -> float | None:
    candidates = (
        output.parent / "output_report.json",
        output.parent / "mutation_report.json",
        output_dir / "run_report.json",
    )
    for path in candidates:
        if not path.exists():
            continue
        with contextlib.suppress(OSError, ValueError, TypeError):
            data = read_json(path)
            values = (
                data.get("actual_duration"),
                data.get("actual_scene_duration"),
                (data.get("outputs") or {}).get("output_duration"),
                (data.get("outputs") or {}).get("duration"),
            )
            value = next((float(item) for item in values if item is not None), None)
            if value is not None:
                return value
    return None


def summarize_whisper_model_used(output_dir: Path, configured_model: str) -> tuple[str, str | None]:
    report_path = output_dir / "run_report.json"
    if not report_path.exists():
        return configured_model, None
    try:
        report = read_json(report_path)
    except (OSError, ValueError):
        return configured_model, None
    config = report.get("config", {}) if isinstance(report, dict) else {}
    model = str(config.get("whisper_model") or configured_model)
    warning = None
    for key in ("source_events", "destination_timeline"):
        artifact = report.get(key, {}) if isinstance(report, dict) else {}
        if not isinstance(artifact, dict):
            continue
        artifact_warning = artifact.get("whisper_model_warning")
        artifact_model = artifact.get("whisper_model")
        if artifact_model:
            model = str(artifact_model)
        if artifact_warning:
            warning = str(artifact_warning)
            break
    return model, warning


def summarize_output_dir(output_dir: Path) -> dict[str, str | int | None]:
    problem_path = output_dir / "problem_regions.json"
    editorial_path = output_dir / "editorial_highlights.json"
    preview_manifest_path = output_dir / "previews" / "problem_regions" / "problem_region_previews.json"
    problem_count = 0
    fallback_count = 0
    undercovered_count = 0
    highlight_count = 0
    needs_attention_count = 0
    convincing_count = 0
    funny_count = 0
    awkward_count = 0
    preview_count = 0
    preview_dir: str | None = None
    if editorial_path.exists():
        with contextlib.suppress(Exception):
            editorial_report = read_json(editorial_path)
            summary = editorial_report.get("summary", {})
            needs_attention_count = int(summary.get("needs_review_count", 0) or 0)
            highlights = editorial_report.get("highlights", {})
            convincing_count = len(highlights.get("most_convincing", []) or [])
            funny_count = len(highlights.get("funniest", []) or [])
            awkward_count = len(highlights.get("most_awkward", []) or [])
            highlight_count = convincing_count + funny_count + awkward_count
    if problem_path.exists():
        with contextlib.suppress(Exception):
            problem_report = read_json(problem_path)
            problem_count = int(problem_report.get("problem_count", 0) or 0)
            summary = problem_report.get("summary", {})
            fallback_count = int(summary.get("fallback_mapping_count", 0) or 0)
            undercovered_count = int(summary.get("undercovered_speech_window_count", 0) or 0)
    if preview_manifest_path.exists():
        with contextlib.suppress(Exception):
            preview_manifest = read_json(preview_manifest_path)
            preview_count = int(preview_manifest.get("preview_count", 0) or 0)
            preview_dir = str(preview_manifest_path.parent)
    if highlight_count:
        message = f"Highlights ready: {convincing_count} convincing, {funny_count} funny, {awkward_count} awkward."
        if needs_attention_count:
            message = f"{message} {needs_attention_count} performance(s) need review."
    elif problem_count:
        message = f"{problem_count} review region(s): {undercovered_count} undercovered speech window(s), {fallback_count} fallback mapping(s)."
        if preview_count:
            message = f"{message} {preview_count} preview clip(s) ready."
    else:
        message = "No problem regions reported."
    return {
        "message": message,
        "problem_count": problem_count,
        "fallback_count": fallback_count,
        "undercovered_count": undercovered_count,
        "highlight_count": highlight_count,
        "needs_attention_count": needs_attention_count,
        "preview_count": preview_count,
        "preview_dir": preview_dir,
    }


def main() -> int:
    app = MovieMasherApp()
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


