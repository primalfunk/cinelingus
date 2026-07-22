from __future__ import annotations

from pathlib import Path
from tkinter import filedialog, messagebox
from typing import Any

from .gui_implementation import *  # noqa: F401,F403 - compatibility surface
from .gui_implementation import CinelingusInstrumentApp as _InstrumentImplementation
from .filter_lab.gui_controller import current_filter_definition
from .reliable_inputs import chooser_initial_directory, default_input_directory, preflight_media_inputs
from .util import read_json


OUTPUT_FORM = "Full Source Timeline"


class CinelingusInstrumentApp(_InstrumentImplementation):
    """Retired Tk shell retained only for explicit compatibility diagnostics."""

    def __init__(self) -> None:
        super().__init__()
        self.default_input_dir = default_input_directory()
        self.last_input_dir: Path | None = None
        self.destination_var.set("")
        self.source_var.set("")
        self.destination_selected_by_user = False
        self.source_selected_by_user = False
        self.input_guidance_var.set(
            f"Choose complete film files. Selection begins in {self.default_input_dir}."
        )

    def _choose_film(self, index: int) -> None:
        title = "Select anchor film" if index == 0 else f"Select Film {film_label(index)}"
        current = self.film_vars[index].get() if index < len(self.film_vars) else ""
        initial_dir = chooser_initial_directory(
            current,
            last_directory=self.last_input_dir,
            default_directory=self.default_input_dir,
        )
        path = filedialog.askopenfilename(title=title, filetypes=VIDEO_TYPES, initialdir=str(initial_dir))
        if not path:
            return
        while len(self.film_vars) <= index:
            self.film_vars.append(tk.StringVar(value=""))
        self.film_vars[index].set(path)
        self.last_input_dir = Path(path).parent
        if index == 0:
            self.destination_selected_by_user = True
        elif index == 1:
            self.source_selected_by_user = True

    def _choose_destination(self) -> None:
        self._choose_film(0)

    def _choose_source(self) -> None:
        self._choose_film(1)

    def _selected_film_paths(self) -> list[Path]:
        return [
            Path(value).expanduser()
            for variable in self.film_vars[: self._active_film_count]
            if (value := variable.get().strip())
        ]

    def _refresh_truth_panel(self) -> None:
        if any(not variable.get().strip() for variable in self.film_vars[: self._active_film_count]):
            self._refresh_setting_definitions()
            self.current_truth_var.set("Awaiting complete film selections. No material has been assumed.")
            return
        super()._refresh_truth_panel()

    def _selected_config(self):
        missing = [
            "Anchor Film" if index == 0 else f"Film {film_label(index)}"
            for index, variable in enumerate(self.film_vars[: self._active_film_count])
            if not variable.get().strip()
        ]
        if missing:
            raise ValueError(f"Choose material for: {', '.join(missing)}.")
        return super()._selected_config()

    def _start_run(self) -> None:
        if self.worker and self.worker.is_alive():
            return
        try:
            config = self._selected_config()
            report = preflight_media_inputs(config.films, output_dir=config.output_dir)
            definition = current_filter_definition(self)
        except ValueError as exc:
            messagebox.showerror("Material could not be verified", str(exc))
            return
        seconds = float(report["predicted_output_duration"])
        minutes, remainder = divmod(int(round(seconds)), 60)
        duration = f"{minutes:d}:{remainder:02d}"
        curtailed = "; the anchor will end with supporting audio" if report["anchor_curtailed"] else ""
        evidence_state = self._certification_state(config.output_dir, definition.id)
        summary = f"Input contract ready. Canonical extent {duration}{curtailed}. Prior evidence: {evidence_state}."
        self.status_var.set("Contract ready")
        self.current_operation_var.set("Awaiting schedule qualification")
        self.input_guidance_var.set(summary)
        self._append_journal("Input contract ready", summary, event_id="media_contract_ready")
        super()._start_run()

    def _run_pipeline(
        self,
        config,
        force: bool,
        app_mode: str = TRANSPOSITION,
        filter_id: str = "translation.echo",
        preference: str = "balanced",
        filter_parameters: dict[str, Any] | None = None,
    ) -> None:
        self.after(0, lambda: self.current_operation_var.set("Qualifying authored schedule against run contract"))
        super()._run_pipeline(config, force, app_mode, filter_id, preference, filter_parameters)
        evidence_state = self._certification_state(config.output_dir, filter_id)
        self.after(0, lambda: self.input_guidance_var.set(f"Latest evidence state: {evidence_state}."))

    @staticmethod
    def _certification_state(output_dir: Path, filter_id: str) -> str:
        path = output_dir / "contracts" / filter_id.replace(".", "_") / "filter_certification.json"
        if not path.exists():
            return "EXPERIMENTAL - no certified run yet"
        try:
            record = read_json(path)
        except (OSError, ValueError):
            return "EXPERIMENTAL - evidence unreadable"
        return str(record.get("state") or "EXPERIMENTAL")


def main() -> None:
    app = CinelingusInstrumentApp()
    app.mainloop()
