from __future__ import annotations

from pathlib import Path
import time

from PIL import ImageGrab

from cinelingus.gui import CinelingusInstrumentApp


def capture(app: CinelingusInstrumentApp, path: Path) -> None:
    app.update_idletasks()
    app.update()
    app.lift()
    app.attributes('-topmost', True)
    app.update()
    time.sleep(0.35)
    app.update_idletasks()
    app.update()
    x, y = app.winfo_rootx(), app.winfo_rooty()
    width, height = app.winfo_width(), app.winfo_height()
    ImageGrab.grab(bbox=(x, y, x + width, y + height), all_screens=True).save(path)
    app.attributes('-topmost', False)


def main() -> None:
    output = Path.cwd() / "output" / "ui_overhaul" / "screenshots"
    output.mkdir(parents=True, exist_ok=True)
    app = CinelingusInstrumentApp()
    try:
        assert app.instrument_canvas.plate_available
        assert len(app.instrument_canvas._overlays) == 10
        app.geometry("1120x780+80+60")
        app._show_wizard_step(1)
        capture(app, output / "01_initial.png")

        app._show_wizard_step(3)
        app._set_running(True, "Experiment in progress")
        app.stage_var.set("Examining recurring voices")
        app.current_operation_var.set("Examining recurring voices")
        app.overall_progress_var.set(58)
        app.progress_percent_var.set("58%")
        app.live_elapsed_var.set("08:14")
        app.live_idle_var.set("00:30")
        app.live_eta_var.set("Calculating...")
        app.specimen_var.set("MADtv - S01 E19")
        app.completed_stage_durations = {"inspect": 8.0, "source_dialogue": 211.0}
        app._mark_stage("destination_speech")
        app._append_journal("Specimens catalogued", "The selected material has been catalogued.", event_id="demo_inspect")
        app._append_journal("Spoken record isolated", "Spoken passages have been separated for examination.", event_id="demo_speech")
        app._append_journal("Examining recurring voices", "Recurring vocal identities are under examination.", event_id="demo_voices")
        capture(app, output / "02_active.png")

        app._append_journal(
            "Observation period exceeded",
            "The vocal examination exceeded its allotted observation period. Continuing by an alternate method.",
            severity="warning",
            event_id="demo_warning",
        )
        app._append_console("Pyannote inference timeout after 780 seconds; fallback=heuristic; device=cuda.\n")
        app.technical_record_var.set(True)
        app._toggle_technical_record()
        capture(app, output / "03_warning.png")

        app._set_running(False, "Experiment complete")
        app._mark_stage(None, finished=True)
        app.output_path_var.set(str(Path.cwd() / "output" / "cinelingus_translation_2026-07-13_17-39-34.mp4"))
        app.completion_summary_var.set(
            "Mode: Translation\n"
            "Source: Hey Dude - Day One at the Bar None\n"
            "Destination: MADtv - S01 E19\n"
            "Observations: 5 exchanges selected\n"
            "Final duration: 04:06\n\n"
            "Vocal analysis: Temporal estimates were used after the full examination exceeded its allotted time."
        )
        app.problem_summary_var.set("Artifact examination passed. Audio provenance and dialogue coverage were verified.")
        app.last_truth_var.set("The resulting artifact has been archived in the selected output folder.")
        app._show_wizard_step(4)
        app.withdraw()
        app.update()
        app.deiconify()
        app.geometry("1120x780+80+60")
        capture(app, output / "04_completed.png")
    finally:
        app.destroy()


if __name__ == "__main__":
    main()
