from __future__ import annotations

import importlib
import os
import sys
from typing import Any


OUTPUT_FORM = "Full Source Timeline"


def legacy_main() -> None:
    """Launch the retired Tk shell only when explicitly requested."""
    from .legacy_gui import main as run_legacy

    run_legacy()


def main(argv: list[str] | None = None) -> int | None:
    """Launch the production Qt interface without importing Tk."""
    args = list(sys.argv[1:] if argv is None else argv)
    legacy_requested = "--legacy-tk" in args or os.environ.get("CINELINGUS_LEGACY_TK", "").lower() in {"1", "true", "yes"}
    if legacy_requested:
        legacy_main()
        return None
    from .qt_faceplate import main as qt_main

    return qt_main(args)


def __getattr__(name: str) -> Any:
    """Lazily expose the historical helper API without loading Tk on Qt startup."""
    if name == "CinelingusInstrumentApp":
        from .legacy_gui import CinelingusInstrumentApp

        return CinelingusInstrumentApp
    implementation = importlib.import_module(".gui_implementation", __package__)
    try:
        return getattr(implementation, name)
    except AttributeError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
