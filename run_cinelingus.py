from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_translation import build_parser, main
from cinelingus.diarization_diagnostic import diagnose_diarization


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "diagnose-diarization":
        diagnose_diarization(Path(__file__).resolve().parent, Path(sys.argv[2]))
        raise SystemExit(0)
    raise SystemExit(main())
