from __future__ import annotations

from pathlib import Path
import logging
import sys
import warnings

# Torch probes optional Triton FLOP-counting support while importing CUDA
# utilities. Cinelingus never uses that profiler. Pyannote also emits a
# one-time notice after deliberately disabling TF32 for reproducible, more
# accurate diarization. Keep the safe behavior and remove only these notices.
logging.getLogger("torch.utils.flop_counter").setLevel(logging.ERROR)
warnings.filterwarnings(
    "ignore",
    message=r"TensorFloat-32 \(TF32\) has been disabled.*",
    category=UserWarning,
    module=r"pyannote\.audio\.utils\.reproducibility",
)

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
