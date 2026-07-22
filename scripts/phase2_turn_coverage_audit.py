from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cinelingus.cinematic_model.turn_coverage import audit_turn_coverage
from cinelingus.util import write_json


def main() -> int:
    models = sorted((ROOT / "temp" / "phase1_corpus_models").glob("*/film_model.json"))
    if not models:
        raise FileNotFoundError("Run scripts/phase1_corpus_validation.py before the Phase 2 turn audit.")
    report = audit_turn_coverage(models)
    output = ROOT / "evaluation" / "phase2_turn_coverage_20260721.json"
    write_json(output, report)
    print(output)
    print(
        f"models={report['model_count']} passages={report['speech_passage_count']} "
        f"turns={report['dialogue_turn_count']} assigned={report['passages_assigned_to_turns']} "
        f"coverage={report['passage_assignment_percent']}% zero_turn_models={report['models_with_zero_turns']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
