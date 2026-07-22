from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cinelingus.cinematic_model.corpus_validation import CorpusCase, validate_corpus_cases
from cinelingus.util import write_json


def main() -> int:
    cache = ROOT / "cache"
    short = ROOT / "output" / "corpus_calibration" / "smoke" / "case_002_transition_sentence_integrity_6a92c69a" / "cache" / "3e396cdd138eb12c11ee8614e463d2abe8dc4bfd134f9fa5deddae74791a6718" / "destination_video"
    cases = (
        CorpusCase("live_action_wkyk", "live_action_strong_diarization", cache / "73d7903e2a4e16f148b8f50ee59e19373e38ada56e1bbd6f5fc4865606268b26" / "source_dialogue", "artifact_rich_dialogue"),
        CorpusCase("animation_mega_man", "animation", cache / "1879b4266b4546984d3113a70a48cf5c51a797f120caebfb1e586d9203bbb98b" / "destination_video", "partial_no_speaker_map"),
        CorpusCase("short_form_excerpt", "short_form", short, "analyzed_phase0_excerpt"),
        CorpusCase("feature_wallace_gromit", "feature_length_animation", cache / "62311279e6dad49448aca3c376f5bdb00a9305e32a90ea910df3d3709b6d0017" / "destination_video", "artifact_rich_feature"),
        CorpusCase("standard_red_dwarf", "live_action", cache / "d8f348a6441985db96ca9bf39e4d46217062c1e9283014253ac729fb2564e459" / "destination_video", "artifact_rich", tier="standard"),
        CorpusCase("standard_magic_schoolbus", "animation", cache / "b2203a07578ab221c829e9e4759841e381041bfa53feb25e80919f562110eae1" / "destination_video", "artifact_rich", tier="standard"),
        CorpusCase("standard_star_trek", "live_action_feature_scale", cache / "a0c757cdae48cfdba4304c28627ade20ae4b52251dde3b96834f6e9ae5331b44" / "destination_video", "schedule_bearing", tier="standard"),
        CorpusCase("standard_sleeping_beauty", "mixed_feature_scale", cache / "18f6b6ed542cca53c693bc5e361d168ba10bf85785b830639898821c8d7cf589" / "destination_video", "schedule_bearing", tier="standard"),
    )
    report = validate_corpus_cases(cases, schemas_dir=ROOT / "schemas", output_root=ROOT / "temp" / "phase1_corpus_models")
    output = ROOT / "evaluation" / "phase1_bounded_corpus_20260721.json"
    write_json(output, report)
    print(output)
    print(f"cases={report['case_count']} valid={report['all_valid']} deterministic={report['all_deterministic']} cache_hits={report['all_cache_hits']} source_unchanged={report['source_media_unchanged']}")
    return 0 if all((report["all_valid"], report["all_deterministic"], report["all_cache_hits"], report["source_media_unchanged"])) else 1


if __name__ == "__main__":
    raise SystemExit(main())
