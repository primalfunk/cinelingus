from pathlib import Path

from cinelingus.semantic.corpus_experiment import aggregate_semantic_schedule_screens
from cinelingus.util import write_json
from cinelingus.validation import validate_artifact


def _screen(path: Path, *, mappings: int, covered: int, changed: int = 0, aggregate_fallback: int = 0) -> None:
    write_json(path, {
        "schema_version": "1.0", "experiment_version": "semantic_schedule_screen_v1",
        "creation_timestamp": "2026-01-01T00:00:00+00:00", "experiment_signature": "sig",
        "source_hash": "source", "destination_hash": "destination", "scheduling_mode": "performance_fill",
        "weight_grid": [0.0, 0.2], "semantic_model_identity": {},
        "invariants": {
            "report_only_selection_equivalent": True, "report_only_scores_equivalent": True,
            "zero_weight_selection_equivalent": True, "zero_weight_scores_equivalent": True,
        },
        "variants": [{
            "variant_id": "assisted_020", "mode": "SEMANTIC_ASSISTED", "weight": 0.2,
            "mapping_count": mappings, "semantic_placement_count": covered,
            "performance_aggregate_fallback_count": aggregate_fallback,
            "boundary_bridge_semantic_placement_count": 1 if covered - aggregate_fallback > 0 else 0,
            "text_bridge_semantic_placement_count": 0,
            "exact_direct_semantic_placement_count": max(0, covered - aggregate_fallback - 1),
            "placements_changed": changed, "conflict_count": 0,
        }],
        "render_selection": ["control", "report_only"] + (["assisted_020"] if changed else []),
        "render_selection_state": "ASSISTED_CANDIDATE_SELECTED" if changed else "NO_CONFLICT_FREE_CHANGED_ASSISTED_CANDIDATE",
        "claim_scope": "fixture",
    })


def test_corpus_aggregate_distinguishes_restraint_from_partial_coverage(tmp_path: Path) -> None:
    full, partial = tmp_path / "full.json", tmp_path / "partial.json"
    _screen(full, mappings=4, covered=4)
    _screen(partial, mappings=4, covered=3, aggregate_fallback=2)
    output = tmp_path / "semantic_corpus_screen.json"

    report = aggregate_semantic_schedule_screens([
        {"case_id": "full", "screen": full, "source_class": "animation"},
        {"case_id": "partial", "screen": partial, "source_class": "feature_animation"},
    ], output_path=output, schemas_dir=Path(__file__).parents[1] / "schemas")

    assert report["corpus_state"] == "INCOMPLETE_SEMANTIC_COVERAGE"
    assert report["summary"]["weighted_semantic_coverage"] == 0.875
    assert report["summary"]["weighted_direct_semantic_coverage"] == 0.625
    assert report["summary"]["performance_aggregate_fallback_count"] == 2
    assert report["summary"]["boundary_bridge_semantic_placement_count"] == 2
    assert report["cases"][1]["direct_semantic_placement_count"] == 1
    assert report["cases"][0]["case_state"] == "SAFE_NO_SELECTION_EFFECT"
    assert report["cases"][1]["case_state"] == "SEMANTIC_COVERAGE_PARTIAL"
    assert validate_artifact("semantic_corpus_screen", output, Path(__file__).parents[1] / "schemas")
