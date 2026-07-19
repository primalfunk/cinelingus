from pathlib import Path

from cinelingus.operations import RepeatOperation, ReplaceOperation, ShuffleOperation
from cinelingus.placement import PlaceIntoPerformances
from cinelingus.selections import SelectDialogue, SelectPerformances
from cinelingus.transformation_plan import (
    TRANSFORMATION_VOCABULARY,
    VERB_PLACE,
    VERB_RENDER,
    VERB_REPLACE,
    VERB_SELECT,
    build_translation_plan,
    write_transformation_plan,
)
from cinelingus.validation import validate_artifact


def test_build_translation_plan_describes_vocabulary_and_counts(tmp_path: Path) -> None:
    plan = build_translation_plan(
        root=tmp_path,
        destination_movie={"path": "dest.mp4", "media_hash": "desthash", "duration": 10.0},
        source_movie={"path": "source.mp4", "media_hash": "sourcehash", "duration": 8.0},
        clip_library={"clips": [{"id": "c1", "duration": 2.0}, {"id": "c2", "duration": 0.0}]},
        destination_timeline={"windows": [{"id": "w1", "duration": 1.0}]},
        visual={"shots": {"shots": [{"id": "shot_000001"}]}},
        source_performances={"performances": [{"id": "sp1", "duration": 2.0}]},
        destination_performances={"performances": [{"id": "dp1", "duration": 2.5}]},
        output_dir=tmp_path / "output",
        max_time_stretch=1.1,
    )
    assert plan["transformation"]["display_name"] == "Translation"
    path = write_transformation_plan(
        plan=plan,
        output_path=tmp_path / "output" / "translation" / "transformation_plan.json",
        latest_path=tmp_path / "output" / "transformation_plan.json",
        schemas_dir=Path.cwd() / "schemas",
    )

    assert path.exists()
    assert plan["vocabulary"] == TRANSFORMATION_VOCABULARY
    assert plan["transformation"]["lifecycle"][:2] == [VERB_SELECT, "TRANSFORM"]
    assert plan["operations"][0]["verb"] == VERB_REPLACE
    assert plan["placement"]["verb"] == VERB_PLACE
    assert plan["render"]["verb"] == VERB_RENDER
    assert plan["selection"][0]["count"] == 1
    validate_artifact("transformation_plan", path, Path.cwd() / "schemas")
    validate_artifact("transformation_plan", tmp_path / "output" / "transformation_plan.json", Path.cwd() / "schemas")


def test_selection_operation_and_placement_primitives() -> None:
    dialogue = SelectDialogue(role="source_dialogue", source_artifact="clip_library").select(
        {"clips": [{"id": "c1", "duration": 1.0}, {"id": "c2", "duration": 0.0}]}
    )
    performances = SelectPerformances(role="destination_video", source_artifact="performance").select(
        {"performances": [{"id": "p1", "duration": 2.0}]}
    )
    replace = ReplaceOperation().apply(dialogue.objects, performances.objects)
    shuffled, shuffle = ShuffleOperation(seed=1).apply(dialogue.objects)
    repeated, repeat = RepeatOperation(times=2).apply(dialogue.objects)
    placement = PlaceIntoPerformances().plan(dialogue.objects, performances.objects)

    assert dialogue.to_plan_entry()["object_ids"] == ["c1"]
    assert performances.to_plan_entry()["object_ids"] == ["p1"]
    assert replace.output_count == 1
    assert shuffle.output_count == len(shuffled) == 1
    assert repeat.output_count == len(repeated) == 2
    assert placement.placement_count == 1
