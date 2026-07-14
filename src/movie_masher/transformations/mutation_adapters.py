from __future__ import annotations

from .base import Transformation, TransformationMetadata, TransformationResult


class _MutationAdapter(Transformation):
    mutation_id: str

    def execute(self) -> TransformationResult:
        paths = self.context.pipeline.run_mutation(self.mutation_id, force=self.context.force, parameters=self.context.parameters)
        artifacts = {key: value for key, value in paths.items() if key not in {"video", "audio"}}
        return TransformationResult(
            transformation_id=self.metadata.id,
            outputs={key: paths[key] for key in ("video", "audio") if key in paths},
            artifacts=artifacts,
        )


class EchoTransformation(_MutationAdapter):
    mutation_id = "echo"
    metadata = TransformationMetadata(
        id="echo",
        display_name="Echo",
        description="Repeat selected dialogue later in the same film.",
        required_inputs=("destination_video",),
        generated_outputs=("echo_schedule.json", "echo_audio.wav", "echo_output.mp4", "filter_recipe.json", "filter_plan.json"),
        supported_modes=("fast_preview", "balanced", "quality"),
        version="1.0",
    )


class DriftTransformation(_MutationAdapter):
    mutation_id = "drift"
    metadata = TransformationMetadata(
        id="drift",
        display_name="Drift",
        description="Move dialogue progressively later while preserving the picture.",
        required_inputs=("destination_video",),
        generated_outputs=("drift_schedule.json", "drift_audio.wav", "drift_output.mp4", "filter_recipe.json", "filter_plan.json"),
        supported_modes=("fast_preview", "balanced", "quality"),
        version="1.0",
    )
