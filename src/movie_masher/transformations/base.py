from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from movie_masher.filter_lab.integration import write_filter_artifacts


@dataclass(frozen=True)
class TransformationMetadata:
    id: str
    display_name: str
    description: str
    required_inputs: tuple[str, ...]
    generated_outputs: tuple[str, ...]
    supported_modes: tuple[str, ...]
    version: str


@dataclass
class TransformationContext:
    pipeline: Any
    force: bool = False
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class TransformationResult:
    transformation_id: str
    outputs: dict[str, Path]
    artifacts: dict[str, Path] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class Transformation(ABC):
    metadata: TransformationMetadata

    def __init__(self, context: TransformationContext) -> None:
        self.context = context

    def initialize(self) -> None:
        pass

    def validate_inputs(self) -> None:
        pass

    def select(self) -> dict[str, Any]:
        return {}

    def transform(self, selections: dict[str, Any]) -> dict[str, Any]:
        return {}

    def validate(self, transformed: dict[str, Any]) -> None:
        pass

    def create_filter_artifacts(self, transformed: dict[str, Any]) -> dict[str, Path]:
        schedule = transformed.get("schedule", {})
        if not schedule.get("mappings"):
            return {}
        pipeline = getattr(self, "_working_pipeline", None) or self.context.pipeline
        output_dir = pipeline.config.output_dir / self.metadata.id
        return write_filter_artifacts(
            pipeline=pipeline,
            filter_id=self.metadata.id,
            parameters=self.context.parameters,
            schedule=schedule,
            output_dir=output_dir,
            output_form="full_length",
            target_duration=getattr(pipeline.config, "target_duration_seconds", None),
        )

    def render(self, transformed: dict[str, Any]) -> dict[str, Path]:
        return {}

    def generate_report(self, result: TransformationResult) -> Path | None:
        return None

    def cleanup(self) -> None:
        pass

    def execute(self) -> TransformationResult:
        self.initialize()
        try:
            self.validate_inputs()
            selections = self.select()
            transformed = self.transform(selections)
            self.validate(transformed)
            artifacts = self.create_filter_artifacts(transformed)
            outputs = self.render(transformed)
            result = TransformationResult(transformation_id=self.metadata.id, outputs=outputs, artifacts=artifacts)
            report = self.generate_report(result)
            if report is not None:
                result.artifacts["transformation_report"] = report
            return result
        finally:
            self.cleanup()
