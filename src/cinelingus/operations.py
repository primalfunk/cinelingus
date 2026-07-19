from __future__ import annotations

import random
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class OperationResult:
    operation: str
    input_count: int
    output_count: int
    parameters: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_plan_entry(self) -> dict[str, Any]:
        return {
            "operation": self.operation,
            "input_count": self.input_count,
            "output_count": self.output_count,
            "parameters": self.parameters,
            "warnings": self.warnings,
        }


class BaseOperation:
    operation = "Operation"

    def apply(self, items: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], OperationResult]:
        output = [deepcopy(item) for item in items]
        return output, OperationResult(self.operation, len(items), len(output))


class ReplaceOperation(BaseOperation):
    operation = "ReplaceOperation"

    def apply(self, source_items: list[dict[str, Any]], destination_items: list[dict[str, Any]]) -> OperationResult:
        return OperationResult(
            operation=self.operation,
            input_count=len(source_items) + len(destination_items),
            output_count=min(len(source_items), len(destination_items)),
            parameters={"source_object_type": "dialogue", "destination_object_type": "performance"},
            warnings=[] if source_items and destination_items else ["replace operation has an empty selection"],
        )


class MoveOperation(BaseOperation):
    operation = "MoveOperation"


class ShuffleOperation(BaseOperation):
    operation = "ShuffleOperation"

    def __init__(self, *, seed: int | None = None) -> None:
        self.seed = seed

    def apply(self, items: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], OperationResult]:
        output = [deepcopy(item) for item in items]
        random.Random(self.seed).shuffle(output)
        return output, OperationResult(self.operation, len(items), len(output), {"seed": self.seed})


class RepeatOperation(BaseOperation):
    operation = "RepeatOperation"

    def __init__(self, *, times: int) -> None:
        self.times = max(0, int(times))

    def apply(self, items: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], OperationResult]:
        output: list[dict[str, Any]] = []
        for _ in range(self.times):
            output.extend(deepcopy(items))
        return output, OperationResult(self.operation, len(items), len(output), {"times": self.times})


class StretchOperation(BaseOperation):
    operation = "StretchOperation"

    def __init__(self, *, max_factor: float) -> None:
        self.max_factor = float(max_factor)

    def apply(self, items: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], OperationResult]:
        output = [deepcopy(item) for item in items]
        return output, OperationResult(self.operation, len(items), len(output), {"max_factor": self.max_factor})


class CompressOperation(StretchOperation):
    operation = "CompressOperation"
