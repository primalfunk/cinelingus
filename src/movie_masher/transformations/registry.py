from __future__ import annotations

from typing import Type

from .base import Transformation, TransformationContext


class TransformationRegistry:
    def __init__(self) -> None:
        self._items: dict[str, Type[Transformation]] = {}

    def register(self, transformation: Type[Transformation]) -> None:
        self._items[transformation.metadata.id] = transformation

    def get(self, transformation_id: str) -> Type[Transformation]:
        try:
            return self._items[transformation_id]
        except KeyError as exc:
            choices = ", ".join(sorted(self._items)) or "none"
            raise ValueError(f"Unknown transformation '{transformation_id}'. Available: {choices}") from exc

    def create(self, transformation_id: str, context: TransformationContext) -> Transformation:
        return self.get(transformation_id)(context)

    def ids(self) -> list[str]:
        return sorted(self._items)


def default_registry() -> TransformationRegistry:
    from .movie_masher import MovieMasherTransformation
    from .mutation_adapters import DriftTransformation, EchoTransformation
    from .self_shuffle import SelfShuffleTransformation

    registry = TransformationRegistry()
    registry.register(MovieMasherTransformation)
    registry.register(SelfShuffleTransformation)
    registry.register(EchoTransformation)
    registry.register(DriftTransformation)
    return registry
