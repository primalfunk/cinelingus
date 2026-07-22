from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum

from ..util import stable_hash

SEMANTIC_SCHEMA_VERSION = "1.0.0"
SEMANTIC_BUILDER_VERSION = "semantic_passage_builder_v1"
DEFAULT_E5_REVISION = "d829207ab28e6a5fb3aafb5d4c44111b8146db32"


class SemanticMode(str, Enum):
    DISABLED = "SEMANTIC_DISABLED"
    REPORT_ONLY = "SEMANTIC_REPORT_ONLY"
    ASSISTED = "SEMANTIC_ASSISTED"


class SemanticTextRole(str, Enum):
    QUERY = "query"
    PASSAGE = "passage"


@dataclass(frozen=True)
class SemanticConfig:
    mode: SemanticMode = SemanticMode.DISABLED
    weight: float = 0.0
    provider: str = "local_e5_transformers"
    model_id: str = "intfloat/multilingual-e5-small"
    model_revision: str = DEFAULT_E5_REVISION
    tokenizer_id: str = "intfloat/multilingual-e5-small"
    dimensions: int = 384
    token_limit: int = 256
    truncation_policy: str = "head_tokens"
    query_prefix: str = "query: "
    passage_prefix: str = "passage: "
    pooling_policy: str = "attention_mask_mean"
    normalization: str = "l2"
    precision: str = "float32"
    device: str = "cpu"

    def __post_init__(self) -> None:
        if isinstance(self.mode, str):
            object.__setattr__(self, "mode", SemanticMode(self.mode))
        if not 0.0 <= self.weight <= 1.0:
            raise ValueError("Semantic weight must be between 0 and 1")
        if self.dimensions < 1 or self.token_limit < 1:
            raise ValueError("Semantic dimensions and token limit must be positive")
        if self.mode is not SemanticMode.ASSISTED and self.weight != 0.0:
            raise ValueError("Only SEMANTIC_ASSISTED may use a non-zero semantic weight")

    @property
    def configuration_signature(self) -> str:
        """Embedding-cache signature; scheduling mode and weight are intentionally excluded."""
        return stable_hash({
            "semantic_schema_version": SEMANTIC_SCHEMA_VERSION,
            "semantic_builder_version": SEMANTIC_BUILDER_VERSION,
            **{key: value for key, value in asdict(self).items() if key not in {"mode", "weight"}},
        })

    @property
    def scheduling_signature(self) -> str:
        return stable_hash({"embedding_configuration_signature": self.configuration_signature, "mode": self.mode, "weight": self.weight})

    def prefix_for(self, role: SemanticTextRole) -> str:
        return self.query_prefix if role is SemanticTextRole.QUERY else self.passage_prefix
