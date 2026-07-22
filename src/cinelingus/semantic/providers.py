from __future__ import annotations

import hashlib
import math
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

from ..util import read_json, stable_hash

from .config import SemanticConfig, SemanticTextRole


class SemanticProviderUnavailable(RuntimeError):
    def __init__(self, state: str, message: str):
        super().__init__(message)
        self.state = state


@dataclass(frozen=True)
class EmbeddingBatch:
    vectors: tuple[tuple[float, ...], ...]
    token_counts: tuple[int, ...]
    truncated: tuple[bool, ...]
    provider_metadata: dict[str, object]


class SemanticProvider(Protocol):
    def describe(self) -> dict[str, object]: ...
    def encode(self, texts: Sequence[str], *, role: SemanticTextRole) -> EmbeddingBatch: ...


class DeterministicFakeProvider:
    def __init__(self, config: SemanticConfig):
        self.config = config

    def describe(self) -> dict[str, object]:
        return _metadata(self.config, provider="deterministic_fake", runtime="stdlib_sha256_v1")

    def encode(self, texts: Sequence[str], *, role: SemanticTextRole) -> EmbeddingBatch:
        vectors: list[tuple[float, ...]] = []
        counts: list[int] = []
        truncated: list[bool] = []
        for text in texts:
            tokens = (self.config.prefix_for(role) + text).split()
            counts.append(len(tokens))
            truncated.append(len(tokens) > self.config.token_limit)
            canonical = " ".join(tokens[: self.config.token_limit]).encode("utf-8")
            values = []
            counter = 0
            while len(values) < self.config.dimensions:
                digest = hashlib.sha256(canonical + counter.to_bytes(4, "big")).digest()
                values.extend((byte - 127.5) / 127.5 for byte in digest)
                counter += 1
            vectors.append(_normalize(values[: self.config.dimensions]))
        return EmbeddingBatch(tuple(vectors), tuple(counts), tuple(truncated), self.describe())


class UnavailableProvider:
    def __init__(self, config: SemanticConfig, *, state: str = "UNAVAILABLE", reason: str = "Semantic provider is unavailable."):
        self.config, self.state, self.reason = config, state, reason

    def describe(self) -> dict[str, object]:
        return {**_metadata(self.config, provider="unavailable", runtime="none"), "availability": self.state, "reason": self.reason}

    def encode(self, texts: Sequence[str], *, role: SemanticTextRole) -> EmbeddingBatch:
        raise SemanticProviderUnavailable(self.state, self.reason)


class LocalE5Provider:
    """Lazy, local-only E5 inference. Construction never downloads model assets."""

    def __init__(self, config: SemanticConfig, *, asset_dir: Path | None = None, allow_download: bool = False):
        self.config = config
        self.asset_dir = asset_dir
        self.allow_download = allow_download
        self._tokenizer = None
        self._model = None
        self._torch = None

    def describe(self) -> dict[str, object]:
        metadata = _metadata(self.config, provider="local_e5_transformers", runtime="transformers")
        try:
            import torch
            import transformers
            metadata.update({"torch_version": torch.__version__, "transformers_version": transformers.__version__, "cuda_available": torch.cuda.is_available()})
        except ImportError:
            metadata.update({"availability": "DOWNLOAD_REQUIRED", "cuda_available": False})
        manifest = self.asset_dir / "semantic_model_manifest.json" if self.asset_dir else None
        if manifest and manifest.is_file():
            metadata["asset_digest"] = read_json(manifest).get("asset_digest")
            metadata["asset_locator"] = self.asset_dir.resolve().as_posix()
        return metadata

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise SemanticProviderUnavailable("DOWNLOAD_REQUIRED", "Install the semantic optional dependencies before preparing local E5 assets.") from exc
        source = str(self.asset_dir) if self.asset_dir else self.config.model_id
        kwargs = {"local_files_only": True} if self.asset_dir else {"revision": self.config.model_revision, "local_files_only": not self.allow_download}
        try:
            if self.asset_dir:
                _verify_asset_manifest(self.asset_dir, self.config)
            tokenizer = AutoTokenizer.from_pretrained(source, **kwargs)
            model = AutoModel.from_pretrained(source, **kwargs)
        except (OSError, ValueError) as exc:
            state = "DOWNLOAD_REQUIRED" if not self.allow_download else "UNAVAILABLE"
            raise SemanticProviderUnavailable(state, f"Pinned E5 assets are not available: {exc}") from exc
        device = torch.device(self.config.device)
        model.to(device)
        model.eval()
        self._tokenizer, self._model, self._torch = tokenizer, model, torch

    def encode(self, texts: Sequence[str], *, role: SemanticTextRole) -> EmbeddingBatch:
        self._load()
        assert self._tokenizer is not None and self._model is not None and self._torch is not None
        prefixed = [self.config.prefix_for(role) + text for text in texts]
        lengths = [len(self._tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"]) for text in prefixed]
        encoded = self._tokenizer(
            prefixed, max_length=self.config.token_limit, truncation=True, padding=True, return_tensors="pt",
        )
        encoded = {key: value.to(self.config.device) for key, value in encoded.items()}
        with self._torch.inference_mode():
            output = self._model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).expand(output.size()).float()
            pooled = (output * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            pooled = self._torch.nn.functional.normalize(pooled, p=2, dim=1).to(dtype=self._torch.float32).cpu()
        vectors = tuple(tuple(float(value) for value in row.tolist()) for row in pooled)
        if any(len(row) != self.config.dimensions for row in vectors):
            raise ValueError("Pinned E5 provider returned an unexpected embedding dimension")
        return EmbeddingBatch(vectors, tuple(lengths), tuple(length > self.config.token_limit for length in lengths), self.describe())


def _metadata(config: SemanticConfig, *, provider: str, runtime: str) -> dict[str, object]:
    return {
        "provider": provider, "model_id": config.model_id, "model_revision": config.model_revision,
        "tokenizer_id": config.tokenizer_id, "dimensions": config.dimensions,
        "prefix_policy": {"query": config.query_prefix, "passage": config.passage_prefix},
        "token_limit": config.token_limit, "truncation_policy": config.truncation_policy,
        "pooling_policy": config.pooling_policy, "normalization": config.normalization,
        "precision": config.precision, "execution_device": config.device,
        "python_version": platform.python_version(), "runtime": runtime,
    }


def _normalize(values: Sequence[float]) -> tuple[float, ...]:
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0.0:
        raise ValueError("Cannot normalize a zero semantic vector")
    return tuple(float(value / norm) for value in values)


def _verify_asset_manifest(asset_dir: Path, config: SemanticConfig) -> None:
    manifest_path = asset_dir / "semantic_model_manifest.json"
    if not manifest_path.is_file():
        raise SemanticProviderUnavailable("DOWNLOAD_REQUIRED", f"Verified semantic model manifest is missing from {asset_dir}")
    manifest = read_json(manifest_path)
    if manifest.get("model_id") != config.model_id or manifest.get("model_revision") != config.model_revision:
        raise SemanticProviderUnavailable("UNAVAILABLE", "Semantic model manifest identity does not match the pinned configuration.")
    observed: dict[str, str] = {}
    for relative, expected in sorted((manifest.get("files") or {}).items()):
        path = asset_dir / relative
        if not path.is_file():
            raise SemanticProviderUnavailable("DOWNLOAD_REQUIRED", f"Pinned semantic model asset is missing: {relative}")
        digest = _file_sha256(path)
        if digest != expected:
            raise SemanticProviderUnavailable("UNAVAILABLE", f"Pinned semantic model asset digest mismatch: {relative}")
        observed[relative] = digest
    if stable_hash(observed) != manifest.get("asset_digest"):
        raise SemanticProviderUnavailable("UNAVAILABLE", "Semantic model aggregate asset digest is invalid.")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
