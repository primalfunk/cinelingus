from __future__ import annotations

import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from huggingface_hub import snapshot_download

from cinelingus.semantic import DEFAULT_E5_REVISION
from cinelingus.util import stable_hash, write_json

MODEL_ID = "intfloat/multilingual-e5-small"
REQUIRED_FILES = (
    "config.json", "model.safetensors", "sentencepiece.bpe.model", "special_tokens_map.json",
    "tokenizer.json", "tokenizer_config.json",
)


def main() -> int:
    output = ROOT / "models" / "semantic" / "intfloat-multilingual-e5-small" / DEFAULT_E5_REVISION
    snapshot_download(
        repo_id=MODEL_ID, revision=DEFAULT_E5_REVISION, local_dir=output,
        allow_patterns=list(REQUIRED_FILES),
    )
    digests = {}
    for name in REQUIRED_FILES:
        path = output / name
        if not path.is_file():
            raise FileNotFoundError(f"Pinned model snapshot omitted required file: {name}")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        digests[name] = digest.hexdigest()
    manifest = {
        "schema_version": "1.0", "model_id": MODEL_ID, "model_revision": DEFAULT_E5_REVISION,
        "files": dict(sorted(digests.items())), "asset_digest": stable_hash(dict(sorted(digests.items()))),
        "source": "Hugging Face immutable revision; prepared only by explicit developer command",
    }
    write_json(output / "semantic_model_manifest.json", manifest)
    print(output)
    print(manifest["asset_digest"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
