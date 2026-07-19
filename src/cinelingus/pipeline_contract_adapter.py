from __future__ import annotations

from functools import wraps
from pathlib import Path
from typing import Any

from .analysis_trust import require_speaker_map_trust, speaker_cache_signature_payload
from .certification import certify_filter_run, write_filter_certification
from .contract_kernel import compile_run_contract, probe_media_descriptors, write_run_contract
from .contract_runtime import (
    activate_run_contract,
    active_multi_input_guarantee,
    active_run_contract,
    active_schedule_qualification,
)
from .filter_lab.registry import default_filter_registry
from .multi_input_guarantee import write_multi_input_guarantee
from .qualification import write_schedule_qualification
from .util import read_json


def install_pipeline_contract_adapter(pipeline_class: type) -> None:
    """Install contract compilation, evidence certification, and cache trust."""
    if getattr(pipeline_class, "_contract_adapter_installed", False):
        return
    original_execute = pipeline_class.execute_transformation
    original_signature = pipeline_class._signature
    original_source_speakers = pipeline_class.build_source_speaker_map
    original_destination_speakers = pipeline_class.build_destination_speaker_map

    @wraps(original_execute)
    def execute_with_contract(
        self,
        transformation_id: str,
        *,
        force: bool = False,
        parameters: dict[str, Any] | None = None,
    ):
        registry = default_filter_registry()
        resolved, _migration = registry.resolve_id(transformation_id)
        definition = registry.get(resolved)
        hashes = [entry.media_hash for entry in self.films]
        descriptors = probe_media_descriptors(self.config.films, media_hashes=hashes)
        contract = compile_run_contract(definition=definition, media=descriptors)
        contract_dir = self.config.output_dir / "contracts" / definition.id.replace(".", "_")
        contract_path = write_run_contract(contract, contract_dir / "run_contract.json", self.schemas_dir)
        self.run_contract = contract
        self.run_contract_path = contract_path
        with activate_run_contract(contract):
            try:
                result = original_execute(self, transformation_id, force=force, parameters=parameters)
            except Exception as exc:
                qualification_path = _persist_qualification(self, contract_dir)
                _persist_multi_input_guarantee(self, contract_dir)
                _persist_certification(
                    self,
                    contract_dir,
                    contract,
                    qualification_path=qualification_path,
                    execution_error=f"{type(exc).__name__}: {exc}",
                )
                raise
            qualification_path = _persist_qualification(self, contract_dir)
            guarantee_path = _persist_multi_input_guarantee(self, contract_dir)
            certification_path = _persist_certification(
                self,
                contract_dir,
                contract,
                result=result,
                qualification_path=qualification_path,
            )
        result.artifacts["run_contract"] = contract_path
        result.artifacts["filter_certification"] = certification_path
        if qualification_path is not None:
            result.artifacts["schedule_qualification"] = qualification_path
        if guarantee_path is not None:
            result.artifacts["multi_input_guarantee"] = guarantee_path
        return result

    @wraps(original_signature)
    def signature_with_analysis_capability(self, phase: str, *parts: Any) -> str:
        if phase == "speaker_map":
            parts = (*parts, speaker_cache_signature_payload(self.config))
        return original_signature(self, phase, *parts)

    def _speaker_wrapper(original):
        @wraps(original)
        def wrapped(self, *args: Any, **kwargs: Any):
            speaker_map = original(self, *args, **kwargs)
            contract = active_run_contract()
            if contract is not None:
                trust = require_speaker_map_trust(
                    speaker_map,
                    str(contract.analysis_requirements.get("speaker_identity", "optional")),
                )
                self.speaker_analysis_trust = trust.label
            return speaker_map

        return wrapped

    pipeline_class.execute_transformation = execute_with_contract
    pipeline_class._signature = signature_with_analysis_capability
    pipeline_class.build_source_speaker_map = _speaker_wrapper(original_source_speakers)
    pipeline_class.build_destination_speaker_map = _speaker_wrapper(original_destination_speakers)
    pipeline_class._contract_adapter_installed = True


def _persist_qualification(self, contract_dir: Path) -> Path | None:
    qualification = active_schedule_qualification()
    if qualification is None:
        return None
    path = contract_dir / "schedule_qualification.json"
    write_schedule_qualification(qualification, path, self.schemas_dir)
    self.schedule_qualification = qualification
    self.schedule_qualification_path = path
    return path


def _persist_multi_input_guarantee(self, contract_dir: Path) -> Path | None:
    guarantee = active_multi_input_guarantee()
    if guarantee is None:
        return None
    path = contract_dir / "multi_input_guarantee.json"
    write_multi_input_guarantee(guarantee, path, self.schemas_dir)
    self.multi_input_guarantee = guarantee
    self.multi_input_guarantee_path = path
    return path


def _persist_certification(
    self,
    contract_dir: Path,
    contract,
    *,
    result=None,
    qualification_path: Path | None,
    execution_error: str | None = None,
) -> Path:
    qualification = active_schedule_qualification()
    filter_acceptance = _read_result_artifact(result, "filter_acceptance")
    render_acceptance = _read_result_artifact(result, "montage_render_acceptance")
    certification = certify_filter_run(
        contract=contract,
        qualification=qualification,
        filter_acceptance=filter_acceptance,
        render_acceptance=render_acceptance,
        execution_error=execution_error,
    )
    path = contract_dir / "filter_certification.json"
    write_filter_certification(certification, path, self.schemas_dir)
    self.filter_certification = certification
    self.filter_certification_path = path
    return path


def _read_result_artifact(result, key: str) -> dict[str, Any] | None:
    if result is None:
        return None
    path = result.artifacts.get(key)
    if path is None:
        return None
    artifact_path = Path(path)
    return read_json(artifact_path) if artifact_path.exists() else None
