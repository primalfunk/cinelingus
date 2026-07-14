import multiprocessing as mp

import torch

from movie_masher.diarization_runtime import (
    _configure_cuda_memory_limit,
    _receive_worker_result,
    _resolve_pipeline_output,
)


def _put_large_worker_result(result_queue) -> None:
    result_queue.put({"kind": "progress", "phase": "model_loaded"})
    result_queue.put({"kind": "result", "ok": True, "blob": "x" * (1024 * 1024)})


def test_single_file_generator_return_value_is_recovered() -> None:
    expected = object()

    def pyannote_style_call():
        if False:
            yield None
        return expected

    assert _resolve_pipeline_output(pyannote_style_call()) is expected


def test_non_generator_pipeline_output_is_unchanged() -> None:
    expected = object()
    assert _resolve_pipeline_output(expected) is expected


def test_large_worker_result_is_consumed_before_joining_worker() -> None:
    context = mp.get_context("spawn")
    result_queue = context.Queue()
    worker = context.Process(target=_put_large_worker_result, args=(result_queue,))
    progress = []
    worker.start()

    payload, timeout_reason = _receive_worker_result(
        worker,
        result_queue,
        inactivity_timeout_seconds=10,
        total_timeout_seconds=20,
        progress_callback=progress.append,
    )
    worker.join(5)

    assert timeout_reason is None
    assert payload is not None and payload["ok"] is True
    assert len(payload["blob"]) == 1024 * 1024
    assert progress == [{"kind": "progress", "phase": "model_loaded"}]
    assert not worker.is_alive()
    result_queue.close()
    result_queue.join_thread()


def test_cuda_memory_limit_passes_numeric_current_device(monkeypatch) -> None:
    call = {}
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 3)
    monkeypatch.setattr(
        torch.cuda,
        "set_per_process_memory_fraction",
        lambda fraction, device: call.update(fraction=fraction, device=device),
    )

    assert _configure_cuda_memory_limit(torch.device("cuda")) == 0.65
    assert call == {"fraction": 0.65, "device": 3}
    assert isinstance(call["device"], int)


def test_cuda_memory_limit_preserves_explicit_numeric_device(monkeypatch) -> None:
    call = {}
    monkeypatch.setattr(
        torch.cuda,
        "set_per_process_memory_fraction",
        lambda fraction, device: call.update(fraction=fraction, device=device),
    )

    assert _configure_cuda_memory_limit(torch.device("cuda:2")) == 0.65
    assert call == {"fraction": 0.65, "device": 2}
