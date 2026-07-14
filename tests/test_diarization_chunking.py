import warnings

import torch

from movie_masher.diarization_runtime import _run_chunked_pipeline


class Segment:
    def __init__(self, start, end):
        self.start = start
        self.end = end


def test_overlapping_chunks_stitch_local_speaker_labels() -> None:
    class Pipeline:
        def __call__(self, _audio):
            return [(Segment(0.0, 10.0), "LOCAL_0"), (Segment(50.0, 60.0), "LOCAL_0")]

    audio = {"waveform": torch.zeros((1, 1000)), "sample_rate": 10}
    turns, chunk_count = _run_chunked_pipeline(Pipeline(), audio, 100.0, chunk_seconds=60.0, overlap_seconds=8.0)
    assert chunk_count == 2
    assert {turn["raw_speaker"] for turn in turns} == {"GLOBAL_001"}
    assert turns == [
        {"start": 0.0, "end": 10.0, "raw_speaker": "GLOBAL_001"},
        {"start": 50.0, "end": 62.0, "raw_speaker": "GLOBAL_001"},
    ]


def test_stitching_follows_speakers_when_local_labels_swap() -> None:
    class Pipeline:
        def __init__(self):
            self.calls = 0

        def __call__(self, _audio):
            self.calls += 1
            if self.calls == 1:
                return [(Segment(14.0, 18.0), "LOCAL_A"), (Segment(18.0, 20.0), "LOCAL_B")]
            return [(Segment(0.0, 3.0), "SWAPPED_B"), (Segment(3.0, 5.0), "SWAPPED_A")]

    audio = {"waveform": torch.zeros((1, 350)), "sample_rate": 10}
    turns, chunk_count = _run_chunked_pipeline(Pipeline(), audio, 35.0, chunk_seconds=20.0, overlap_seconds=5.0)

    assert chunk_count == 2
    assert turns == [
        {"start": 14.0, "end": 18.0, "raw_speaker": "GLOBAL_001"},
        {"start": 18.0, "end": 20.0, "raw_speaker": "GLOBAL_002"},
    ]


def test_stitching_does_not_collapse_two_local_speakers_into_one_global_id() -> None:
    class Pipeline:
        def __init__(self):
            self.calls = 0

        def __call__(self, _audio):
            self.calls += 1
            if self.calls == 1:
                return [(Segment(15.0, 20.0), "LOCAL_A")]
            return [(Segment(0.0, 3.0), "LOCAL_X"), (Segment(3.0, 5.0), "LOCAL_Y")]

    audio = {"waveform": torch.zeros((1, 350)), "sample_rate": 10}
    turns, _ = _run_chunked_pipeline(Pipeline(), audio, 35.0, chunk_seconds=20.0, overlap_seconds=5.0)

    assert turns == [
        {"start": 15.0, "end": 18.0, "raw_speaker": "GLOBAL_001"},
        {"start": 18.0, "end": 20.0, "raw_speaker": "GLOBAL_002"},
    ]


def test_chunking_reports_progress_and_never_sends_an_extra_tiny_chunk() -> None:
    progress = []

    class Pipeline:
        def __call__(self, _audio):
            return []

    audio = {"waveform": torch.zeros((1, 1000)), "sample_rate": 10}
    _, chunk_count = _run_chunked_pipeline(
        Pipeline(),
        audio,
        100.0,
        chunk_seconds=30.0,
        overlap_seconds=6.0,
        progress_callback=lambda completed, total: progress.append((completed, total)),
    )

    assert chunk_count == 4
    assert progress == [(1, 4), (2, 4), (3, 4), (4, 4)]


def test_known_empty_cluster_warnings_do_not_flood_the_operator_console() -> None:
    class Pipeline:
        def __call__(self, _audio):
            warnings.warn("Mean of empty slice", RuntimeWarning)
            warnings.warn("invalid value encountered in divide", RuntimeWarning)
            warnings.warn("std(): degrees of freedom is <= 0", UserWarning)
            return []

    audio = {"waveform": torch.zeros((1, 350)), "sample_rate": 10}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _run_chunked_pipeline(Pipeline(), audio, 35.0, chunk_seconds=20.0, overlap_seconds=5.0)

    assert caught == []
