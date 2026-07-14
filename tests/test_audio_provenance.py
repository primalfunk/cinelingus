from pathlib import Path
import wave

import pytest

from movie_masher.audio_provenance import AudioProvenanceError, compare_wav_audio, verify_audio_provenance


def _wav(path: Path, samples: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(48000)
        frames = b"".join(int(sample).to_bytes(2, "little", signed=True) for sample in samples)
        handle.writeframes(frames)


def _schedule(tmp_path: Path, source_hash: str = "sourcehash") -> tuple[dict, dict]:
    clip = tmp_path / "cache" / source_hash / "clips" / "000001.wav"
    clip.parent.mkdir(parents=True, exist_ok=True)
    clip.write_text("clip")
    mapping = {
        "enabled": True,
        "clip_id": "c000001",
        "clip_path": str(clip),
        "destination_timestamp": 1.0,
        "source_movie_timestamp": 10.0,
        "source_transcript": "source line",
    }
    schedule = {
        "source_media_hash": source_hash,
        "destination_media_hash": "desthash",
        "mappings": [mapping],
    }
    short_schedule = {"mappings": [mapping]}
    return schedule, short_schedule


def test_compare_wav_audio_reports_difference(tmp_path: Path) -> None:
    left = tmp_path / "left.wav"
    right = tmp_path / "right.wav"
    _wav(left, [1000, 1000, 1000])
    _wav(right, [1000, 0, -1000])

    result = compare_wav_audio(left_path=left, right_path=right)

    assert result["diff_rms"] > 0
    assert result["compared_frames"] == 3


def test_audio_provenance_passes_for_selected_source_hash(tmp_path: Path) -> None:
    replacement = tmp_path / "replacement.wav"
    _wav(replacement, [1000] * 100)
    schedule, short_schedule = _schedule(tmp_path)

    report = verify_audio_provenance(
        root=tmp_path,
        destination_video=tmp_path / "dest.mp4",
        destination_hash="desthash",
        source_dialogue=tmp_path / "source.mp4",
        source_hash="sourcehash",
        schedule=schedule,
        short_schedule=short_schedule,
        replacement_audio=replacement,
        final_video=tmp_path / "FINAL.mp4",
        visual_segment=tmp_path / "_visual_segment_original_audio_DO_NOT_REVIEW.mp4",
        output_path=tmp_path / "audio_provenance.json",
        final_audio_analysis={"diff_from_replacement_rms": 20.0, "diff_from_original_segment_rms": 1000.0},
        original_segment_analysis={"rms": 1000.0},
    )

    assert report["status"] == "pass"
    assert report["checks"]["all_clip_roots_match_source"] is True


def test_audio_provenance_fails_when_schedule_uses_wrong_source_hash(tmp_path: Path) -> None:
    replacement = tmp_path / "replacement.wav"
    _wav(replacement, [1000] * 100)
    schedule, short_schedule = _schedule(tmp_path, source_hash="wronghash")

    with pytest.raises(AudioProvenanceError, match="source_hash_matches_schedule"):
        verify_audio_provenance(
            root=tmp_path,
            destination_video=tmp_path / "dest.mp4",
            destination_hash="desthash",
            source_dialogue=tmp_path / "source.mp4",
            source_hash="sourcehash",
            schedule=schedule,
            short_schedule=short_schedule,
            replacement_audio=replacement,
            final_video=tmp_path / "FINAL.mp4",
            visual_segment=tmp_path / "_visual_segment_original_audio_DO_NOT_REVIEW.mp4",
            output_path=tmp_path / "audio_provenance.json",
            final_audio_analysis={"diff_from_replacement_rms": 20.0, "diff_from_original_segment_rms": 1000.0},
            original_segment_analysis={"rms": 1000.0},
        )


def test_audio_provenance_rejects_mostly_silent_replacement(tmp_path: Path) -> None:
    replacement = tmp_path / "replacement.wav"
    _wav(replacement, [2000] * 4800 + [0] * 43200)
    schedule, short_schedule = _schedule(tmp_path)

    with pytest.raises(AudioProvenanceError, match="replacement_audio_has_sufficient_activity"):
        verify_audio_provenance(
            root=tmp_path,
            destination_video=tmp_path / "dest.mp4",
            destination_hash="desthash",
            source_dialogue=tmp_path / "source.mp4",
            source_hash="sourcehash",
            schedule=schedule,
            short_schedule=short_schedule,
            replacement_audio=replacement,
            final_video=tmp_path / "FINAL.mp4",
            visual_segment=tmp_path / "_visual_segment_original_audio_DO_NOT_REVIEW.mp4",
            output_path=tmp_path / "audio_provenance.json",
            final_audio_analysis={"diff_from_replacement_rms": 20.0, "diff_from_original_segment_rms": 1000.0},
            original_segment_analysis={"rms": 1000.0},
        )
