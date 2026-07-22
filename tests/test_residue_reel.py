import wave

from cinelingus.residue_reel import build_residue_verification_reel, rebase_reel_timeline, schedule_for_verification_regions


def _wav(path, *, seconds=5, rate=1000):
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(rate)
        output.writeframes(b"\x01\x00" * seconds * rate)


def test_verification_reel_compacts_regions_and_rebases_timestamps(tmp_path) -> None:
    source = tmp_path / "source.wav"
    reel = tmp_path / "reel.wav"
    reel_map_path = tmp_path / "reel.json"
    _wav(source)

    reel_map = build_residue_verification_reel(
        input_wav=source,
        regions=[{"start": 1.0, "end": 2.0}, {"start": 4.0, "end": 4.5}],
        output_wav=reel,
        output_map=reel_map_path,
        context_padding=0.0,
        separator_seconds=0.5,
    )
    timeline = rebase_reel_timeline(
        reel_timeline={"windows": [
            {"start": 0.2, "end": 0.8, "duration": 0.6, "transcript": "first", "confidence": 0.9},
            {"start": 1.6, "end": 1.9, "duration": 0.3, "transcript": "second", "confidence": 0.8},
        ]},
        reel_map=reel_map,
    )

    assert reel_map["reel_duration"] == 2.0
    assert [(row["start"], row["end"]) for row in timeline["windows"]] == [(1.2, 1.8), (4.1, 4.4)]


def test_verification_schedule_is_scoped_to_requested_regions() -> None:
    schedule = {"destination_speech_regions": [
        {"id": "one", "start": 1.0, "end": 2.0},
        {"id": "two", "start": 5.0, "end": 6.0},
    ]}

    scoped = schedule_for_verification_regions(schedule, [{"start": 4.5, "end": 5.5}])

    assert [row["id"] for row in scoped["destination_speech_regions"]] == ["two"]


def test_rebase_assigns_separator_spanning_transcript_to_only_one_segment(tmp_path) -> None:
    source = tmp_path / "source.wav"
    _wav(source)
    reel_map = build_residue_verification_reel(
        input_wav=source,
        regions=[{"start": 1.0, "end": 2.0}, {"start": 4.0, "end": 5.0}],
        output_wav=tmp_path / "reel.wav",
        output_map=tmp_path / "reel.json",
        context_padding=0.0,
        separator_seconds=0.5,
    )

    timeline = rebase_reel_timeline(
        reel_timeline={"windows": [{
            "start": 0.8, "end": 1.8, "duration": 1.0, "transcript": "one indivisible phrase",
        }]},
        reel_map=reel_map,
    )

    assert len(timeline["windows"]) == 1
    assert timeline["windows"][0]["verification_segment_id"] == "verification_segment_000002"
    assert timeline["windows"][0]["spanned_verification_segment_count"] == 2
