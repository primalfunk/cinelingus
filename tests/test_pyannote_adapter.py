from movie_masher.pyannote_adapter import diarization_tracks


class Annotation:
    def itertracks(self, yield_label=False):
        assert yield_label
        yield "turn", "track", "SPEAKER_00"


class Output:
    exclusive_speaker_diarization = Annotation()


def test_current_pyannote_output_shape_is_supported() -> None:
    assert list(diarization_tracks(Output())) == [("turn", "track", "SPEAKER_00")]


def test_pair_iterable_output_shape_is_supported() -> None:
    assert list(diarization_tracks([("turn", "SPEAKER_00")])) == [("turn", 0, "SPEAKER_00")]
