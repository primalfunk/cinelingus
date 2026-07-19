from __future__ import annotations

from typing import Any, Iterable


def diarization_tracks(output: Any) -> Iterable[tuple[Any, Any, str]]:
    """Normalize current and legacy pyannote pipeline output shapes."""
    annotation = None
    if isinstance(output, dict):
        annotation = output.get("exclusive_speaker_diarization") or output.get("speaker_diarization")
    else:
        annotation = getattr(output, "exclusive_speaker_diarization", None)
        if annotation is None:
            annotation = getattr(output, "speaker_diarization", None)
    annotation = annotation if annotation is not None else output
    # pyannote.audio 4/community-1 exposes direct (segment, speaker) pairs.
    # Prefer that public API before the legacy Annotation.itertracks method.
    try:
        direct_rows = list(annotation)
    except TypeError:
        direct_rows = []
    emitted = False
    for index, row in enumerate(direct_rows):
        if isinstance(row, tuple) and len(row) == 2 and hasattr(row[0], "start"):
            emitted = True
            yield row[0], index, str(row[1])
        elif isinstance(row, tuple) and len(row) >= 3 and hasattr(row[0], "start"):
            emitted = True
            yield row[0], row[1], str(row[2])
    if emitted:
        return
    if hasattr(annotation, "itertracks"):
        yield from annotation.itertracks(yield_label=True)
        return
    for index, row in enumerate(direct_rows):
        if isinstance(row, tuple) and len(row) == 2:
            yield row[0], index, str(row[1])
        elif isinstance(row, tuple) and len(row) >= 3:
            yield row[0], row[1], str(row[2])