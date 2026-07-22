from __future__ import annotations

from pathlib import Path
from statistics import mean
from typing import Any

from . import __version__
from .util import utc_now, write_json


VISUAL_PERFORMANCE_VERSION = "visual_performance_v2_conservative_face_unknown"


def analyze_visual_performance(
    *,
    media_path: Path,
    media_hash: str,
    shots: list[dict[str, Any]],
    speech_windows: list[dict[str, Any]],
    output_path: Path,
    config_signature: str,
    frame_evidence: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    backend = "provided_frame_evidence"
    backend_status = "AVAILABLE"
    if frame_evidence is None:
        try:
            frame_evidence = extract_frame_evidence(media_path=media_path, shots=shots)
            backend = "opencv_haar_phase_correlation"
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            frame_evidence = {}
            backend = "conservative_fallback"
            backend_status = f"UNAVAILABLE: {type(exc).__name__}: {str(exc)[:160]}"
    rows = [
        describe_shot_performance(
            shot=shot,
            samples=frame_evidence.get(str(shot.get("id")), []),
            speech_windows=speech_windows,
        )
        for shot in shots
    ]
    artifact = {
        "schema_version": "1.0",
        "tool_version": __version__,
        "analysis_version": VISUAL_PERFORMANCE_VERSION,
        "creation_timestamp": utc_now(),
        "media_hash": media_hash,
        "config_signature": config_signature,
        "backend": backend,
        "backend_status": backend_status,
        "shot_count": len(rows),
        "shots": rows,
    }
    write_json(output_path, artifact)
    return artifact


def describe_shot_performance(
    *,
    shot: dict[str, Any],
    samples: list[dict[str, Any]],
    speech_windows: list[dict[str, Any]],
) -> dict[str, Any]:
    start, end = _bounds(shot)
    duration = max(0.001, end - start)
    speech_duration = _union_duration([
        (max(start, _bounds(row)[0]), min(end, _bounds(row)[1]))
        for row in speech_windows
        if _overlap(start, end, *_bounds(row))
    ])
    speech_probability = _clamp(speech_duration / duration)
    faces = [face for sample in samples for face in sample.get("faces", [])]
    face_counts = [len(sample.get("faces", [])) for sample in samples]
    face_count = round(mean(face_counts), 3) if face_counts else 0.0
    face_confidence = _clamp(len(samples) / 3.0 * (0.2 + min(face_count, 2.0) * 0.2)) if samples else 0.0
    mouth_values = [float(face.get("mouth_movement", 0.0) or 0.0) for face in faces]
    mouth_activity = mean(mouth_values) if mouth_values else 0.0
    optical_motion = mean(float(row.get("optical_motion", 0.0) or 0.0) for row in samples) if samples else 0.0
    camera_motion = mean(float(row.get("camera_motion", 0.0) or 0.0) for row in samples) if samples else 0.0
    subject_motion = mean(float(row.get("subject_motion", 0.0) or 0.0) for row in samples) if samples else 0.0
    action_level = _clamp(subject_motion * 1.8 + optical_motion * 0.5)
    stillness = _clamp(1.0 - optical_motion * 2.2)
    if faces:
        close_up = _clamp(max(float(face.get("area_ratio", 0.0) or 0.0) for face in faces) * 5.0)
        wide = _clamp(1.0 - close_up) * (0.75 if face_count <= 1.2 else 0.6)
    else:
        # A generic face detector missing an animated face is not evidence of a
        # wide shot. Preserve uncertainty until a domain-tuned model is present.
        close_up, wide = 0.25, 0.45
    conversation = _clamp(speech_probability * 0.38 + min(face_count / 2.0, 1.0) * 0.24 + mouth_activity * 0.28 + stillness * 0.10)
    reaction = _clamp((1.0 - speech_probability) * 0.38 + close_up * 0.32 + stillness * 0.18 + min(face_count, 1.0) * 0.12)
    listening = _clamp((1.0 - mouth_activity) * min(face_count, 1.0) * 0.45 + speech_probability * 0.2 + stillness * 0.2)
    action = _clamp(action_level * 0.72 + camera_motion * 0.18 + wide * 0.10)
    establishing = _clamp(wide * 0.55 + (1.0 - speech_probability) * 0.20 + (1.0 - min(face_count, 1.0)) * 0.25)
    speaker_cut = _clamp(speech_probability * close_up * 0.45 + mouth_activity * 0.35 + (1.0 if duration < 4.0 else 0.3) * 0.20)
    overall_confidence = _clamp(mean([
        face_confidence,
        _clamp(len(samples) / 3.0),
        mean(float(row.get("confidence", 0.5) or 0.5) for row in samples) if samples else 0.0,
    ]))
    intentions = _intent_probabilities(
        speech=speech_probability,
        conversation=conversation,
        reaction=reaction,
        listening=listening,
        action=action,
        establishing=establishing,
        close_up=close_up,
        wide=wide,
        stillness=stillness,
    )
    return {
        "shot_id": str(shot.get("id")),
        "start": round(start, 3),
        "end": round(end, 3),
        "duration": round(duration, 3),
        "visible_face_count": {"estimate": face_count, "confidence": round(face_confidence, 4)},
        "faces": _aggregate_faces(faces),
        "mouth_activity_probability": round(_clamp(mouth_activity), 4),
        "mouth_activity_confidence": round(face_confidence if faces else 0.0, 4),
        "eye_gaze": {"left": 0.0, "center": 0.0, "right": 0.0, "unknown": 1.0, "confidence": 0.0},
        "head_orientation": {"left": 0.0, "frontal": 0.0, "right": 0.0, "unknown": 1.0, "confidence": 0.0},
        "camera_motion_probability": round(_clamp(camera_motion), 4),
        "subject_motion_probability": round(_clamp(subject_motion), 4),
        "optical_motion_magnitude": round(_clamp(optical_motion), 4),
        "action_level": round(action_level, 4),
        "stillness_probability": round(stillness, 4),
        "conversation_probability": round(conversation, 4),
        "reaction_shot_probability": round(reaction, 4),
        "listening_shot_probability": round(listening, 4),
        "action_shot_probability": round(action, 4),
        "establishing_shot_probability": round(establishing, 4),
        "close_up_probability": round(close_up, 4),
        "wide_shot_probability": round(wide, 4),
        "speaker_cut_probability": round(speaker_cut, 4),
        "speech_overlap_probability": round(speech_probability, 4),
        "cinematic_intent": intentions,
        "overall_confidence": round(overall_confidence, 4),
        "sample_count": len(samples),
        "capability": "PROBABILISTIC_OBSERVATION" if samples else "CONSERVATIVE_FALLBACK",
        "face_detection_applicability": "generic_detector_low_domain_confidence",
    }


def extract_frame_evidence(*, media_path: Path, shots: list[dict[str, Any]], samples_per_shot: int = 3) -> dict[str, list[dict[str, Any]]]:
    import cv2  # type: ignore

    capture = cv2.VideoCapture(str(media_path))
    if not capture.isOpened():
        raise OSError(f"Could not open video for visual-performance analysis: {media_path}")
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(str(cascade_path))
    if detector.empty():
        capture.release()
        raise RuntimeError("OpenCV face detector data is unavailable.")
    evidence: dict[str, list[dict[str, Any]]] = {}
    try:
        for shot in shots:
            start, end = _bounds(shot)
            if end <= start:
                evidence[str(shot.get("id"))] = []
                continue
            times = [start + (end - start) * (index + 1) / (samples_per_shot + 1) for index in range(samples_per_shot)]
            frames = []
            for timestamp in times:
                capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
                ok, frame = capture.read()
                if ok and frame is not None:
                    frames.append((timestamp, frame))
            rows = []
            previous_gray = None
            previous_faces: list[dict[str, Any]] = []
            for timestamp, frame in frames:
                height, width = frame.shape[:2]
                scale = min(1.0, 640.0 / max(width, 1))
                small = cv2.resize(frame, None, fx=scale, fy=scale) if scale < 1.0 else frame
                gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                found = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24))
                faces = []
                for x, y, w, h in found:
                    normalized = {
                        "x": x / gray.shape[1], "y": y / gray.shape[0],
                        "width": w / gray.shape[1], "height": h / gray.shape[0],
                        "area_ratio": (w * h) / (gray.shape[0] * gray.shape[1]),
                        "mouth_movement": 0.0,
                    }
                    if previous_gray is not None:
                        prior = _nearest_face(normalized, previous_faces)
                        if prior is not None:
                            normalized["mouth_movement"] = _mouth_difference(gray, previous_gray, normalized, prior)
                    faces.append(normalized)
                optical = float(cv2.absdiff(gray, previous_gray).mean() / 255.0) if previous_gray is not None else 0.0
                camera = 0.0
                if previous_gray is not None:
                    shift, response = cv2.phaseCorrelate(previous_gray.astype("float32"), gray.astype("float32"))
                    camera = min(1.0, ((shift[0] ** 2 + shift[1] ** 2) ** 0.5) / max(gray.shape[:2])) * max(0.0, min(1.0, response))
                rows.append({
                    "time": round(timestamp, 3), "faces": faces,
                    "optical_motion": min(1.0, optical * 5.0),
                    "camera_motion": min(1.0, camera * 8.0),
                    "subject_motion": min(1.0, max(0.0, optical * 5.0 - camera * 4.0)),
                    "confidence": 0.65,
                })
                previous_gray, previous_faces = gray, faces
            evidence[str(shot.get("id"))] = rows
    finally:
        capture.release()
    return evidence


def _mouth_difference(current: Any, previous: Any, face: dict[str, Any], prior: dict[str, Any]) -> float:
    height, width = current.shape[:2]
    x = int(face["x"] * width)
    y = int((face["y"] + face["height"] * 0.55) * height)
    w = max(1, int(face["width"] * width))
    h = max(1, int(face["height"] * 0.4 * height))
    current_roi = current[y:min(height, y + h), x:min(width, x + w)]
    px = int(prior["x"] * width)
    py = int((prior["y"] + prior["height"] * 0.55) * height)
    previous_roi = previous[py:min(height, py + h), px:min(width, px + w)]
    if current_roi.size == 0 or previous_roi.size == 0 or current_roi.shape != previous_roi.shape:
        return 0.0
    import cv2  # type: ignore
    return min(1.0, float(cv2.absdiff(current_roi, previous_roi).mean() / 255.0) * 8.0)


def _nearest_face(face: dict[str, Any], prior: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not prior:
        return None
    return min(prior, key=lambda row: abs(float(row["x"]) - float(face["x"])) + abs(float(row["y"]) - float(face["y"])))


def _aggregate_faces(faces: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "location": {key: round(float(face.get(key, 0.0) or 0.0), 4) for key in ("x", "y", "width", "height")},
            "size_ratio": round(float(face.get("area_ratio", 0.0) or 0.0), 4),
            "mouth_movement_probability": round(_clamp(float(face.get("mouth_movement", 0.0) or 0.0)), 4),
            "confidence": 0.6,
        }
        for face in faces[:12]
    ]


def _intent_probabilities(**values: float) -> dict[str, float]:
    speech = values["speech"]
    return {
        "dialogue": round(_clamp(values["conversation"] * 0.7 + speech * 0.3), 4),
        "reaction": round(values["reaction"], 4),
        "listening": round(values["listening"], 4),
        "reveal": round(_clamp(values["close_up"] * 0.35 + (1.0 - values["stillness"]) * 0.25), 4),
        "threat": round(_clamp(values["action"] * 0.45 + values["close_up"] * 0.25), 4),
        "thinking": round(_clamp(values["close_up"] * values["stillness"] * (1.0 - speech)), 4),
        "watching": round(_clamp(values["listening"] * 0.7 + values["stillness"] * 0.2), 4),
        "walking": round(_clamp(values["action"] * 0.45 + values["wide"] * 0.25), 4),
        "action": round(values["action"], 4),
        "montage": round(_clamp(values["action"] * 0.4 + (1.0 - values["stillness"]) * 0.3), 4),
        "establishing": round(values["establishing"], 4),
        "insert": round(_clamp((1.0 - values["wide"]) * (1.0 - speech) * 0.6), 4),
        "close_up": round(values["close_up"], 4),
        "wide": round(values["wide"], 4),
        "transition": round(_clamp((1.0 - values["stillness"]) * (1.0 - speech) * 0.45), 4),
        "silence": round(_clamp(1.0 - speech), 4),
    }


def _bounds(row: dict[str, Any]) -> tuple[float, float]:
    start = float(row.get("start", 0.0) or 0.0)
    return start, float(row.get("end", start + float(row.get("duration", 0.0) or 0.0)) or start)


def _overlap(a: float, b: float, c: float, d: float) -> bool:
    return min(b, d) > max(a, c)


def _union_duration(intervals: list[tuple[float, float]]) -> float:
    valid = sorted((a, b) for a, b in intervals if b > a)
    if not valid:
        return 0.0
    merged = [list(valid[0])]
    for start, end in valid[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return sum(end - start for start, end in merged)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
