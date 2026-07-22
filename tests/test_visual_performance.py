from cinelingus.visual_performance import describe_shot_performance


def test_visual_performance_describes_every_required_probability() -> None:
    row = describe_shot_performance(
        shot={"id": "shot_1", "start": 0.0, "end": 4.0, "duration": 4.0},
        samples=[
            {
                "faces": [{"x": 0.2, "y": 0.2, "width": 0.3, "height": 0.4, "area_ratio": 0.12, "mouth_movement": 0.8}],
                "optical_motion": 0.08, "camera_motion": 0.03, "subject_motion": 0.07, "confidence": 0.7,
            },
            {
                "faces": [{"x": 0.21, "y": 0.2, "width": 0.3, "height": 0.4, "area_ratio": 0.12, "mouth_movement": 0.7}],
                "optical_motion": 0.09, "camera_motion": 0.03, "subject_motion": 0.08, "confidence": 0.7,
            },
            {
                "faces": [{"x": 0.22, "y": 0.2, "width": 0.3, "height": 0.4, "area_ratio": 0.12, "mouth_movement": 0.75}],
                "optical_motion": 0.07, "camera_motion": 0.02, "subject_motion": 0.06, "confidence": 0.7,
            },
        ],
        speech_windows=[{"start": 0.5, "end": 3.5, "duration": 3.0}],
    )

    assert row["visible_face_count"]["estimate"] == 1.0
    assert row["mouth_activity_probability"] > 0.7
    assert row["conversation_probability"] > row["action_shot_probability"]
    assert row["cinematic_intent"]["dialogue"] > 0.5
    assert 0.0 <= row["overall_confidence"] <= 1.0
    assert all(0.0 <= value <= 1.0 for value in row["cinematic_intent"].values())


def test_visual_performance_fallback_is_explicitly_uncertain() -> None:
    row = describe_shot_performance(
        shot={"id": "shot_1", "start": 0.0, "end": 2.0, "duration": 2.0},
        samples=[],
        speech_windows=[],
    )

    assert row["capability"] == "CONSERVATIVE_FALLBACK"
    assert row["overall_confidence"] == 0.0
    assert row["eye_gaze"]["unknown"] == 1.0
    assert row["head_orientation"]["confidence"] == 0.0
