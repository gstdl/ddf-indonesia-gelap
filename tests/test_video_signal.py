import pytest

from pipeline.layer2_label.track2_ai_generated.video_detector import video_signal


@pytest.mark.parametrize(
    ("frame_signal", "audio_signal", "has_audio", "expected"),
    [
        # Either track being synthetic makes the video synthetic (deepfake audio over real footage).
        ("synthetic", "real", True, "synthetic"),
        ("real", "synthetic", True, "synthetic"),
        ("disagree", "synthetic", True, "synthetic"),
        ("synthetic", "disagree", False, "synthetic"),
        # Real needs real frames and real audio, or real frames and no audio track.
        ("real", "real", True, "real"),
        ("real", "disagree", False, "real"),
        # Audio models disagreeing on an existing track is not the same as a silent video.
        ("real", "disagree", True, "disagree"),
        ("disagree", "real", True, "disagree"),
        ("disagree", "disagree", False, "disagree"),
    ],
)
def test_video_signal(frame_signal, audio_signal, has_audio, expected):
    assert video_signal(frame_signal, audio_signal, has_audio) == expected
