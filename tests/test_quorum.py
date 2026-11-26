import math

import pytest

from pipeline.layer2_label.track2_ai_generated.quorum import quorum
from pipeline.layer2_label.track2_ai_generated.signals_agree import decide_ai


@pytest.mark.parametrize(
    ("probabilities", "expected"),
    [
        ([0.9, 0.8, 0.1], "synthetic"),
        ([0.1, 0.2, 0.9], "real"),
        ([0.5, 0.5, 0.49], "synthetic"),  # threshold is inclusive
        ([0.9, 0.1, math.nan], "disagree"),  # one vote each, no quorum
        ([0.9, math.nan, math.nan], "disagree"),  # a single model cannot decide
        ([0.9, 0.8, math.nan], "synthetic"),
        ([math.nan, math.nan, math.nan], "disagree"),
    ],
)
def test_quorum(probabilities, expected):
    assert quorum(probabilities) == expected


def test_disagreeing_detector_makes_post_uncertain():
    assert decide_ai(["disagree", "synthetic"]) == "uncertain"
