import math

import pytest

from pipeline.layer2_label.track2_ai_generated.classify_replies import decide_reply_signal


@pytest.mark.parametrize(
    ("authors", "stances", "expected"),
    [
        # The latest Grok stance wins over earlier Grok stances and over the community majority.
        (["grok", "warga_a", "warga_b", "grok"], ["synthetic", "real", "real", "real"], ("real", "grok")),
        (["warga_a", "warga_b", "grok"], ["real", "real", "synthetic"], ("synthetic", "grok")),
        # A Grok reply without a stance leaves the decision to the community.
        (["grok", "warga_a", "warga_b"], [None, "real", math.nan], ("real", "community")),
        (["warga_a", "warga_b", "warga_c"], ["synthetic", "synthetic", "real"], ("synthetic", "community")),
        # A community tie, or no stance at all, gives no signal.
        (["warga_a", "warga_b"], ["synthetic", "real"], (None, None)),
        (["warga_a", "grok"], [None, math.nan], (None, None)),
    ],
)
def test_decide_reply_signal(authors, stances, expected):
    assert decide_reply_signal(authors, stances) == expected
