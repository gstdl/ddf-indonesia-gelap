import math

import pandas as pd
import pytest

from pipeline.layer2_label.track1_hoax.similar_to_known_hoax import decide_hoax
from pipeline.layer2_label.track2_ai_generated.signals_agree import decide_ai
from pipeline.layer3_analyze import merge


@pytest.mark.parametrize(
    ("claimed_false", "similar", "expected"),
    [
        (True, False, "hoax"),
        (False, True, "hoax"),
        (False, False, "manual_review"),
    ],
)
def test_decide_hoax(claimed_false, similar, expected):
    assert decide_hoax(claimed_false, similar) == expected


@pytest.mark.parametrize(
    ("signals", "expected"),
    [
        (["synthetic", "synthetic"], "ai_generated"),
        (["real", "real"], "authentic"),
        (["synthetic", "real"], "uncertain"),
        (["synthetic", math.nan], "ai_generated"),
        ([None, math.nan], "uncertain"),
    ],
)
def test_decide_ai(signals, expected):
    assert decide_ai(signals) == expected


def test_merge_keeps_posts_without_labels():
    posts = pd.DataFrame({"post_id": ["a", "b"]})
    hoax = pd.DataFrame({"post_id": ["a"], "hoax_label": ["hoax"]})
    merged = merge.run(posts, hoax)
    assert merged["post_id"].tolist() == ["a", "b"]
    assert merged["hoax_label"].isna().tolist() == [False, True]


def test_merge_rejects_duplicate_label_rows():
    posts = pd.DataFrame({"post_id": ["a"]})
    hoax = pd.DataFrame({"post_id": ["a", "a"], "hoax_label": ["hoax", "manual_review"]})
    with pytest.raises(pd.errors.MergeError):
        merge.run(posts, hoax)
