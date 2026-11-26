import math

import pandas as pd
import pytest

from pipeline.layer2_label.track1_hoax.claimed_false import false_claims, is_false_rating


@pytest.mark.parametrize(
    ("rating", "expected"),
    [
        ("Salah", True),
        ("Keliru", True),
        ("Menyesatkan", True),
        ("False dan misleading", True),
        ("Hasil Kecerdasan Buatan (AI)", True),
        ("Benar", False),
        ("Sebagian benar", False),
        ("Satire", False),
        ("Missing context", False),
        ("Belum ada bukti", False),
        (math.nan, False),
    ],
)
def test_is_false_rating(rating, expected):
    assert is_false_rating(rating) == expected


def test_source_without_rating_counts_every_claim_false():
    rated = pd.DataFrame(
        {"claim_text": ["a", "b"], "source_url": ["u1", "u2"], "textual_rating": ["Salah", "Benar"]}
    )
    hoaxes_only = pd.DataFrame({"claim_text": ["c"], "source_url": ["u3"]})
    assert false_claims((rated, hoaxes_only))["claim_text"].tolist() == ["a", "c"]
