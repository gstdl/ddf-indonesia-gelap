import pandas as pd
import pytest

from pipeline.layer2_label.track4_buzzer.score_accounts import buzzer_label
from pipeline.layer2_label.track6_traction.traction_signals import is_monetized
from pipeline.layer4_publish import anonymize


@pytest.mark.parametrize(
    ("burst", "shared_content", "follower_ratio", "expected"),
    [
        (5, 2, 1.0, "buzzer"),
        (1, 2, 0.05, "buzzer"),
        (5, 0, 0.05, "buzzer"),
        (5, 0, 1.0, "uncertain"),
        (1, 2, 1.0, "uncertain"),
        (1, 0, 0.09, "uncertain"),
        (4, 1, 0.1, "organic"),
    ],
)
def test_buzzer_label(burst, shared_content, follower_ratio, expected):
    assert buzzer_label(burst, shared_content, follower_ratio) == expected


@pytest.mark.parametrize(
    ("expanded_urls", "expected"),
    [
        ("https://saweria.co/dukung", True),
        ("https://example.com/a, https://www.trakteer.id/x", True),
        ("https://s.shopee.co.id/abc", True),
        ("https://notsaweria.co/x", False),
        ("https://kompas.com/berita", False),
        (float("nan"), False),
    ],
)
def test_is_monetized(expanded_urls, expected):
    assert is_monetized(expanded_urls) == expected


def test_anonymize(monkeypatch):
    monkeypatch.setenv("ANONYMIZATION_SALT", "fixed")
    table = pd.DataFrame(
        {"author": ["warga_yoga", None], "community_id": ["grok", "grok"], "url": ["u", "u"], "posts": [1, 2]}
    )
    result = anonymize.run(table)
    assert list(result.columns) == ["author", "community_id", "posts"]
    assert result.loc[0, "author"] == anonymize.pseudonym("warga_yoga", "fixed")
    assert len(result.loc[0, "author"]) == 16
    assert pd.isna(result.loc[1, "author"])
    assert "grok" not in result.to_string()
