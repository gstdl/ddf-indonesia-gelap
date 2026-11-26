import math

import pytest

from pipeline.layer1_prepare.x_only import is_x_post


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://x.com/warga_andi/status/1900000000000000000", True),
        ("https://twitter.com/warga_andi/status/1900000000000000000", True),
        ("https://mobile.twitter.com/warga_andi/status/1900000000000000000", True),
        ("https://x.com/warga_andi", False),  # profile page, not a post
        ("https://news.detik.com/berita/d-7790001/aksi-indonesia-gelap", False),
        ("https://www.tiktok.com/@warga/video/7470000000000000000", False),
        ("https://notx.com/a/status/1", False),
        (math.nan, False),
    ],
)
def test_is_x_post(url, expected):
    assert is_x_post(url) == expected
