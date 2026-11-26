import math

import numpy as np
import pandas as pd

from pipeline.layer1_prepare.cluster import cluster_ids, near_duplicate_pairs
from pipeline.layer1_prepare.document import build_document
from pipeline.layer1_prepare.link import parent_post_id, post_type
from pipeline.layer1_prepare.normalize import normalize_text

# "gelap" is a fake entry: it must change the bare word but never the hashtag; placeholders must never change.
SLANG = {"gak": "tidak", "yg": "yang", "gelap": "suram", "user": "WRONG", "httpurl": "WRONG"}


def test_normalize_text_replaces_urls_mentions_and_slang_but_keeps_hashtags_and_casing():
    text = "@budi_1 Gak percaya yg ini #IndonesiaGelap #gelap Cek https://t.co/abc www.x.com/a saweria.co/nkri Gelap"
    assert normalize_text(text, SLANG) == (
        "@USER tidak percaya yang ini #IndonesiaGelap #gelap Cek HTTPURL HTTPURL HTTPURL suram"
    )


def test_build_document_appends_media_text_only_when_present():
    posts = pd.DataFrame({"text": ["teks", "teks"], "media_text": ["ocr", math.nan]})
    assert build_document(posts)["document"].tolist() == ["teks\nocr", "teks"]


def test_near_duplicates_by_cosine_or_phash():
    embeddings = np.array([[1.0, 0.0], [0.95, math.sqrt(1 - 0.95**2)], [0.0, 1.0], [0.0, -1.0]])
    phashes = [None, None, "ffffffffffffffff", "ffffffffffffff00"]  # 8 bits apart: near-duplicate
    assert near_duplicate_pairs(embeddings, phashes) == [(0, 1), (2, 3)]
    assert near_duplicate_pairs(embeddings, [None, None, "ffffffffffffffff", "fffffffffffffe00"]) == [(0, 1)]  # 9 bits


def test_cluster_id_is_smallest_post_id_of_connected_component():
    post_ids = ["30", "100", "9", "7"]
    assert cluster_ids(post_ids, [(0, 1), (1, 2)]) == ["9", "9", "9", "7"]


def test_post_type_and_parent():
    assert post_type(math.nan) == "original"
    assert post_type("RETWEET") == "repost"
    quote = pd.Series({"post_type": "quote", "url": "http://twitter.com/b/statuses/9",
                       "expanded_urls": "https://x.com/a/status/123/photo/1"})
    assert parent_post_id(quote) == "123"
    assert parent_post_id(pd.Series({"post_type": "quote", "expanded_urls": "https://saweria.co/x"})) is None
    reply = pd.Series({"post_type": "reply", "reply_to_url": "http://twitter.com/kumparan/statuses/1984231535952969952"})
    assert parent_post_id(reply) == "1984231535952969952"
    assert parent_post_id(pd.Series({"post_type": "original"})) is None
