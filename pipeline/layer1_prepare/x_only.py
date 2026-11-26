"""Box (not on the board): Keep X / Twitter posts only.

A Brandwatch query also returns news, Facebook, TikTok, YouTube and other sources. A row is an X post
when its `url` is an x.com or twitter.com status link: https://x.com/<handle>/status/<id>, or
http://twitter.com/<handle>/statuses/<id> as the real Brandwatch export writes it.
Output: the X rows plus `post_id` (the status id). The kept and dropped counts are logged per
"Content Source Name".
"""

import logging
import re
from urllib.parse import urlparse

import pandas as pd

log = logging.getLogger(__name__)

X_HOSTS = {"x.com", "twitter.com", "www.x.com", "www.twitter.com", "mobile.x.com", "mobile.twitter.com"}

STATUS_ID = re.compile(r"/status(?:es)?/(\d+)")


def status_id(url: str) -> str:
    return STATUS_ID.search(url).group(1)


def is_x_post(url: str) -> bool:
    return pd.notna(url) and urlparse(url).hostname in X_HOSTS and STATUS_ID.search(url) is not None


def run(posts: pd.DataFrame) -> pd.DataFrame:
    keep = posts["url"].map(is_x_post)
    counts = posts.groupby([posts["Content Source Name"].fillna("(empty)"), keep.rename("kept")]).size()
    log.info("x_only kept %d of %d rows; rows per source and kept: %s", keep.sum(), len(posts), counts.to_dict())
    x_posts = posts[keep]
    return pd.concat([x_posts, x_posts["url"].map(status_id).rename("post_id")], axis=1).reset_index(drop=True)
