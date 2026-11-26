"""Box: Link replies, reposts, quotes to parents.

Output: the clustered posts plus
    post_type        "reply" | "repost" | "quote" from Brandwatch `engagement_type`; "original" when it is missing
    parent_post_id   replies: status id in `reply_to_url`; reposts: in `retweet_of_url`; quotes: the first X status
                     link in `expanded_urls` (Brandwatch's `original_url` is the quote's own URL in the real export).
                     NaN for originals, for quotes without such a link and for the few replies whose `reply_to_url`
                     Brandwatch left empty. A parent not in the dataset keeps its id.
Input for every layer 2 track.
"""

import pandas as pd

from pipeline.layer1_prepare.x_only import STATUS_ID, status_id

# Brandwatch engagement_type -> post_type. A missing engagement_type is an original post.
POST_TYPES = {"REPLY": "reply", "RETWEET": "repost", "QUOTE": "quote"}
# post_type -> the Brandwatch column holding the parent post URL.
PARENT_URL_COLUMNS = {"reply": "reply_to_url", "repost": "retweet_of_url"}


def post_type(engagement_type: str) -> str:
    return POST_TYPES[engagement_type] if pd.notna(engagement_type) else "original"


def quoted_post_id(expanded_urls: str) -> str | None:
    match = STATUS_ID.search(expanded_urls) if pd.notna(expanded_urls) else None
    return match.group(1) if match else None


def parent_post_id(post: pd.Series) -> str | None:
    if post["post_type"] == "original":
        return None
    if post["post_type"] == "quote":
        return quoted_post_id(post["expanded_urls"])
    parent_url = post[PARENT_URL_COLUMNS[post["post_type"]]]
    return status_id(parent_url) if pd.notna(parent_url) else None


def run(posts: pd.DataFrame) -> pd.DataFrame:
    posts = posts.assign(post_type=[post_type(value) for value in posts["engagement_type"]])
    return posts.assign(parent_post_id=[parent_post_id(post) for _, post in posts.iterrows()])
