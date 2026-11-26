"""Box: Collect replies.

Output: reply posts, with `parent_post_id` pointing at the post they reply to.
"""

import pandas as pd


def run(posts: pd.DataFrame) -> pd.DataFrame:
    return posts[posts["post_type"] == "reply"]
