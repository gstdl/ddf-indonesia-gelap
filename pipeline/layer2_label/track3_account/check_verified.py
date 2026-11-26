"""Box: Check verified status of the author.

Output: one row per post. Columns: post_id, account_label ("verified" | "unverified", from Brandwatch's author_verified).
Named account_label because the layer 1 posts already carry author_verified and merge joins both tables.
"""

import pandas as pd


def run(posts: pd.DataFrame) -> pd.DataFrame:
    labels = posts[["post_id"]].copy()
    labels["account_label"] = posts["author_verified"].map({True: "verified", False: "unverified"})
    return labels
