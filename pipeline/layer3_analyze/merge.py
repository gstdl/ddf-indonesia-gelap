"""Box: Merged dataset: posts, accounts, clusters, all labels.

Output: one row per post: layer 1 posts plus every label table given, joined on post_id.
"""

import pandas as pd


def run(posts: pd.DataFrame, *labels: pd.DataFrame) -> pd.DataFrame:
    merged = posts
    for label in labels:
        merged = merged.merge(label, on="post_id", how="left", validate="one_to_one")
    return merged
