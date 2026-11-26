"""Box (RQ1b, H1): Network metrics, communities, coordination; compare AI vs non-AI hoax cascades.

Output: map of spreaders and buzzer communities, one row per community_id. Columns: community_id, authors,
buzzer_authors, verified_authors (account_label verified), ai_hoax_posts (hoax and ai_generated), other_hoax_posts,
ai_hoax_cascade_size and other_hoax_cascade_size (mean cascade_size of those posts' cascades; NaN if none).
"""

import pandas as pd


def run(merged: pd.DataFrame, network_analysis: pd.DataFrame) -> pd.DataFrame:
    posts = merged.merge(network_analysis, on="post_id", how="left", validate="one_to_one")
    hoax = posts["hoax_label"] == "hoax"
    ai_hoax = hoax & (posts["ai_label"] == "ai_generated")
    posts = posts.assign(
        buzzer_author=posts["author"].where(posts["buzzer_label"] == "buzzer"),
        verified_author=posts["author"].where(posts["account_label"] == "verified"),
        ai_hoax=ai_hoax,
        other_hoax=hoax & ~ai_hoax,
        ai_hoax_cascade_size=posts["cascade_size"].where(ai_hoax),
        other_hoax_cascade_size=posts["cascade_size"].where(hoax & ~ai_hoax),
    )
    spreaders = posts.groupby("community_id").agg(
        authors=("author", "nunique"),
        buzzer_authors=("buzzer_author", "nunique"),
        verified_authors=("verified_author", "nunique"),
        ai_hoax_posts=("ai_hoax", "sum"),
        other_hoax_posts=("other_hoax", "sum"),
        ai_hoax_cascade_size=("ai_hoax_cascade_size", "mean"),
        other_hoax_cascade_size=("other_hoax_cascade_size", "mean"),
    )
    return spreaders.reset_index().sort_values("community_id", ignore_index=True)
