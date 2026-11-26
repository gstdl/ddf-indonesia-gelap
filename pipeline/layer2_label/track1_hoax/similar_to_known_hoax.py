"""Box: Similar to a known hoax? Takes the "No" arrow of "Claimed false in any sources?".

Known hoaxes are the false-rated claims of the fact-check sources (same set as claimed_false). A post is
similar to a known hoax when its document is semantically close to one, looser than claimed_false.

Output: one row per post, in posts order. Columns: post_id, claimed_false, matched_fact_check_url,
similar_to_known_hoax (bool), hoax_label ("hoax" | "manual_review").
"""

import pandas as pd

from pipeline.layer2_label.track1_hoax.claimed_false import best_match, false_claims

# Minimum cosine similarity between a post document and a known hoax to call the post similar to it.
KNOWN_HOAX_SIMILARITY = 0.70


def decide_hoax(claimed_false: bool, similar_to_known_hoax: bool) -> str:
    if claimed_false or similar_to_known_hoax:
        return "hoax"
    return "manual_review"


def run(posts: pd.DataFrame, claimed_false: pd.DataFrame, *fact_checks: pd.DataFrame) -> pd.DataFrame:
    max_similarity, _ = best_match(posts, false_claims(fact_checks))
    similar = pd.DataFrame(
        {"post_id": posts["post_id"].to_numpy(), "similar_to_known_hoax": max_similarity >= KNOWN_HOAX_SIMILARITY}
    )
    labels = claimed_false.merge(similar, on="post_id", validate="one_to_one")
    labels["hoax_label"] = [
        decide_hoax(claimed, similar)
        for claimed, similar in zip(labels["claimed_false"], labels["similar_to_known_hoax"])
    ]
    return labels
