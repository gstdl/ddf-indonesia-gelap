"""Box: Claimed false in any sources?

A post is claimed false when its document is a near paraphrase of a claim that a fact-check source
rated false. Fact-check sources are variadic, so main.py can comment one out.

Output: one row per post, in posts order. Columns: post_id, claimed_false (bool),
matched_fact_check_url (source_url of the best-matching false claim when claimed_false, else NaN).
"""

import numpy as np
import pandas as pd

from pipeline.hf import embed

# Minimum cosine similarity between a post document and a false-rated claim to call the post claimed false.
CLAIMED_FALSE_SIMILARITY = 0.85
# Case-insensitive substrings of textual_rating that mark a claim false: false, misleading, fabricated or
# manipulated (incl. partly false and AI-generated). Satire, missing context and unproven are NOT false.
FALSE_RATING_TERMS = [
    "salah",
    "keliru",
    "sesat",
    "menyesatkan",
    "tidak benar",
    "hoaks",
    "hoax",
    "disinformasi",
    "misinformasi",
    "palsu",
    "editan",
    "kecerdasan buatan",
    "false",
    "misleading",
    "fabricated",
    "manipulated",
    "altered",
]


def is_false_rating(textual_rating: str) -> bool:
    if pd.isna(textual_rating):
        return False
    rating = textual_rating.casefold()
    return any(term in rating for term in FALSE_RATING_TERMS)


def false_claims(fact_checks: tuple[pd.DataFrame, ...]) -> pd.DataFrame:
    """Columns: claim_text, source_url. Every false-rated claim of every source, deduplicated and sorted."""
    parts = []
    for source in fact_checks:
        if "textual_rating" in source.columns:
            source = source[source["textual_rating"].map(is_false_rating)]
        # A source without textual_rating (Komdigi, CekFakta) publishes only hoaxes: every claim is false.
        parts.append(source[["claim_text", "source_url"]])
    claims = pd.concat(parts, ignore_index=True).dropna(subset=["claim_text"]).drop_duplicates()
    return claims.sort_values(["claim_text", "source_url"], kind="stable").reset_index(drop=True)


def best_match(posts: pd.DataFrame, claims: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Per post: highest cosine similarity to any claim, and that claim's source_url."""
    similarity = embed(posts["document"].tolist()) @ embed(claims["claim_text"].tolist()).T
    best = similarity.argmax(axis=1)
    return similarity.max(axis=1), claims["source_url"].to_numpy()[best]


def run(posts: pd.DataFrame, *fact_checks: pd.DataFrame) -> pd.DataFrame:
    max_similarity, best_url = best_match(posts, false_claims(fact_checks))
    claimed = max_similarity >= CLAIMED_FALSE_SIMILARITY
    return pd.DataFrame(
        {
            "post_id": posts["post_id"].to_numpy(),
            "claimed_false": claimed,
            "matched_fact_check_url": pd.Series(best_url, dtype="str").where(claimed),
        }
    )
