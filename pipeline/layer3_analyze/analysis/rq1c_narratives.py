"""Box (RQ1c, H2): Narrative distribution; compare traction and monetization of AI vs non-AI hoax accounts.

Output: narrative typology and evidence on traction-seeking, one row per narrative_theme x ai_hoax
(hoax and ai_generated). Columns: narrative_theme, ai_hoax, posts, authors, traction_score (mean),
monetization_share (share of posts with monetization_flag), buzzer_author_share (share of authors labelled buzzer).
"""

import pandas as pd


def run(merged: pd.DataFrame) -> pd.DataFrame:
    posts = merged.assign(
        ai_hoax=(merged["hoax_label"] == "hoax") & (merged["ai_label"] == "ai_generated"),
        buzzer_author=merged["author"].where(merged["buzzer_label"] == "buzzer"),
        monetization_flag=merged["monetization_flag"].astype(bool),
    )
    narratives = posts.groupby(["narrative_theme", "ai_hoax"]).agg(
        posts=("post_id", "size"),
        authors=("author", "nunique"),
        traction_score=("traction_score", "mean"),
        monetization_share=("monetization_flag", "mean"),
        buzzer_authors=("buzzer_author", "nunique"),
    )
    narratives["buzzer_author_share"] = narratives.pop("buzzer_authors") / narratives["authors"]
    return narratives.reset_index().sort_values(["narrative_theme", "ai_hoax"], ignore_index=True)
