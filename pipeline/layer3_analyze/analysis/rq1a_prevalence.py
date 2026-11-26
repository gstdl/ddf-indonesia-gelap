"""Box (RQ1a): Cross-tab hoax x AI-generated; share by format; time series.

Output: distribution of AI-generated hoaxes. Columns: date (UTC day), media_type, hoax_label, ai_label, posts.
"""

import pandas as pd

KEYS = ["date", "media_type", "hoax_label", "ai_label"]


def run(merged: pd.DataFrame) -> pd.DataFrame:
    posts = merged.assign(date=merged["created_at"].dt.tz_convert("UTC").dt.date)
    return posts.groupby(KEYS, dropna=False).size().rename("posts").reset_index().sort_values(KEYS, ignore_index=True)
