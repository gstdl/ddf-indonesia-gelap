"""Box: Export posts, media, replies, authors (from Brandwatch: #IndonesiaGelap posts) — the CSV part.

Reads the Brandwatch CSV export at `path` (passed by main.py from the command line) as is, every
content source included (x_only keeps the X posts).
Output: the columns in SOURCE_COLUMNS (renamed), `created_at` parsed to UTC.
Every other Brandwatch column is kept under its original name.
"""

from pathlib import Path

import pandas as pd

# The Brandwatch "Bulk Mentions Download" starts with 5 report-metadata rows (Report, Brand, From, To, Label)
# and a blank line; pandas skips blank lines when counting, so the column header is row 5.
HEADER_ROW = 5

SOURCE_COLUMNS = {
    "Url": "url",
    "Date": "created_at",
    "Author": "author",
    "X Author ID": "author_id",
    "Full Text": "full_text",
    "Language": "language",
    "Subtype": "brandwatch_media_type",
    "Media URLs": "media_urls",
    "Engagement Type": "engagement_type",
    "X Reply to": "reply_to_url",
    "X Repost of": "retweet_of_url",
    "Original Url": "original_url",
    "Mentioned Authors": "mentioned_authors",
    "Expanded URLs": "expanded_urls",
    "X Verified": "author_verified",
    "X Followers": "author_followers",
    "X Following": "author_following",
    "X Posts": "author_post_count",
    "X Likes": "likes",
    "X Reposts": "reposts",
    "X Replies": "replies",
    "Impressions": "impressions",
    "Reach (new)": "reach",
}


def run(path: Path) -> pd.DataFrame:
    posts = pd.read_csv(path, header=HEADER_ROW, dtype={"X Author ID": str}).rename(
        columns=SOURCE_COLUMNS
    )
    posts["created_at"] = pd.to_datetime(posts["created_at"], utc=True)
    return posts
