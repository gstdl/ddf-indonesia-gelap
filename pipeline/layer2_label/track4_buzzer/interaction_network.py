"""Box: Build interaction network: reposts, replies, quotes, mentions.

Output: author-to-author edges, nodes are lowercase X handles (URLs and mentions only give handles).
Columns: source_author, target_author, interaction ("reply" | "repost" | "quote" | "mention"), weight (count).
Self-loops are dropped.
Open question: the export has no reposts (dropped as duplicates); find literature on detecting
coordination without them.
"""

import re

import pandas as pd

# Which URL column holds the parent post for each post_type.
PARENT_URL_COLUMN = {"reply": "reply_to_url", "repost": "retweet_of_url", "quote": "original_url"}

HANDLE_IN_URL = re.compile(r"x\.com/([A-Za-z0-9_]+)/status/")
MENTION = re.compile(r"@([A-Za-z0-9_]+)")


def edges_of(post: pd.Series) -> list[tuple[str, str, str]]:
    source = post["author"].lower()
    edges = []
    url_column = PARENT_URL_COLUMN.get(post["post_type"])
    if url_column is not None and pd.notna(post[url_column]):
        match = HANDLE_IN_URL.search(post[url_column])
        if match:
            edges.append((source, match.group(1).lower(), post["post_type"]))
    if pd.notna(post["mentioned_authors"]):
        edges += [(source, handle.lower(), "mention") for handle in MENTION.findall(post["mentioned_authors"])]
    return edges


def run(posts: pd.DataFrame) -> pd.DataFrame:
    rows = [edge for _, post in posts.iterrows() for edge in edges_of(post)]
    edges = pd.DataFrame(rows, columns=["source_author", "target_author", "interaction"])
    edges = edges[edges["source_author"] != edges["target_author"]]
    weighted = edges.groupby(["source_author", "target_author", "interaction"]).size().rename("weight")
    return weighted.reset_index().sort_values(["source_author", "target_author", "interaction"], ignore_index=True)
