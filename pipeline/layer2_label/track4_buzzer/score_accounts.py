"""Box: Score accounts: burst posting, shared content, account age, follower ratio.

Output: one row per post, features are of the post's author.
Columns: post_id, buzzer_label ("buzzer" | "organic" | "uncertain"), burst, shared_content, follower_ratio.
    burst           max posts by the author within any WINDOW
    shared_content  author's posts in a near-duplicate cluster posted by >= SHARED_MIN_AUTHORS authors within WINDOW,
                    not counting the cluster's first post (the source others copy or repost)
    follower_ratio  latest author_followers / max(latest author_following, 1)
Account age is not in the Brandwatch export, so it is not scored.
The interaction network adds none of these features, so this box reads only the posts.
"""

import pandas as pd

WINDOW = pd.Timedelta(minutes=60)  # time window for burst posting and shared content
SHARED_MIN_AUTHORS = 3  # distinct authors posting one cluster within WINDOW = coordinated content
BURST_FLAG = 5  # burst >= this is a flag
SHARED_CONTENT_FLAG = 2  # shared_content >= this is a flag
FOLLOWER_RATIO_FLAG = 0.1  # follower_ratio < this is a flag
BUZZER_MIN_FLAGS = 2  # flags >= this is a buzzer


def buzzer_label(burst: int, shared_content: int, follower_ratio: float) -> str:
    flags = (burst >= BURST_FLAG) + (shared_content >= SHARED_CONTENT_FLAG) + (follower_ratio < FOLLOWER_RATIO_FLAG)
    if flags >= BUZZER_MIN_FLAGS:
        return "buzzer"
    if flags == 0:
        return "organic"
    return "uncertain"


def max_in_window(times: pd.Series) -> int:
    times = times.sort_values()
    return max(int(((times >= start) & (times <= start + WINDOW)).sum()) for start in times)


def is_shared(post: pd.Series, cluster: pd.DataFrame) -> bool:
    """True if some WINDOW containing the post holds >= SHARED_MIN_AUTHORS authors of its cluster."""
    starts = cluster["created_at"][(cluster["created_at"] <= post["created_at"]) & (cluster["created_at"] >= post["created_at"] - WINDOW)]
    for start in starts:
        in_window = cluster[(cluster["created_at"] >= start) & (cluster["created_at"] <= start + WINDOW)]
        if in_window["author"].nunique() >= SHARED_MIN_AUTHORS:
            return True
    return False


def run(posts: pd.DataFrame) -> pd.DataFrame:
    posts = posts.sort_values(["created_at", "post_id"])
    clusters = {cluster_id: cluster for cluster_id, cluster in posts.groupby("cluster_id")}
    shared = posts.apply(lambda post: is_shared(post, clusters[post["cluster_id"]]), axis=1)
    shared &= posts.duplicated("cluster_id")  # sorted by time: the first post of each cluster is the source
    latest = posts.groupby("author").last()
    accounts = pd.DataFrame(
        {
            "burst": posts.groupby("author")["created_at"].apply(max_in_window),
            "shared_content": shared.groupby(posts["author"]).sum().astype(int),
            "follower_ratio": latest["author_followers"] / latest["author_following"].clip(lower=1),
        }
    )
    accounts["buzzer_label"] = [buzzer_label(*row) for row in accounts[["burst", "shared_content", "follower_ratio"]].itertuples(index=False)]
    labels = posts[["post_id", "author"]].merge(accounts, left_on="author", right_index=True, how="left", validate="many_to_one")
    return labels[["post_id", "buzzer_label", "burst", "shared_content", "follower_ratio"]].sort_values("post_id", ignore_index=True)
