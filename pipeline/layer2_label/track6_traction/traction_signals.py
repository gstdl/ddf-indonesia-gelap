"""Box: Engagement, reach, follower growth, monetized links.

Output: one row per post. Columns: post_id, engagement (likes + reposts + replies),
traction_score (engagement / max(reach, 1)), follower_growth (author's max - min author_followers in the dataset),
monetization_flag (an expanded URL's domain is in MONETIZATION_DOMAINS or a subdomain of one).
Open question: how to detect monetized links; the domain list covers Indonesian tipping and marketplace links only.
"""

import re
from urllib.parse import urlparse

import pandas as pd

# Tipping (saweria, trakteer, sociabuzz), link-in-bio shop (lynk) and marketplace (shopee, tokopedia) domains.
MONETIZATION_DOMAINS = ("saweria.co", "trakteer.id", "sociabuzz.com", "lynk.id", "shopee.co.id", "tokopedia.com", "tokopedia.link")


def is_monetized(expanded_urls: str | float) -> bool:
    if pd.isna(expanded_urls):
        return False
    for url in re.split(r"[,\s]+", expanded_urls.strip()):
        host = (urlparse(url).hostname or "").removeprefix("www.")
        if any(host == domain or host.endswith("." + domain) for domain in MONETIZATION_DOMAINS):
            return True
    return False


def run(posts: pd.DataFrame) -> pd.DataFrame:
    followers = posts.groupby("author")["author_followers"]
    signals = posts[["post_id"]].copy()
    signals["engagement"] = posts["likes"] + posts["reposts"] + posts["replies"]
    signals["traction_score"] = signals["engagement"] / posts["reach"].clip(lower=1)
    signals["follower_growth"] = followers.transform("max") - followers.transform("min")
    signals["monetization_flag"] = posts["expanded_urls"].map(is_monetized).astype(bool)
    return signals.sort_values("post_id", ignore_index=True)
