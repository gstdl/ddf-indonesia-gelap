"""Box: CekFakta.

Output: CekFakta fact-checks. Columns: claim_text, source_url, published_at (+ textual_rating if published;
without it every claim counts as false, since CekFakta publishes only hoaxes).

Not built: from this machine cekfakta.com and old.cekfakta.com time out, so the scraper can neither be
written against real pages nor verified. Commented out in main.py until reachable.
"""

import pandas as pd


def run() -> pd.DataFrame:
    raise NotImplementedError("Box: CekFakta (cekfakta.com and old.cekfakta.com time out)")
