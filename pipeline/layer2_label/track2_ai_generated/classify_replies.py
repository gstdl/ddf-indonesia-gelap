"""Box: Classify replies: community, Grok / AI.

Each reply gets a stance on whether its parent post (or the parent's media) is AI-generated, by zero-shot NLI
over the reply `text` against the three Indonesian hypotheses in HYPOTHESES. The top hypothesis is the stance
when its score >= STANCE_THRESHOLD; otherwise, or when the top hypothesis is "none", the reply takes no stance.
Per parent post (decide_reply_signal): the latest Grok reply with a stance decides; without one, the majority
stance of community replies decides (a tie gives no signal). Parent posts without a signal get no row.

Output: one row per parent post with a reply signal. Columns:
    post_id         the parent post (the replies' parent_post_id)
    reply_signal    "synthetic" | "real"
    reply_source    "grok" | "community"
Open question: replies mostly discuss chronology and opinions; Grok replies may be the more reliable signal.
"""

from collections import Counter
from functools import cache

import pandas as pd
from transformers import pipeline

from pipeline.hf import DEVICE

# Multilingual (incl. Indonesian) NLI model for zero-shot classification.
NLI_MODEL = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
NLI_MODEL_REVISION = "b5113eb38ab63efdd7f280f8c144ea8b13f978ce"
# Stance -> hypothesis tested against each reply; "none" means the reply says nothing about authenticity.
HYPOTHESES = {
    "synthetic": "Media ini buatan AI atau hasil manipulasi digital.",
    "real": "Media ini asli, bukan buatan AI atau hasil manipulasi digital.",
    "none": "Balasan ini tidak membahas apakah media ini asli atau buatan AI.",
}
# The top hypothesis counts as the reply's stance only from this score up.
STANCE_THRESHOLD = 0.5
# X handle of the Grok AI assistant; every other author is community.
GROK_AUTHOR = "grok"


@cache
def nli():
    return pipeline("zero-shot-classification", model=NLI_MODEL, revision=NLI_MODEL_REVISION, device=DEVICE)


def stance(text: str) -> str | None:
    """"synthetic" | "real", or None when the reply takes no stance."""
    result = nli()(text, candidate_labels=list(HYPOTHESES.values()), hypothesis_template="{}")
    label = next(name for name, hypothesis in HYPOTHESES.items() if hypothesis == result["labels"][0])
    return label if label != "none" and result["scores"][0] >= STANCE_THRESHOLD else None


def decide_reply_signal(authors: list[str], stances: list[str | None]) -> tuple[str | None, str | None]:
    """(reply_signal, reply_source) of one parent post's replies in chronological order; a None/NaN stance = none."""
    grok = [s for a, s in zip(authors, stances) if a == GROK_AUTHOR and isinstance(s, str)]
    if grok:
        return grok[-1], "grok"
    community = Counter(s for a, s in zip(authors, stances) if a != GROK_AUTHOR and isinstance(s, str))
    if community["synthetic"] > community["real"]:
        return "synthetic", "community"
    if community["real"] > community["synthetic"]:
        return "real", "community"
    return None, None


def run(replies: pd.DataFrame) -> pd.DataFrame:
    replies = replies.sort_values(["parent_post_id", "created_at", "post_id"]).assign(
        stance=lambda table: [stance(text) for text in table["text"]]
    )
    rows = []
    for post_id, group in replies.groupby("parent_post_id", sort=True):
        signal, source = decide_reply_signal(group["author"].to_list(), group["stance"].to_list())
        if signal is not None:
            rows.append({"post_id": post_id, "reply_signal": signal, "reply_source": source})
    return pd.DataFrame(rows, columns=["post_id", "reply_signal", "reply_source"])
