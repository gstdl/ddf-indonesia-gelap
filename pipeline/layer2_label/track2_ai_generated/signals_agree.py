"""Box: Do the signals agree?

Output: one row per post. Columns: post_id, media_signal, reply_signal,
ai_label ("ai_generated" = Synthetic | "authentic" = Real | "uncertain" = Disagree, manual review).
"""

import pandas as pd


def decide_ai(signals: list[str | None]) -> str:
    present = [signal for signal in signals if isinstance(signal, str)]
    if present and all(signal == "synthetic" for signal in present):
        return "ai_generated"
    if present and all(signal == "real" for signal in present):
        return "authentic"
    return "uncertain"


def run(
    posts: pd.DataFrame,
    text_detector: pd.DataFrame,
    image_detector: pd.DataFrame,
    video_detector: pd.DataFrame,
    classify_replies: pd.DataFrame,
) -> pd.DataFrame:
    media_signals = pd.concat([text_detector, image_detector, video_detector], ignore_index=True)
    labels = (
        posts[["post_id"]]
        .merge(media_signals, on="post_id", how="left", validate="one_to_one")
        .merge(classify_replies, on="post_id", how="left", validate="one_to_one")
    )
    labels["ai_label"] = [
        decide_ai([media, reply]) for media, reply in zip(labels["media_signal"], labels["reply_signal"])
    ]
    return labels
