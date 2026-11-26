"""Box: One document per post (source fields kept).

Output: all posts from the three media boxes, sorted by post_id, plus
    document   `text` + "\n" + `media_text` when the post has media text, else `text`
"""

import pandas as pd


def build_document(posts: pd.DataFrame) -> pd.DataFrame:
    """Combine `text` and `media_text` into `document`. Keep every source field."""
    return posts.assign(
        document=[
            f"{text}\n{media_text}" if pd.notna(media_text) else text
            for text, media_text in zip(posts["text"], posts["media_text"])
        ]
    )


def run(text_only: pd.DataFrame, image: pd.DataFrame, video: pd.DataFrame) -> pd.DataFrame:
    posts = pd.concat([text_only, image, video], ignore_index=True)
    return build_document(posts.sort_values("post_id", key=lambda ids: ids.astype("int64"), ignore_index=True))
