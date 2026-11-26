"""Box: Text only: no extraction. Takes the "Text" arrow of "Media type?".

Output: text posts (media_type == "text") with the media columns of the image and video boxes set to NaN:
`keyframe_paths`, `audio_path`, `media_ocr`, `media_caption`, `media_transcript`, `media_text`, `phash`.
"""

import pandas as pd


def run(posts: pd.DataFrame) -> pd.DataFrame:
    return posts[posts["media_type"] == "text"].assign(
        phash=None,
        media_ocr=None,
        media_caption=None,
        media_transcript=None,
        media_text=None,
        keyframe_paths=None,
        audio_path=None,
    )
