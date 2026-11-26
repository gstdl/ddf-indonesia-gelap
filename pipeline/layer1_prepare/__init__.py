"""Layer 1: collect and prepare posts. Brandwatch export -> one linked document per post.

Output of the last box (link), one row per post, read by every layer 2 track:
    post_id            str   X status id
    url                str
    author             str   X handle, without "@"
    author_id          str
    created_at         datetime (UTC)
    full_text          str   raw post text
    text               str   normalized post text
    language           str   Brandwatch language code
    media_type         str   "text" | "image" | "video"
    media_url          str | NaN   first media URL of the post
    media_path         str | NaN   media file, relative to data/
    keyframe_paths     list[str] | NaN   video keyframe images, relative to data/
    audio_path         str | NaN   video audio track (16 kHz mono wav), relative to data/
    media_ocr          str | NaN
    media_caption      str | NaN
    media_transcript   str | NaN   speech-to-text of the video audio
    media_text         str | NaN   OCR, caption and transcript joined
    phash              str | NaN   perceptual hash of the image or middle video keyframe
    document           str   text and media_text in one field
    cluster_id         str   near-duplicate cluster; every post is kept
    post_type          str   "original" | "reply" | "repost" | "quote"
    parent_post_id     str | NaN   post this one replies to, reposts or quotes
    engagement_type, reply_to_url, retweet_of_url, original_url, mentioned_authors, expanded_urls,
    author_verified, author_followers, author_following, author_post_count,
    likes, reposts, replies, impressions, reach   renamed Brandwatch fields (see export.SOURCE_COLUMNS)
    plus every other Brandwatch column under its original name
"""
