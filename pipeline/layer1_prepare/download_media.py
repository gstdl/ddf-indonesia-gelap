"""Box: Export posts, media, replies, authors — the media part. Downloads each X post's first media file.

Output: the X posts plus media_type ("text" | "image" | "video"), media_url (first URL in "Media URLs"),
media_path (file under data/raw/media/, relative to data/). A re-run reuses files already downloaded.

media_type follows Brandwatch "Subtype" when the media can be analysed. A photo/video post becomes "text"
(its text is all there is to analyse) when the export lists no media URL (Brandwatch writes "null#gif"
for GIFs, which counts as none), or the URL answers 403/404/410
(pbs.twimg.com answers 403 for media removed or withheld on X). Both counts are logged; any other
download error stops the run.
"""

import logging
import os

import pandas as pd
import requests

from pipeline.step import DATA, progress

log = logging.getLogger(__name__)

MEDIA_DIR = DATA / "raw" / "media"

# Brandwatch "Subtype" (the UI's media type for X posts) -> pipeline media_type. Anything else
# (link, status, other, empty) is "text". The export's "Media Type" column is a different, broadcast field.
MEDIA_TYPES = {"photo": "image", "video": "video"}
# HTTP statuses meaning the media is no longer available on X (pbs.twimg.com uses 403 for removed media).
GONE_STATUSES = {403, 404, 410}


def first_media_url(media_urls: str) -> str | None:
    """First real URL in Brandwatch's comma-separated "Media URLs"; placeholders like "null#gif" are skipped."""
    if pd.isna(media_urls):
        return None
    return next((url.strip() for url in media_urls.split(",") if url.strip().startswith("http")), None)


def download(url: str, post_id: str) -> str | None:
    """Save the media file once; a re-run reuses the saved file. None when X no longer has the media."""
    path = MEDIA_DIR / f"{post_id}{os.path.splitext(url.split('?')[0])[1]}"
    if not path.exists():
        response = requests.get(url, timeout=60)
        if response.status_code in GONE_STATUSES:
            return None
        response.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(path.name + ".tmp")
        tmp_path.write_bytes(response.content)
        os.replace(tmp_path, path)
    return str(path.relative_to(DATA))


def run(posts: pd.DataFrame) -> pd.DataFrame:
    listed_type = posts["brandwatch_media_type"].map(MEDIA_TYPES).fillna("text")
    media_url = [
        first_media_url(urls) if kind != "text" else None for urls, kind in zip(posts["media_urls"], listed_type)
    ]
    media_path = [
        download(url, post_id) if url is not None else None
        for url, post_id in progress(list(zip(media_url, posts["post_id"])), "download media")
    ]
    media_type = listed_type.where(pd.Series(media_path, index=posts.index).notna(), "text")
    no_url = sum(kind != "text" and url is None for kind, url in zip(listed_type, media_url))
    gone = sum(url is not None and path is None for url, path in zip(media_url, media_path))
    log.info("download_media: %d media posts, %d without a media URL, %d deleted on X (both kept as text)",
             (listed_type != "text").sum(), no_url, gone)
    derived = pd.DataFrame({"media_type": media_type, "media_url": media_url, "media_path": media_path}, index=posts.index)
    # One concat instead of column-by-column inserts: the export has ~180 columns.
    return pd.concat([posts, derived], axis=1)
