"""Box: Komdigi.

Every item of Komdigi's "Klarifikasi Hoaks" category, read from the JSON API behind komdigi.go.id:
GET https://web.komdigi.go.id/api/v1/contents/category/berita-hoaks?perPage=<n>&page=<p>
The API answers 403 "Unauthorized" unless the request carries the site's Origin header.
(The www.komdigi.go.id pages themselves return Akamai 403 from this machine; the API host does not.)

Output, one row per item, newest first:
    title         title as published, e.g. "[HOAKS] Pembagian Uang ke Mahasiswa Demonstran Indonesia Gelap"
    claim_text    title without its "[HOAKS]"-style label: the text the hoax checks match posts against
    explanation   Komdigi's one-paragraph explanation (API "excerpt")
    content       full article as plain text (API "body" with HTML tags removed)
    source_url    the item's page on www.komdigi.go.id, built from its slug
    published_at  publication time (Asia/Jakarta, as the API returns it)
    komdigi_id    Komdigi content id
There is deliberately no textual_rating column: the category holds only hoax clarifications, so
claimed_false counts every Komdigi claim as false (title labels carry typos such as "DISNFORMASI").
"""

import html
import re

import pandas as pd
import requests

API = "https://web.komdigi.go.id/api/v1/contents/category/berita-hoaks"
HEADERS = {"Origin": "https://www.komdigi.go.id", "Accept": "application/json"}
PER_PAGE = 1000  # the API accepts large pages; ~17k items take ~18 requests
# Public page of an item. The API's own short_url is empty for most items.
DETAIL_URL = "https://www.komdigi.go.id/berita/berita-hoaks/detail/{slug}"

LABEL = re.compile(r"^\s*\[[^\]]+\]\s*")
TAG = re.compile(r"<[^>]+>")


def plain_text(body: str | None) -> str | None:
    if body is None:
        return None
    return " ".join(html.unescape(TAG.sub(" ", body)).split())


def fetch_page(page: int) -> dict:
    response = requests.get(API, params={"perPage": PER_PAGE, "page": page}, headers=HEADERS, timeout=60)
    response.raise_for_status()
    return response.json()["response"]


def to_row(item: dict) -> dict:
    return {
        "title": item["title"],
        "claim_text": LABEL.sub("", item["title"]),
        "explanation": item["excerpt"],
        "content": plain_text(item["body"]),
        "source_url": DETAIL_URL.format(slug=item["slug"]),
        "published_at": item["published_at"],
        "komdigi_id": item["id"],
    }


def run() -> pd.DataFrame:
    first = fetch_page(1)
    items = first["data"]
    for page in range(2, first["last_page"] + 1):
        items.extend(fetch_page(page)["data"])
    claims = pd.DataFrame([to_row(item) for item in items])
    claims["published_at"] = pd.to_datetime(claims["published_at"]).dt.tz_localize("Asia/Jakarta")
    return claims.drop_duplicates("komdigi_id").sort_values(["published_at", "komdigi_id"], ascending=False, ignore_index=True)
