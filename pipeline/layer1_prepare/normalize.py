"""Box: Normalize text: URLs, mentions, slang, language.

Output: the exported posts with `text` normalized from `full_text` (IndoBERTweet convention):
URLs (incl. bare domains) -> "HTTPURL", @mentions -> "@USER", Indonesian slang words -> formal words from the
colloquial Indonesian lexicon. Hashtags and casing stay as written. `full_text` and the Brandwatch
`language` column are kept unchanged; no language is filtered or translated.
"""

import os
import re
from functools import cache

import pandas as pd
import requests

from pipeline.step import DATA

# Colloquial Indonesian lexicon (Salsabila et al. 2018), pinned to a commit of github.com/nasalsabila/kamus-alay.
LEXICON_URL = (
    "https://raw.githubusercontent.com/nasalsabila/kamus-alay/"
    "c710fa9f3f8d8727c24f3209f4f6a4b6ecaf6d8b/colloquial-indonesian-lexicon.csv"
)
LEXICON_PATH = DATA / "raw" / "lexicon" / "colloquial-indonesian-lexicon.csv"

# Links with a scheme, "www." links, and bare domains such as "saweria.co/abc" (lowercase top-level domain).
URL = re.compile(r"https?://\S+|www\.\S+|\b[\w-]+(?:\.[\w-]+)*\.[a-z]{2,}\b(?:/\S*)?")
MENTION = re.compile(r"@\w+")
# A word that is not part of a hashtag or a mention.
WORD = re.compile(r"(?<![#@\w])\w+")


def download_lexicon() -> None:
    """Save the lexicon once; a re-run reuses the saved file."""
    response = requests.get(LEXICON_URL, timeout=60)
    response.raise_for_status()
    LEXICON_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = LEXICON_PATH.with_name(LEXICON_PATH.name + ".tmp")
    tmp_path.write_bytes(response.content)
    os.replace(tmp_path, LEXICON_PATH)


@cache
def lexicon() -> dict[str, str]:
    """Lowercase slang word -> formal word. The first row wins when a slang word is listed twice."""
    if not LEXICON_PATH.exists():
        download_lexicon()
    rows = pd.read_csv(LEXICON_PATH, usecols=["slang", "formal"], dtype=str).dropna()
    rows = rows.assign(slang=rows["slang"].str.lower()).drop_duplicates("slang")
    return dict(zip(rows["slang"], rows["formal"]))


def formal_word(word: re.Match, slang: dict[str, str]) -> str:
    return word.group() if word.group() == "HTTPURL" else slang.get(word.group().lower(), word.group())


def normalize_text(text: str, slang: dict[str, str]) -> str:
    text = URL.sub("HTTPURL", text)
    text = MENTION.sub("@USER", text)
    return WORD.sub(lambda word: formal_word(word, slang), text)


def run(posts: pd.DataFrame) -> pd.DataFrame:
    slang = lexicon()
    return posts.assign(text=[normalize_text(text, slang) for text in posts["full_text"]])
