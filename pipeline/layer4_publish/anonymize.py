"""Box: Ethics and anonymization review. Code anonymizes; the ethics review is done by people.

Output: the input table with IDENTIFYING_COLUMNS replaced by sha256(ANONYMIZATION_SALT + value) hex[:HASH_LENGTH]
and URL and free-text columns (DROPPED_COLUMNS) removed, for whichever of them are present. Missing values stay missing.
ANONYMIZATION_SALT comes from .env.
"""

import hashlib
import os
from pathlib import Path

import dotenv
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

# Columns holding X handles or account ids.
IDENTIFYING_COLUMNS = ["author", "author_id", "community_id", "source_author", "target_author"]
# URL and free-text columns.
DROPPED_COLUMNS = [
    "url", "full_text", "text", "document", "media_url", "reply_to_url", "retweet_of_url",
    "original_url", "expanded_urls", "mentioned_authors",
]
HASH_LENGTH = 16  # hex characters kept from the sha256 digest


def pseudonym(value: object, salt: str) -> str:
    return hashlib.sha256((salt + str(value)).encode()).hexdigest()[:HASH_LENGTH]


def run(table: pd.DataFrame) -> pd.DataFrame:
    dotenv.load_dotenv(ROOT / ".env")
    salt = os.environ["ANONYMIZATION_SALT"]
    table = table.drop(columns=[column for column in DROPPED_COLUMNS if column in table.columns])
    for column in IDENTIFYING_COLUMNS:
        if column in table.columns:
            table[column] = table[column].map(lambda value: pseudonym(value, salt) if pd.notna(value) else value)
    return table
