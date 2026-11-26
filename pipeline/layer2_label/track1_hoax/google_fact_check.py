"""Box: Google Fact Check.

Output: one row per Indonesian claim review from the Google Fact Check Tools API (claims:search).
Columns: claim_text, claimant, claim_date, publisher_name, publisher_site, source_url (review URL),
review_title, textual_rating, published_at (review date), query.

claims:search accepts only an API key, so Application Default Credentials are used against the
API Keys API to reuse or mint a key restricted to factchecktools.googleapis.com (named
KEY_DISPLAY_NAME in PROJECT_ID, both from .env). The key travels in the X-Goog-Api-Key header,
never in a URL, so it cannot leak into errors, logs or output.
"""

import os
import time
from pathlib import Path

import google.auth
import google.auth.transport.requests
import pandas as pd
import requests
import urllib3
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[3]

CLAIMS_SEARCH_API = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
API_KEYS_API = "https://apikeys.googleapis.com/v2"
SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]
COLUMNS = [
    "claim_text",
    "claimant",
    "claim_date",
    "publisher_name",
    "publisher_site",
    "source_url",
    "review_title",
    "textual_rating",
    "published_at",
    "query",
]

# Language of the claim reviews to keep (Indonesian fact-checkers).
LANGUAGE_CODE = "id"
# Claims per request (API maximum).
PAGE_SIZE = 100
# Safety cap on token-paginated pages per query (100 claims each).
MAX_PAGES = 20
# Search vocabulary for the #IndonesiaGelap protests (Feb 2025 onward): the movement name, the protest
# actors and grievances (budget cuts, KIP Kuliah scholarships, free meals, Danantara, TNI law), the
# government figures targeted, and the AI-manipulation terms fact-checkers use for synthetic media.
QUERIES = [
    "indonesia gelap",
    "demo mahasiswa",
    "aksi mahasiswa",
    "efisiensi anggaran",
    "pemotongan anggaran",
    "danantara",
    "kip kuliah",
    "makan bergizi gratis",
    "uu tni",
    "prabowo",
    "gibran",
    "sri mulyani",
    "dpr",
    "deepfake",
    "video ai",
    "rekayasa ai",
]


def http_session() -> requests.Session:
    """Pooled session that retries GETs on rate limits and server errors."""
    session = requests.Session()
    retry = urllib3.util.Retry(
        total=4, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["GET"]
    )
    session.mount("https://", requests.adapters.HTTPAdapter(max_retries=retry))
    return session


def read_json(response: requests.Response) -> dict:
    """JSON body; on failure raise with the server message (the key is in a header, never in the URL)."""
    if not response.ok:
        raise RuntimeError(f"{response.request.method} {response.status_code}: {response.text}")
    return response.json()


def api_keys_session(project_id: str) -> requests.Session:
    """Session carrying an Application Default Credentials bearer token for the API Keys API."""
    credentials, _ = google.auth.default(scopes=SCOPES)
    credentials.refresh(google.auth.transport.requests.Request())
    session = http_session()
    session.headers.update({"Authorization": f"Bearer {credentials.token}", "x-goog-user-project": project_id})
    return session


def wait_for_operation(session: requests.Session, operation: dict) -> dict:
    """Poll a long-running API Keys operation until it finishes."""
    for _ in range(60):
        if operation.get("done"):
            break
        time.sleep(1)
        operation = read_json(session.get(f"{API_KEYS_API}/{operation['name']}", timeout=30))
    if not operation.get("done"):
        raise TimeoutError(f"operation {operation['name']} did not finish within 60 s")
    if "error" in operation:
        raise RuntimeError(f"operation failed: {operation['error']}")
    return operation["response"]


def ensure_api_key(project_id: str, key_display_name: str) -> str:
    """Key string of the Fact Check Tools API key; reuses the key named key_display_name if present."""
    session = api_keys_session(project_id)
    parent = f"{API_KEYS_API}/projects/{project_id}/locations/global"
    existing = read_json(session.get(f"{parent}/keys", params={"pageSize": 300}, timeout=30))
    names = [key["name"] for key in existing.get("keys", []) if key.get("displayName") == key_display_name]
    if names:
        name = names[0]
    else:
        body = {
            "displayName": key_display_name,
            "restrictions": {"apiTargets": [{"service": "factchecktools.googleapis.com"}]},
        }
        name = wait_for_operation(session, read_json(session.post(f"{parent}/keys", json=body, timeout=30)))["name"]
    return read_json(session.get(f"{API_KEYS_API}/{name}/keyString", timeout=30))["keyString"]


def search_claims(session: requests.Session, query: str) -> list[dict]:
    """Every claim for one query, following nextPageToken up to MAX_PAGES pages."""
    claims = []
    params = {"query": query, "languageCode": LANGUAGE_CODE, "pageSize": PAGE_SIZE}
    for _ in range(MAX_PAGES):
        payload = read_json(session.get(CLAIMS_SEARCH_API, params=params, timeout=30))
        claims.extend(payload.get("claims", []))
        if "nextPageToken" not in payload:
            break
        params["pageToken"] = payload["nextPageToken"]
    return claims


def claim_rows(claim: dict, query: str) -> list[dict]:
    """One row per Indonesian review of the claim."""
    return [
        {
            "claim_text": claim.get("text"),
            "claimant": claim.get("claimant"),
            "claim_date": claim.get("claimDate"),
            "publisher_name": review.get("publisher", {}).get("name"),
            "publisher_site": review.get("publisher", {}).get("site"),
            "source_url": review.get("url"),
            "review_title": review.get("title"),
            "textual_rating": review.get("textualRating"),
            "published_at": review.get("reviewDate"),
            "query": query,
        }
        for review in claim.get("claimReview", [])
        if review.get("languageCode") == LANGUAGE_CODE
    ]


def run() -> pd.DataFrame:
    load_dotenv(ROOT / ".env")
    session = http_session()
    session.headers["X-Goog-Api-Key"] = ensure_api_key(os.environ["PROJECT_ID"], os.environ["KEY_DISPLAY_NAME"])
    rows = [row for query in QUERIES for claim in search_claims(session, query) for row in claim_rows(claim, query)]
    claims = pd.DataFrame(rows, columns=COLUMNS)
    claims["claim_date"] = pd.to_datetime(claims["claim_date"], utc=True, format="ISO8601")
    claims["published_at"] = pd.to_datetime(claims["published_at"], utc=True, format="ISO8601")
    # A claim found by several queries keeps its first query (QUERIES order).
    claims["query_order"] = claims["query"].map({query: order for order, query in enumerate(QUERIES)})
    claims = claims.sort_values(["source_url", "claim_text", "query_order"], kind="stable")
    claims = claims.drop_duplicates(["source_url", "claim_text"]).drop(columns="query_order")
    return claims.reset_index(drop=True)
