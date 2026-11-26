"""Box: Narrative coding: IndoBERT clusters, classifier on coded sample.

Output: one row per post. Columns: post_id, narrative_cluster (int), narrative_theme (top TFIDF_TERMS
TF-IDF terms of the cluster, joined by ", ").
Documents are embedded with IndoBERT (mean-pooled last hidden state) and clustered with KMeans.
Not built: the board's "classifier on coded sample" needs a manually coded sample of posts, which does not exist yet.
Open question: an LLM could group narratives instead.
"""

from functools import cache

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModel, AutoTokenizer

from pipeline.hf import DEVICE

INDOBERT_MODEL = "indobenchmark/indobert-base-p1"
INDOBERT_REVISION = "c2cd0b51ddce6580eb35263b39b0a1e5fb0a39e2"
NARRATIVE_CLUSTERS = 8  # KMeans clusters, capped at the number of unique documents
TFIDF_TERMS = 3  # terms per narrative_theme
MAX_TOKENS = 512  # IndoBERT input limit
SEED = 42


@cache
def indobert() -> tuple[AutoTokenizer, AutoModel]:
    tokenizer = AutoTokenizer.from_pretrained(INDOBERT_MODEL, revision=INDOBERT_REVISION)
    model = AutoModel.from_pretrained(INDOBERT_MODEL, revision=INDOBERT_REVISION).to(DEVICE).eval()
    return tokenizer, model


def embed_indobert(documents: list[str]) -> np.ndarray:
    tokenizer, model = indobert()
    vectors = []
    for document in documents:
        inputs = tokenizer(document, truncation=True, max_length=MAX_TOKENS, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            hidden = model(**inputs).last_hidden_state[0]
        vectors.append(hidden.mean(dim=0).cpu().numpy())
    return np.stack(vectors)


def themes(documents: list[str], clusters: np.ndarray) -> dict[int, str]:
    tfidf = TfidfVectorizer()
    weights = tfidf.fit_transform(documents).toarray()
    terms = tfidf.get_feature_names_out()
    result = {}
    for cluster in sorted(set(clusters)):
        mean = weights[clusters == cluster].mean(axis=0)
        top = sorted(range(len(terms)), key=lambda i: (-mean[i], terms[i]))[:TFIDF_TERMS]
        result[int(cluster)] = ", ".join(terms[i] for i in top)
    return result


def run(posts: pd.DataFrame) -> pd.DataFrame:
    documents = sorted(posts["document"].unique())
    kmeans = KMeans(n_clusters=min(NARRATIVE_CLUSTERS, len(documents)), random_state=SEED, n_init=10)
    clusters = kmeans.fit_predict(embed_indobert(documents))
    cluster_of = dict(zip(documents, clusters.tolist()))
    theme_of = themes(documents, clusters)
    labels = posts[["post_id"]].copy()
    labels["narrative_cluster"] = posts["document"].map(cluster_of).astype(int)
    labels["narrative_theme"] = labels["narrative_cluster"].map(theme_of)
    return labels.sort_values("post_id", ignore_index=True)
