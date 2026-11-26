"""Box: Cluster near-duplicates, keep all posts.

Output: the documents plus
    cluster_id   smallest post_id of the post's near-duplicate cluster (its own post_id when it has no duplicate)
Two posts are near-duplicates when their `document` embeddings are close or both have media with close
`phash`; clusters are the connected components of that graph.
"""

import imagehash
import networkx as nx
import numpy as np
import pandas as pd

from pipeline.hf import embed

# Cosine similarity of `document` embeddings at or above which two posts are near-duplicates.
MIN_DOCUMENT_COSINE = 0.90
# pHash Hamming distance (of 64 bits) at or below which two media are near-duplicates.
MAX_PHASH_DISTANCE = 8


def near_duplicate_pairs(embeddings: np.ndarray, phashes: list[str | None]) -> list[tuple[int, int]]:
    """Row index pairs (i < j) that are near-duplicates by document embedding or by pHash."""
    hashes = [imagehash.hex_to_hash(value) if pd.notna(value) else None for value in phashes]
    similarity = embeddings @ embeddings.T
    pairs = []
    for i in range(len(hashes)):
        for j in range(i + 1, len(hashes)):
            close_text = similarity[i, j] >= MIN_DOCUMENT_COSINE
            close_media = hashes[i] is not None and hashes[j] is not None and hashes[i] - hashes[j] <= MAX_PHASH_DISTANCE
            if close_text or close_media:
                pairs.append((i, j))
    return pairs


def cluster_ids(post_ids: list[str], pairs: list[tuple[int, int]]) -> list[str]:
    """Each post's cluster_id: the smallest post_id among its connected near-duplicates."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(post_ids)))
    graph.add_edges_from(pairs)
    ids = [""] * len(post_ids)
    for component in nx.connected_components(graph):
        smallest = min((post_ids[i] for i in component), key=int)
        for i in component:
            ids[i] = smallest
    return ids


def run(posts: pd.DataFrame) -> pd.DataFrame:
    pairs = near_duplicate_pairs(embed(posts["document"].tolist()), posts["phash"].tolist())
    return posts.assign(cluster_id=cluster_ids(posts["post_id"].tolist(), pairs))
