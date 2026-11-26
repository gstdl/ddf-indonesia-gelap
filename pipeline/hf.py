"""Shared Hugging Face helpers: one device choice and one sentence-embedding model for every box."""

from functools import cache

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# Multilingual (incl. Indonesian) sentence embeddings for similarity and clustering.
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_REVISION = "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"


@cache
def embedder() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL, revision=EMBEDDING_REVISION, device=DEVICE)


def embed(texts: list[str]) -> np.ndarray:
    """Unit-length embeddings, so a dot product is the cosine similarity."""
    return embedder().encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
