"""
retriever/query_retriever.py

Load FAISS index + metadata, and provide retrieval with optional
clause_number filtering.

Functions exposed:
 - load_faiss_and_metadata()
 - retrieve_top_k_chunks(query, index, metadata, k=5, clause_number=None)
"""

import os
import pickle
from pathlib import Path
from typing import List, Dict, Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

EMB_DIR = Path("embeddings")
FAISS_PATH = EMB_DIR / "pdf_index.faiss"
META_PATH = EMB_DIR / "chunk_metadata.pkl"
MODEL_NAME = "all-MiniLM-L6-v2"

# load embedding model once
embedding_model = SentenceTransformer(MODEL_NAME)


def load_faiss_and_metadata():
    """
    Load FAISS index and metadata list. metadata[i] corresponds to vector id i.
    """
    if not FAISS_PATH.exists():
        raise FileNotFoundError(f"FAISS index not found at {FAISS_PATH}. Run embedding generator.")
    if not META_PATH.exists():
        raise FileNotFoundError(f"Metadata not found at {META_PATH}. Run embedding generator.")

    index = faiss.read_index(str(FAISS_PATH))
    with open(META_PATH, "rb") as fh:
        metadata = pickle.load(fh)

    return index, metadata


def retrieve_top_k_chunks(query: str,
                          index,
                          metadata: List[Dict],
                          k: int = 5,
                          clause_number: Optional[str] = None,
                          search_k: int = 50) -> List[Dict]:
    """
    Retrieve top-k chunks for a query.
    If clause_number is provided, filter results to only chunks whose metadata
    'clause_number' equals it (string match). Because FAISS doesn't support
    metadata filtering, we over-retrieve `search_k` results and filter in Python.

    Returns list of dicts:
    {
      "chunk": ...,
      "source": ...,
      "page": ...,
      "clause_number": ...,
      "score": float  # FAISS distance
    }
    """
    query = (query or "").strip()
    if not query:
        return []

    q_emb = embedding_model.encode([query]).astype("float32")
    D, I = index.search(q_emb, search_k)
    distances = D[0]
    indices = I[0]

    results = []
    for dist, idx in zip(distances, indices):
        if idx < 0:
            continue
        try:
            meta = metadata[int(idx)]
        except Exception:
            # skip mismatched index
            continue

        # clause filtering if requested
        if clause_number:
            meta_clause = meta.get("clause_number")
            if not meta_clause:
                continue
            if str(meta_clause).strip() != str(clause_number).strip():
                continue

        results.append({
            "chunk": meta.get("chunk"),
            "source": meta.get("source"),
            "page": meta.get("page"),
            "clause_number": meta.get("clause_number"),
            "score": float(dist)
        })

        if len(results) >= k:
            break

    return results
