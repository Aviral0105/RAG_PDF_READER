"""
embeddings/generate_embeddings.py

Creates a FAISS index from PDFs in `data/` and saves:
 - embeddings/pdf_index.faiss
 - embeddings/chunk_metadata.pkl

Each metadata entry includes:
 - chunk (text)
 - source (filename)
 - page (None for now; you can extend to per-page chunking)
 - clause_number (e.g., "4.1" or None)
"""

import os
import re
import pickle
from pathlib import Path
from typing import List, Dict, Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# helpers from your repo
from utils.pdf_processing import extract_text_from_pdf, clean_pdf_text, chunk_text_by_tokens

# Configuration
DATA_DIR = Path("data")
EMB_DIR = Path("embeddings")
FAISS_PATH = EMB_DIR / "pdf_index.faiss"
META_PATH = EMB_DIR / "chunk_metadata.pkl"
MODEL_NAME = "all-MiniLM-L6-v2"

# Load embedding model (load once)
embedding_model = SentenceTransformer(MODEL_NAME)

# Regex to capture clause-like patterns such as 3, 3.1, 3.1.2, optionally preceded by "Clause" or "Section"
_clause_regex = re.compile(
    r"""
    (?:
      (?:(?:Clause|Section|SECTION|CLAUSE)\s*)?  # optional leading word
      (\d+(?:\.\d+)+)                           # capture number like 3.1 or 4.2.1
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)


def extract_clause_number_from_text(text: str) -> Optional[str]:
    """
    Try to find a clause/section number in `text`.
    Prefer the beginning of the chunk (first ~200 chars) to detect headings.
    Returns normalized clause string like "5.2" or None.
    """
    if not text:
        return None
    head = text[:200]  # look at start first
    m = _clause_regex.search(head)
    if m:
        return m.group(1)
    m = _clause_regex.search(text)
    if m:
        return m.group(1)
    return None


def process_single_pdf(pdf_path: Path) -> List[Dict]:
    """
    Extract text from a single PDF, clean, chunk, and return metadata items.
    Each returned dict has: chunk, source, page (None), clause_number
    """
    raw_text = extract_text_from_pdf(str(pdf_path), method="pymupdf")
    cleaned = clean_pdf_text(raw_text)
    # chunk_text_by_tokens should return list[str]
    chunks = chunk_text_by_tokens(cleaned, chunk_size=512, overlap=64)

    out = []
    for chunk in chunks:
        clause = extract_clause_number_from_text(chunk)
        out.append({
            "chunk": chunk,
            "source": pdf_path.name,
            "page": None,
            "clause_number": clause
        })
    return out


def process_multiple_pdfs_and_create_index(folder: str = "data"):
    """
    Walk `folder`, process PDFs, build FAISS index and save metadata.
    Returns True on success.
    """
    EMB_DIR = Path("embeddings")
    EMB_DIR.mkdir(parents=True, exist_ok=True)

    all_metadata: List[Dict] = []
    texts: List[str] = []

    data_path = Path(folder)
    pdf_files = sorted([p for p in data_path.iterdir() if p.suffix.lower() == ".pdf"])
    print(f"Found {len(pdf_files)} PDF(s) in {folder}")

    for pdf in pdf_files:
        print(f"Processing {pdf.name} ...")
        items = process_single_pdf(pdf)
        for it in items:
            texts.append(it["chunk"])
            all_metadata.append(it)

    if len(texts) == 0:
        raise RuntimeError("No text chunks found. Check PDF extraction and folder path.")

    # Compute embeddings in batches
    batch_size = 64
    emb_batches = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        emb = embedding_model.encode(batch, show_progress_bar=False)
        emb_batches.append(emb)
    embeddings = np.vstack(emb_batches).astype("float32")

    dim = embeddings.shape[1]
    print(f"Embeddings shape: {embeddings.shape} (dim={dim})")

    # Use IndexFlatL2 wrapped in IndexIDMap for stable id -> metadata mapping
    index = faiss.IndexFlatL2(dim)
    index = faiss.IndexIDMap(index)
    ids = np.arange(len(embeddings)).astype("int64")
    index.add_with_ids(embeddings, ids)

    # Save index & metadata
    faiss.write_index(index, str(FAISS_PATH))
    with open(META_PATH, "wb") as f:
        pickle.dump(all_metadata, f)

    print(f"✅ FAISS index saved to: {FAISS_PATH}")
    print(f"✅ Metadata saved to: {META_PATH}")
    return True
