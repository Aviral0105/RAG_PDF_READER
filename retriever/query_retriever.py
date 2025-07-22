import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ✅ Paths for FAISS & metadata
FAISS_PATH = "embeddings/pdf_index.faiss"
METADATA_PATH = "embeddings/chunk_metadata.pkl"

# ✅ Same embedding model used in Phase 2
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def load_faiss_and_metadata():
    """
    Loads FAISS index + multi-PDF metadata.
    Returns:
        faiss_index (FAISS object)
        chunk_metadata (list of dicts: [{chunk, source}])
    """
    print("🔄 Loading FAISS index & metadata...")

    # ✅ Load FAISS index
    faiss_index = faiss.read_index(FAISS_PATH)

    # ✅ Load metadata
    with open(METADATA_PATH, "rb") as f:
        chunk_metadata = pickle.load(f)

    print(f"✅ Loaded FAISS index with {faiss_index.ntotal} vectors")
    return faiss_index, chunk_metadata

def retrieve_top_k_chunks(query: str, faiss_index, chunk_metadata, k=3):
    """
    Given a query → retrieves top-k most similar chunks from multi-PDF index.
    Returns a list of dicts:
        [{"chunk": "...", "source": "PDF_NAME", "score": 0.89}, ...]
    """

    # ✅ Encode the query
    query_embedding = embedding_model.encode([query])

    # ✅ Normalize for cosine similarity
    faiss.normalize_L2(query_embedding)

    # ✅ Search FAISS index → get similarity scores & indices
    scores, indices = faiss_index.search(
        np.array(query_embedding, dtype="float32"), k
    )

    # ✅ Collect top-k results with metadata
    results = []
    for idx, score in zip(indices[0], scores[0]):
        meta = chunk_metadata[idx]  # {"chunk": "...", "source": "..."}
        results.append({
            "chunk": meta["chunk"],
            "source": meta["source"],
            "score": float(score)
        })

    return results
