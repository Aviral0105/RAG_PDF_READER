import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# ✅ File paths
FAISS_PATH = "embeddings/pdf_index.faiss"
METADATA_PATH = "embeddings/chunk_metadata.pkl"

# ✅ Load the same embedding model used in Phase 2
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


def load_faiss_and_metadata():
    """
    Load the FAISS index (chunk embeddings) and metadata (chunk texts).
    """
    print("🔄 Loading FAISS index & metadata...")

    # ✅ Load FAISS index created in Phase 2
    index = faiss.read_index(FAISS_PATH)

    # ✅ Load chunk metadata (list of chunk texts)
    with open(METADATA_PATH, "rb") as f:
        metadata = pickle.load(f)

    print(f"✅ Loaded FAISS index with {index.ntotal} vectors")
    return index, metadata


def retrieve_top_k_chunks(query, faiss_index, chunk_metadata, k=3):
    """
    1️⃣ Convert query text → embedding
    2️⃣ Normalize query → for cosine similarity
    3️⃣ Search FAISS → top-K most similar chunks
    """
    # ✅ 1. Convert query → embedding (384-dim vector)
    query_embedding = embedding_model.encode([query])

    # ✅ 2. Normalize query → dot product = cosine similarity
    faiss.normalize_L2(query_embedding)

    # ✅ 3. Search FAISS for top-K similar chunks
    similarities, indices = faiss_index.search(
        np.array(query_embedding, dtype="float32"), k
    )

    # ✅ 4. Collect top-K chunks with similarity scores
    retrieved_chunks = []
    for idx, sim in zip(indices[0], similarities[0]):
        retrieved_chunks.append((chunk_metadata[idx], sim))

    return retrieved_chunks


if __name__ == "__main__":
    # ✅ Step 1: Load FAISS & metadata
    faiss_index, chunk_metadata = load_faiss_and_metadata()

    while True:
        # ✅ Step 2: Ask user for query
        query = input("\n❓ Enter your query (or 'exit' to quit): ")
        if query.lower() == "exit":
            break

        # ✅ Step 3: Retrieve top-3 relevant chunks
        results = retrieve_top_k_chunks(
            query,
            faiss_index,
            chunk_metadata,
            k=3  # fixed K = 3
        )

        # ✅ Step 4: Print results
        print("\n🔎 Top 3 relevant chunks (using cosine similarity):\n")
        for i, (chunk_text, sim) in enumerate(results, 1):
            print(f"--- Result {i} (similarity={sim:.4f}) ---\n{chunk_text[:500]}\n")
