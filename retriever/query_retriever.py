import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ✅ Paths
FAISS_PATH = "embeddings/pdf_index.faiss"
METADATA_PATH = "embeddings/chunk_metadata.pkl"

# ✅ Load embedding model once
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def load_faiss_and_metadata():
    """
    Load FAISS index + metadata.
    Metadata now includes: chunk, source, page.
    """
    print("🔄 Loading FAISS index & metadata...")

    faiss_index = faiss.read_index(FAISS_PATH)

    with open(METADATA_PATH, "rb") as f:
        chunk_metadata = pickle.load(f)

    print(f"✅ Loaded FAISS index with {faiss_index.ntotal} vectors")
    return faiss_index, chunk_metadata


def retrieve_top_k_chunks(query, faiss_index, chunk_metadata, k=3, pdf_name=None, page_range=None):
    """
    Retrieve top-k chunks for a query with optional filtering.
    - pdf_name: restrict search to specific PDF
    - page_range: (start_page, end_page) to restrict by pages
    """
    # ✅ Start with all chunks
    allowed_indices = list(range(len(chunk_metadata)))

    # ✅ Filter by PDF name
    if pdf_name:
        allowed_indices = [i for i in allowed_indices if chunk_metadata[i]["source"] == pdf_name]

    # ✅ Filter by page range
    if page_range:
        start_page, end_page = page_range
        allowed_indices = [
            i for i in allowed_indices
            if start_page <= chunk_metadata[i]["page"] <= end_page
        ]

    # ✅ Apply filtering if needed
    if len(allowed_indices) != len(chunk_metadata):
        print(f"📄 Applying metadata filter: {len(allowed_indices)} chunks remain")

        # Extract ALL vectors from FAISS
        all_vectors = faiss_index.reconstruct_n(0, faiss_index.ntotal)

        # Filtered vectors + metadata
        filtered_vectors = np.array([all_vectors[i] for i in allowed_indices])
        filtered_metadata = [chunk_metadata[i] for i in allowed_indices]

        # Create a temporary FAISS index for filtered subset
        temp_index = faiss.IndexFlatIP(filtered_vectors.shape[1])
        temp_index.add(filtered_vectors)

        search_index = temp_index
        metadata_to_use = filtered_metadata
    else:
        search_index = faiss_index
        metadata_to_use = chunk_metadata

    # ✅ Encode query
    query_embedding = embedding_model.encode([query])
    faiss.normalize_L2(query_embedding)

    # ✅ Search FAISS
    scores, indices = search_index.search(
        np.array(query_embedding, dtype="float32"), k
    )

    results = []
    for idx, score in zip(indices[0], scores[0]):
        meta = metadata_to_use[idx]
        results.append({
            "chunk": meta["chunk"],
            "source": meta["source"],
            "page": meta["page"],    # ✅ include page number
            "score": float(score)
        })

    return results


# ✅ Test standalone
# if _name_ == "_main_":
#     faiss_index, metadata = load_faiss_and_metadata()

#     # Example: normal retrieval
#     query = "What is Artificial Intelligence?"
#     results = retrieve_top_k_chunks(query, faiss_index, metadata, k=3)

#     print("\n🔍 Results:")
#     for r in results:
#         print(f"\n📄 {r['source']} (Page {r['page']}) | Score: {r['score']:.4f}")
#         print(r["chunk"])