import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# ✅ Import Phase 1 functions
from utils.pdf_processing import extract_text_from_pdf, clean_pdf_text, chunk_text_by_tokens

# ✅ Load local embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # 384-dim embeddings

def generate_embeddings_for_pdf(pdf_path, chunk_size=512, overlap=50):
    """
    1. Extract -> Clean -> Chunk PDF
    2. Generate embeddings for each chunk
    3. Save embeddings + FAISS index + metadata
    """
    
    # 1️⃣ Extract, Clean & Chunk text
    print(f"📄 Processing PDF: {pdf_path}")
    raw_text = extract_text_from_pdf(pdf_path, method="pymupdf")
    cleaned_text = clean_pdf_text(raw_text)
    chunks = chunk_text_by_tokens(cleaned_text, chunk_size, overlap)
    print(f"✅ Total Chunks: {len(chunks)}")

    # 2️⃣ Generate embeddings for all chunks
    print("🔄 Generating embeddings...")
    embeddings = embedding_model.encode(chunks, show_progress_bar=True)  # numpy array [num_chunks x 384]

    # 3️⃣ Create FAISS index using Euclidean distance
    #dimension = embeddings.shape[1]  # e.g., 384 for MiniLM
    #index = faiss.IndexFlatL2(dimension)  # L2 distance (Euclidean)
    #index.add(np.array(embeddings, dtype="float32"))  # add all chunk vectors
    #print(f"✅ FAISS index created with {index.ntotal} vectors.")
    
    # ✅ Normalize embeddings to unit length (so inner product = cosine similarity)
    faiss.normalize_L2(embeddings)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner Product index
    index.add(np.array(embeddings, dtype="float32"))  # Now it stores normalized vectors
    print(f"✅ FAISS index created with {index.ntotal} vectors.")



    # 4️⃣ Save FAISS index to disk
    faiss_path = "embeddings/pdf_index.faiss"
    faiss.write_index(index, faiss_path)

    # 5️⃣ Save chunk metadata (so we know which text belongs to each vector)
    metadata_path = "embeddings/chunk_metadata.pkl"
    with open(metadata_path, "wb") as f:
        pickle.dump(chunks, f)

    print(f"✅ Saved FAISS index → {faiss_path}")
    print(f"✅ Saved chunk metadata → {metadata_path}")
    
    return index, chunks


if __name__ == "__main__":
    sample_pdf = "data/Document_1.pdf"  # Put your test PDF here
    generate_embeddings_for_pdf(sample_pdf)
