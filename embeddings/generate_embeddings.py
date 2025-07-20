from utils.pdf_processing import extract_text_from_pdf, clean_pdf_text, chunk_text_by_tokens
from sentence_transformers import SentenceTransformer
import numpy as np
import pickle
import faiss
import os

# ✅ Load embedding model once
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def process_pdf_and_create_index():
    """
    Phase 2:
    - Asks for a PDF path
    - Extracts, cleans, and chunks the text
    - Creates embeddings & saves FAISS index + metadata
    Returns:
        True if indexing successful
        None if indexing failed
    """

    # ✅ 1. Ask user for PDF path
    pdf_path = input("📂 Enter the path of your PDF: ").strip().strip('"')  # removes accidental quotes

    # ✅ Validate the path
    if not os.path.exists(pdf_path):
        print("❌ File not found! Please check the path.")
        return None  # ❌ return None so pipeline knows it failed

    print(f"📄 Processing: {os.path.basename(pdf_path)}")

    # ✅ 2. Extract text from PDF
    raw_text = extract_text_from_pdf(pdf_path, method="pymupdf")

    # ✅ 3. Clean & normalize text
    cleaned_text = clean_pdf_text(raw_text)

    # ✅ 4. Split into overlapping chunks
    chunks = chunk_text_by_tokens(cleaned_text, chunk_size=512, overlap=50)
    print(f"✅ Extracted & split into {len(chunks)} chunks")

    # ✅ 5. Generate embeddings for chunks
    print("🔄 Generating embeddings...")
    embeddings = embedding_model.encode(chunks, show_progress_bar=True)

    # ✅ Normalize embeddings for cosine similarity search
    faiss.normalize_L2(embeddings)

    # ✅ 6. Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(np.array(embeddings, dtype="float32"))

    # ✅ 7. Save FAISS index & metadata
    faiss.write_index(index, "embeddings/pdf_index.faiss")
    with open("embeddings/chunk_metadata.pkl", "wb") as f:
        pickle.dump(chunks, f)

    print("✅ PDF indexed successfully! (FAISS + metadata saved)")

    return True  # ✅ success
