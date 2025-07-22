import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from utils.pdf_processing import extract_text_from_pdf, clean_pdf_text, chunk_text_by_tokens

# ✅ Load embedding model once
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def process_single_pdf(pdf_path):
    """
    Process one PDF: extract, clean, chunk, add metadata
    Returns a list of dicts: [{chunk: "...", source: "PDF_NAME"}]
    """
    pdf_name = os.path.basename(pdf_path)
    print(f"📄 Processing: {pdf_name}")

    # ✅ Extract & clean text
    raw_text = extract_text_from_pdf(pdf_path, method="pymupdf")
    cleaned_text = clean_pdf_text(raw_text)

    # ✅ Chunk text
    chunks = chunk_text_by_tokens(cleaned_text, chunk_size=512, overlap=50)
    print(f"✅ Created {len(chunks)} chunks from {pdf_name}")

    # ✅ Attach metadata
    chunks_with_meta = [{"chunk": c, "source": pdf_name} for c in chunks]
    return chunks_with_meta

def process_multiple_pdfs_and_create_index(pdf_folder):
    """
    Process all PDFs in a given folder.
    - Extract & chunk each PDF
    - Embed all chunks
    - Save FAISS index & metadata
    """
    # ✅ Check if folder exists
    if not os.path.exists(pdf_folder) or not os.path.isdir(pdf_folder):
        print("❌ Folder not found!")
        return False

    # ✅ List all PDF files
    pdf_files = [f for f in os.listdir(pdf_folder) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print("⚠️ No PDFs found in this folder.")
        return False

    all_chunks = []

    # ✅ Process each PDF
    for pdf in pdf_files:
        pdf_path = os.path.join(pdf_folder, pdf)
        pdf_chunks = process_single_pdf(pdf_path)
        all_chunks.extend(pdf_chunks)

    print(f"\n✅ Finished processing {len(pdf_files)} PDFs")
    print(f"✅ Total chunks created: {len(all_chunks)}")

    if not all_chunks:
        print("❌ No chunks generated!")
        return False

    # ✅ Extract only chunk texts for embeddings
    chunk_texts = [item["chunk"] for item in all_chunks]

    # ✅ Generate embeddings for all chunks
    print("\n🔄 Generating embeddings for ALL chunks...")
    embeddings = embedding_model.encode(chunk_texts, show_progress_bar=True)
    faiss.normalize_L2(embeddings)

    # ✅ Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(np.array(embeddings, dtype="float32"))

    # ✅ Save FAISS index
    faiss.write_index(index, "embeddings/pdf_index.faiss")

    # ✅ Save metadata (chunk + PDF name)
    with open("embeddings/chunk_metadata.pkl", "wb") as f:
        pickle.dump(all_chunks, f)

    print("✅ Multi-PDF index created! (FAISS + metadata saved)")
    return True
