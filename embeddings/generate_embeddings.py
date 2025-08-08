import os
import pickle
import numpy as np
import faiss
import fitz  # PyMuPDF for per-page text extraction
from sentence_transformers import SentenceTransformer
from utils.pdf_processing import clean_pdf_text, chunk_text_by_tokens

# ✅ Load embedding model once
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def process_single_pdf(pdf_path):
    """
    Process one PDF: extract, clean, chunk per page, add metadata with page number
    Returns a list of dicts: [{chunk, source, page}]
    """
    pdf_name = os.path.basename(pdf_path)
    print(f"📄 Processing: {pdf_name}")

    all_chunks = []

    # ✅ Open PDF with PyMuPDF
    doc = fitz.open(pdf_path)
    total_pages = len(doc)

    for page_num in range(total_pages):
        page = doc.load_page(page_num)
        page_text = page.get_text("text")

        # ✅ Clean text
        cleaned_text = clean_pdf_text(page_text)

        # ✅ Skip empty pages
        if not cleaned_text.strip():
            continue

        # ✅ Chunk text from this page
        chunks = chunk_text_by_tokens(cleaned_text, chunk_size=512, overlap=50)

        # ✅ Attach metadata with PDF name + page number (1-based indexing)
        for c in chunks:
            all_chunks.append({
                "chunk": c,
                "source": pdf_name,
                "page": page_num + 1
            })

    doc.close()
    print(f"✅ Created {len(all_chunks)} chunks from {pdf_name} ({total_pages} pages)")

    return all_chunks

def process_multiple_pdfs_and_create_index(pdf_folder):
    """
    Process all PDFs in a given folder.
    - Extract & chunk each PDF (page by page)
    - Embed all chunks
    - Save FAISS index & metadata
    """
    # ✅ Check folder
    if not os.path.exists(pdf_folder) or not os.path.isdir(pdf_folder):
        print("❌ Folder not found!")
        return False

    # ✅ List all PDF files
    pdf_files = [f for f in os.listdir(pdf_folder) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print("⚠ No PDFs found in this folder.")
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

    # ✅ Save metadata (chunk + PDF name + page)
    with open("embeddings/chunk_metadata.pkl", "wb") as f:
        pickle.dump(all_chunks, f)

    print("✅ Multi-PDF index created with page metadata! (FAISS + metadata saved)")
    return True

# # ✅ Allow running Phase 2 standalone
# if _name_ == "_main_":
#     folder = input("📂 Enter folder path containing PDFs: ").strip().strip('"')
#     process_multiple_pdfs_and_create_index(folder)