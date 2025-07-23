from embeddings.generate_embeddings import process_multiple_pdfs_and_create_index   # ✅ correct file
from llm.answer_generator import interactive_qa                                     # Multi-PDF Q&A


def run_full_rag_pipeline():
    """
    Full Multi-PDF RAG Pipeline:
    1️⃣ Ask for folder → Index all PDFs
    2️⃣ Launch interactive multi-PDF Q&A
    """
    print("🚀 Starting Full Multi‑PDF RAG Pipeline...\n")

    # ✅ Ask for folder path only ONCE here
    folder_path = input("📂 Enter the folder containing PDFs: ").strip().strip('"')

    # ✅ Phase 2: Multi-PDF indexing
    success = process_multiple_pdfs_and_create_index(folder_path)

    # ✅ If indexing failed → stop pipeline
    if not success:
        print("⚠️ Pipeline stopped (no PDFs indexed)")
        return

    # ✅ Phase 4: Start interactive Q&A session
    interactive_qa()
