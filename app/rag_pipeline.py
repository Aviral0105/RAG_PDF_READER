from embeddings.generate_embeddings import process_pdf_and_create_index   # Phase 2
from llm.answer_generator import interactive_qa                        # Phase 4

def run_full_rag_pipeline():
    """
    Full RAG pipeline:
    1️⃣ Index the selected PDF (Phase 2)
    2️⃣ Launch interactive Q&A (Phase 4)
    """
    print("🚀 Starting Full RAG Pipeline...\n")

    # ✅ Phase 2: Index PDF
    success = process_pdf_and_create_index()
    
    # ✅ If indexing failed → stop pipeline
    if not success:
        print("⚠️ Pipeline stopped because PDF could not be indexed.")
        return

    # ✅ Phase 4: Start interactive Q&A session
    interactive_qa()
