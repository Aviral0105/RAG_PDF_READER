# rag_pipeline.py

import os
import tempfile
import requests
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Import your custom functions from other project files
from embeddings.generate_embeddings import process_single_pdf
from llm.answer_generator import generate_answer_with_gpt
from retriever.query_retriever import retrieve_top_k_chunks

# --- Model Loading ---
# Load the embedding model once when the application starts to improve performance
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


# --- Helper Function: Processes a PDF from a URL ---
def create_index_from_url(url: str):
    """
    Downloads a PDF from a URL, processes it using your existing function,
    and creates an in-memory FAISS vector database.
    """
    print(f"📥 Downloading PDF from: {url}")
    try:
        response = requests.get(url, timeout=30)
        # Raise an exception for bad status codes (like 404 Not Found)
        response.raise_for_status()
    except requests.RequestException as e:
        raise ConnectionError(f"Failed to download PDF: {e}")

    # Use a temporary file to safely handle the downloaded content
    with tempfile.NamedTemporaryFile(delete=True, suffix=".pdf") as temp_file:
        temp_file.write(response.content)
        pdf_path = temp_file.name

        # Reuse your existing function to chunk the PDF and add metadata
        chunks = process_single_pdf(pdf_path)
        if not chunks:
            return None, None

    # Create the vector database directly in memory
    print("🧠 Creating in-memory vector index...")
    chunk_texts = [item["chunk"] for item in chunks]
    embeddings = embedding_model.encode(chunk_texts, show_progress_bar=False)
    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(np.array(embeddings, dtype="float32"))

    print("✅ In-memory index created successfully.")
    # Return the index and the chunk metadata for this request
    return index, chunks


# --- The Single Interface Function for the API ---
def process_api_request(doc_url: str, questions: list) -> list:
    """
    This single function handles the entire RAG process for an API request.
    It maintains a conversational context using a sliding window of the last 3 exchanges.
    """
    print(f"🚀 Starting new RAG job for URL: {doc_url}")
    try:
        faiss_index, chunk_metadata = create_index_from_url(doc_url)
        if not faiss_index:
            raise ValueError("Could not extract any content from the document.")
    except Exception as e:
        # If document processing fails, return an error for all questions
        return [{"question": q, "answer": str(e)} for q in questions]

    final_answers = []
    # 1. Initialize chat history for this specific API request
    chat_history = []
    # A window of 3 questions means 6 items total (3 user, 3 assistant)
    window_size = 6 

    # 2. Loop through the list of questions from the API
    for question in questions:
        retrieved_chunks = retrieve_top_k_chunks(
            question, faiss_index, chunk_metadata, k=3
        )
        context = "\n\n".join([item["chunk"] for item in retrieved_chunks])

        if not context:
            answer = "I could not find relevant information in the document to answer this question."
        else:
            # 3. Pass the current history to the LLM
            answer = generate_answer_with_gpt(
                chat_history=chat_history,
                query=question,
                context=context
            )
        
        final_answers.append({"question": question, "answer": answer})

        # 4. Update the history with the latest exchange
        chat_history.append({"role": "user", "content": question})
        chat_history.append({"role": "assistant", "content": answer})

        # 5. Trim the history to maintain the sliding window
        if len(chat_history) > window_size:
            # Keep only the most recent items
            chat_history = chat_history[-window_size:]

    print("✅ RAG job finished successfully.")
    return final_answers