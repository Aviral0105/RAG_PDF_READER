import os
import sys
import tempfile
from pathlib import Path
from typing import List

import faiss
import numpy as np
import requests
import uvicorn
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, HttpUrl
from sentence_transformers import SentenceTransformer

# Load environment variables from a .env file for API keys
load_dotenv()

# Add parent directory to path to import your custom modules
# This assumes your script is in an 'app' directory, e.g., 'RAG_PDF_READER-main/app/api.py'
sys.path.append(str(Path(__file__).parent.parent))

from embeddings.generate_embeddings import process_single_pdf
from llm.answer_generator import generate_answer_with_gpt
from retriever.query_retriever import retrieve_top_k_chunks

# --- FastAPI App Initialization ---
app = FastAPI(
    title="RAG PDF Reader API",
    description="A public API for querying PDF documents using RAG.",
    version="1.0.0",
)

# --- Pydantic Models for Request and Response ---
class HackRxRequest(BaseModel):
    documents: HttpUrl  # Pydantic validates this is a proper URL
    questions: List[str]

class HackRxResponse(BaseModel):
    answers: List[str]

# --- Caching and Model Loading ---
# Cache to store processed documents (FAISS index + metadata) in memory
# Key: document URL, Value: (faiss_index, chunk_metadata)
document_cache = {}

# Use a singleton pattern for the embedding model to avoid reloading it
embedding_model = None

def get_embedding_model():
    """Loads the sentence-transformer model into memory."""
    global embedding_model
    if embedding_model is None:
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return embedding_model

# --- Security and Authentication ---
security_scheme = HTTPBearer()

def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security_scheme)):
    """Verifies the API key from the Authorization header."""
    # Securely fetch the expected API key from environment variables
    expected_key = os.getenv("API_KEY")
    if not expected_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API_KEY is not configured on the server.",
        )

    if credentials.credentials != expected_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return True

# --- Core RAG Logic Functions ---
def get_or_create_document_index(doc_url: str):
    """
    Retrieves a processed document from the cache or processes it from its URL.
    This function contains the core logic for downloading, chunking, and embedding.
    """
    # 1. Check if the document is already in our cache
    if doc_url in document_cache:
        print(f"✅ Loading document from cache: {doc_url}")
        return document_cache[doc_url]

    print(f"📥 Downloading and processing document: {doc_url}")
    # 2. Download the PDF to a temporary file
    try:
        response = requests.get(doc_url, timeout=30)
        response.raise_for_status()  # Raise an exception for bad status codes
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(response.content)
            pdf_path = temp_file.name
    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to download PDF: {e}")

    # 3. Process the PDF: chunking, embedding, and indexing
    try:
        # Assume process_single_pdf returns a list of chunk dictionaries
        chunks = process_single_pdf(pdf_path)
        if not chunks:
            raise ValueError("No content could be extracted from the PDF.")

        model = get_embedding_model()
        chunk_texts = [item["chunk"] for item in chunks]
        embeddings = model.encode(chunk_texts, show_progress_bar=False)
        faiss.normalize_L2(embeddings)

        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(np.array(embeddings, dtype="float32"))

        # 4. Store the processed index and metadata in the cache
        document_cache[doc_url] = (index, chunks)
        print(f"✅ Document processed and cached successfully.")
        return index, chunks

    except Exception as e:
        # If processing fails, raise a server error
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {e}")
    finally:
        # 5. Clean up the temporary file
        if os.path.exists(pdf_path):
            os.unlink(pdf_path)


# --- API Endpoints ---
@app.post("/hackrx/run", response_model=HackRxResponse, tags=["RAG"])
async def process_questions(
    request: HackRxRequest,
    authenticated: bool = Depends(verify_api_key)
):
    """
    Processes questions against a PDF document using the RAG pipeline.
    This endpoint will cache the processed document for subsequent requests.
    """
    try:
        # Get the processed document index (from cache or by processing it)
        doc_url_str = str(request.documents)
        faiss_index, chunk_metadata = get_or_create_document_index(doc_url_str)

        answers = []
        # For each question, retrieve context and generate an answer
        for question in request.questions:
            # Assume retrieve_top_k_chunks uses the embedding model internally
            retrieved_chunks = retrieve_top_k_chunks(question, faiss_index, chunk_metadata, k=3)

            if not retrieved_chunks:
                answers.append("I could not find relevant information in the document to answer this question.")
                continue

            # Build context string from retrieved chunks
            context = "\n\n".join([item["chunk"] for item in retrieved_chunks])
            
            # Assume generate_answer_with_gpt takes the question and context
            answer = generate_answer_with_gpt([], question, context)
            answers.append(answer)

        return HackRxResponse(answers=answers)
    except HTTPException as e:
        # Re-raise known HTTP exceptions
        raise e
    except Exception as e:
        # Catch any other unexpected errors
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {e}")


@app.get("/", tags=["General"])
async def root():
    """Root endpoint with API information."""
    return {"message": "Welcome to the RAG PDF Reader API"}


@app.get("/health", tags=["General"])
async def health_check():
    """Health check endpoint to verify the API is running."""
    return {"status": "healthy"}