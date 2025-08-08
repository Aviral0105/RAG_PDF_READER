import os
from typing import List

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, HttpUrl

# Import the single, unified function from your RAG pipeline
from rag_pipeline import process_api_request

# --- FastAPI App Initialization ---
app = FastAPI(
    title="RAG PDF API",
    description="An API that processes a PDF from a URL and answers questions using a RAG pipeline.",
    version="1.0.0",
)

# --- Pydantic Models: Defines the structure of API requests and responses ---
class APIRequest(BaseModel):
    documents: HttpUrl
    questions: List[str]

class Answer(BaseModel):
    question: str
    answer: str

class APIResponse(BaseModel):
    answers: List[Answer]

# --- Security: Handles API Key authentication ---
security_scheme = HTTPBearer()
# Reads the secret key from environment variables for security
API_KEY = os.getenv("API_KEY")

def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security_scheme)):
    """Validates the API key provided in the Authorization header."""
    if not API_KEY:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API Key not configured on the server.",
        )
    if credentials.credentials != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key",
        )

# --- API Endpoints ---
@app.get("/health", tags=["General"])
async def health_check():
    """A simple endpoint to confirm the API is running."""
    return {"status": "healthy"}


@app.post("/process-pdf", response_model=APIResponse, tags=["RAG"])
async def process_pdf(
    request: APIRequest,
    # This dependency ensures the verify_api_key function runs first
    authenticated: bool = Depends(verify_api_key)
):
    """
    The main endpoint to process a PDF and answer questions.
    It takes a PDF URL and a list of questions, then returns all answers.
    """
    try:
        # 1. Delegate all work to the RAG pipeline function
        results = process_api_request(str(request.documents), request.questions)

        # 2. Return the results in the structured Pydantic model
        return APIResponse(answers=results)

    except Exception as e:
        # Catch any unexpected errors from the pipeline and return a server error
        print(f"An internal error occurred: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An internal error occurred: {e}"
        )