import fitz  # PyMuPDF
from pdfminer.high_level import extract_text as pdfminer_extract_text
import re
import unicodedata
import tiktoken


def extract_text_from_pdf(pdf_path: str, method: str = "pymupdf") -> str:
    """
    Extracts text from a PDF file using the specified method.
    
    method:
        - "pymupdf"  -> fastest, recommended for QA
        - "pdfminer" -> more layout-aware but slower
    """
    text = ""
    
    if method == "pymupdf":
        doc = fitz.open(pdf_path)
        for page in doc:
            text += page.get_text()
        doc.close()
    
    elif method == "pdfminer":
        text = pdfminer_extract_text(pdf_path)
    
    else:
        raise ValueError("Invalid method. Use 'pymupdf' or 'pdfminer'.")
    
    return text


def clean_pdf_text(text: str) -> str:
    """
    Cleans and normalizes extracted PDF text.
    - Fix unicode
    - Remove hyphenation at line breaks
    - Merge single newlines into spaces
    - Preserve paragraph breaks
    - Remove extra spaces
    """
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"-\n", "", text)          # merge hyphenated words
    text = re.sub(r"\n{2,}", "\n\n", text)   # keep paragraph breaks
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)  # merge single line breaks
    text = re.sub(r"\s+", " ", text)         # remove extra spaces
    return text.strip()


def chunk_text_by_tokens(text: str, chunk_size: int = 512, overlap: int = 50):
    """
    Splits cleaned text into overlapping token chunks.
    
    chunk_size: max tokens per chunk
    overlap:    number of tokens overlapped for context
    """
    tokenizer = tiktoken.get_encoding("cl100k_base")  # GPT tokenizer
    tokens = tokenizer.encode(text)
    
    chunks = []
    start = 0
    
    while start < len(tokens):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_tokens)
        chunks.append(chunk_text)
        start += chunk_size - overlap  # sliding window with overlap
    
    return chunks


if __name__ == "__main__":
    # 🧪 Test with a sample PDF
    sample_pdf = "data/Document_1.pdf"  # put your PDF in data/
    
    # 1️⃣ Extract text
    raw_text = extract_text_from_pdf(sample_pdf, method="pymupdf")
    print(f"\n[RAW TEXT SAMPLE]\n{raw_text[:500]}\n")
    
    # 2️⃣ Clean text
    cleaned_text = clean_pdf_text(raw_text)
    print(f"\n[CLEANED TEXT SAMPLE]\n{cleaned_text[:500]}\n")
    
    # 3️⃣ Chunk text
    chunks = chunk_text_by_tokens(cleaned_text, chunk_size=512, overlap=50)
    print(f"✅ Total Chunks: {len(chunks)}")
    
    # Show first 2 chunks
    for i, chunk in enumerate(chunks[:2]):
        print(f"\n--- Chunk {i+1} ---\n{chunk[:300]}...\n")
