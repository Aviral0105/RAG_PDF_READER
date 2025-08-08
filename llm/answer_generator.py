"""
llm/answer_generator.py

Builds context from retrieved chunks and calls OpenAI to generate an answer.
Also extracts clause numbers from user queries and passes them to the retriever.
"""

import re
from typing import List, Dict, Optional
from openai import OpenAI

from retriever.query_retriever import load_faiss_and_metadata, retrieve_top_k_chunks

# Initialize OpenAI client (expects OPENAI_API_KEY in env)
client = OpenAI()

# Clause detection regex for queries (captures patterns like "clause 4.2", "Clause 5.1", or "5.1")
_clause_query_regex = re.compile(
    r"(?:\b[Cc]lause\b|\b[Ss]ection\b)?\s*:?\.?\s*(\d+(?:\.\d+)+)"
)


def extract_clause_from_query(query: str) -> Optional[str]:
    """
    Extract a clause number like '4.2' from a user query if present.
    Returns the clause string or None.
    """
    if not query:
        return None
    m = _clause_query_regex.search(query)
    if m:
        return m.group(1)
    return None


def build_context_from_chunks(results: List[Dict]) -> str:
    """
    Build a concatenated context string from retrieved chunks with citation headers.
    """
    ctx = ""
    for r in results:
        src = r.get("source", "unknown")
        page = r.get("page", "N/A")
        clause = r.get("clause_number", "")
        header = f"[From {src} | Page {page} | Clause {clause}]\n"
        ctx += header + r.get("chunk", "") + "\n\n"
    return ctx


def generate_answer_with_gpt(chat_history: List[Dict], query: str, context: str) -> str:
    """
    Calls OpenAI (chat completions) with provided context and chat history.
    Keep this wrapper minimal to match your existing pattern.
    """
    # Build the messages payload
    messages = []
    # system prompt can be tuned
    messages.append({"role": "system", "content": "You are a helpful assistant that answers based on provided policy documents."})
    # include conversation history (if any)
    for m in chat_history:
        messages.append(m)
    # add the context as system or assistant content
    if context and context.strip():
        messages.append({"role": "system", "content": f"CONTEXT:\n{context}"})
    # user's current query
    messages.append({"role": "user", "content": query})

    # Call OpenAI Chat completions (using new OpenAI client pattern)
    resp = client.chat.completions.create(
        model="gpt-4o-mini",  # you can change model name as desired
        messages=messages,
        max_tokens=512,
        temperature=0.0
    )
    # Pull assistant response
    answer = resp.choices[0].message.content
    return answer


def interactive_qa():
    """
    Interactive QA loop used by your pipeline (keeps chat_history & uses clause filtering).
    This function is intended to be imported/used by other modules (it is unchanged in signature),
    but retains interactive CLI behavior if someone calls it directly.
    """
    # load index & metadata once
    index, metadata = load_faiss_and_metadata()
    chat_history: List[Dict] = []

    print("Interactive QA (type 'exit' to quit)")
    while True:
        query = input("\nEnter your question: ").strip()
        if not query:
            continue
        if query.lower() in ("exit", "quit"):
            break

        # Try to extract clause number from query
        clause = extract_clause_from_query(query)
        if clause:
            print(f"→ Detected clause filter: {clause}")

        # Retrieve chunks (pass clause filter if detected)
        results = retrieve_top_k_chunks(query, index, metadata, k=5, clause_number=clause, search_k=50)

        if results:
            sources_used = [f"{r['source']} (p.{r['page']}) [clause {r['clause_number']}]" for r in results]
            print(f"\n🔍 Retrieved from: {', '.join(sources_used)}")
        else:
            print("\n⚠️ No relevant chunks found with the given filters.")

        # Build context & generate answer
        context = build_context_from_chunks(results)
        answer = generate_answer_with_gpt(chat_history, query, context)

        # Output
        print("\n🤖 Answer:")
        print(answer)
        print("\n" + "-"*50)

        # Save to chat history
        chat_history.append({"role": "user", "content": query})
        chat_history.append({"role": "assistant", "content": answer})
