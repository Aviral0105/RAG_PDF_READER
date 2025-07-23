import re
from openai import OpenAI
from retriever.query_retriever import load_faiss_and_metadata, retrieve_top_k_chunks

# ✅ Initialize OpenAI client
client = OpenAI()

# ✅ --- Helper: Build context ---
def build_context_from_chunks(results):
    """
    Build context string from retrieved chunks with source & page
    """
    context = ""
    for r in results:
        context += f"\n[From {r['source']} (Page {r['page']})]\n{r['chunk']}\n"
    return context


# ✅ --- Helper: Detect filters from query ---
def parse_query_filters(query, all_pdf_names):
    """
    Try to extract PDF name and page range from the query.
    Returns (pdf_name, page_range) or (None, None) if not found.
    """
    pdf_name = None
    page_range = None

    # ✅ 1. Detect PDF name from query
    for name in all_pdf_names:
        short_name = name.replace(".pdf", "")
        if short_name.lower() in query.lower():
            pdf_name = name
            break

    # ✅ 2. Detect page numbers like "page 1" or "pages 5-10"
    match_range = re.search(r"pages?\s+(\d+)\s*-\s*(\d+)", query, re.IGNORECASE)
    match_single = re.search(r"page\s+(\d+)", query, re.IGNORECASE)

    if match_range:
        start, end = int(match_range.group(1)), int(match_range.group(2))
        page_range = (start, end)
    elif match_single:
        page = int(match_single.group(1))
        page_range = (page, page)

    return pdf_name, page_range


# ✅ --- GPT Answer Generator ---
def generate_answer_with_gpt(chat_history, query, context):
    """
    Ask GPT with retrieved context + previous chat history
    """
    # System prompt
    messages = [
        {"role": "system", "content": "You are a helpful assistant that ONLY answers using the provided context."}
    ]

    # Add chat history
    messages.extend(chat_history)

    # Add user query with context
    messages.append({
        "role": "user",
        "content": f"Context:\n{context}\n\nQuestion: {query}\nAnswer using ONLY the context above."
    })

    # Call GPT
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )
    return response.choices[0].message.content


# ✅ --- Interactive Q&A ---
def interactive_qa():
    """
    Phase 4: Interactive Q&A with:
    ✅ Chat history
    ✅ Auto-detect PDF/page filters in query
    """
    # Load FAISS + metadata
    faiss_index, chunk_metadata = load_faiss_and_metadata()
    print("\n✅ Multi-PDF QA ready! Type 'exit' to quit.\n")

    # Collect available PDF names
    all_pdf_names = list({m["source"] for m in chunk_metadata})
    print(f"📂 Available PDFs: {', '.join(all_pdf_names)}")

    chat_history = []

    while True:
        query = input("\n❓ Ask a question: ").strip()
        if query.lower() in ["exit", "quit", "q"]:
            print("👋 Exiting QA session.")
            break

        # ✅ Detect filters in query
        pdf_name, page_range = parse_query_filters(query, all_pdf_names)
        if pdf_name:
            print(f"🎯 Detected PDF filter → {pdf_name}")
        if page_range:
            print(f"🎯 Detected Page filter → {page_range}")

        # ✅ Retrieve top-k chunks with optional filters
        results = retrieve_top_k_chunks(
            query,
            faiss_index,
            chunk_metadata,
            k=3,
            pdf_name=pdf_name,
            page_range=page_range
        )

        # ✅ Show where retrieved chunks came from
        if results:
            sources_used = [f"{r['source']} (p.{r['page']})" for r in results]
            print(f"\n🔍 Retrieved from: {', '.join(sources_used)}")
        else:
            print("\n⚠️ No relevant chunks found with given filters.")

        # ✅ Build GPT context & generate answer
        context = build_context_from_chunks(results)
        answer = generate_answer_with_gpt(chat_history, query, context)

        # ✅ Show GPT answer
        print("\n🤖 Answer:")
        print(answer)
        print("\n" + "-"*50)

        # ✅ Save chat history for follow-up questions
        chat_history.append({"role": "user", "content": query})
        chat_history.append({"role": "assistant", "content": answer})


# ✅ Run standalone
if __name__ == "__main__":
    interactive_qa()
