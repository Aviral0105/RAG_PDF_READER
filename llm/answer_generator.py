from openai import OpenAI
from retriever.query_retriever import load_faiss_and_metadata, retrieve_top_k_chunks

# ✅ Initialize OpenAI client
client = OpenAI()

def build_context_from_chunks(results):
    """
    Combine retrieved chunks into a single context string
    """
    context = ""
    for r in results:
        context += f"\n[From {r['source']}]\n{r['chunk']}\n"
    return context

def generate_answer_with_gpt(chat_history, query, context):
    """
    Ask GPT with retrieved context + previous chat history
    """
    # ✅ Base system prompt
    messages = [
        {"role": "system", "content": "You are a helpful assistant that ONLY answers using the provided context."}
    ]

    # ✅ Add previous chat history
    messages.extend(chat_history)

    # ✅ Add new user query with context
    messages.append({
        "role": "user",
        "content": f"Context:\n{context}\n\nQuestion: {query}\nAnswer using ONLY the context above."
    })

    # ✅ Call GPT
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # you can switch to gpt-4o if needed
        messages=messages
    )
    
    return response.choices[0].message.content

def interactive_qa():
    """
    Phase 4: Interactive Q&A with chat history
    """
    # ✅ Load FAISS & metadata
    faiss_index, chunk_metadata = load_faiss_and_metadata()
    print("\n✅ Multi-PDF QA ready! Type 'exit' to quit.\n")

    chat_history = []  # ✅ Store previous questions/answers

    while True:
        query = input("❓ Ask a question: ").strip()
        if query.lower() in ["exit", "quit", "q"]:
            print("👋 Exiting QA session.")
            break

        # ✅ Retrieve top-k chunks for this query
        results = retrieve_top_k_chunks(query, faiss_index, chunk_metadata, k=3)
        sources_used = list({r["source"] for r in results})
        print(f"\n🔍 Retrieved from: {', '.join(sources_used)}")

        # ✅ Build retrieval context
        context = build_context_from_chunks(results)

        # ✅ Generate GPT answer with chat history
        answer = generate_answer_with_gpt(chat_history, query, context)

        # ✅ Show GPT answer
        print("\n🤖 Answer:")
        print(answer)
        print("\n" + "-"*50)

        # ✅ Update chat history
        chat_history.append({"role": "user", "content": query})
        chat_history.append({"role": "assistant", "content": answer})

# ✅ Allow running Phase 4 standalone
if __name__ == "__main__":
    interactive_qa()
