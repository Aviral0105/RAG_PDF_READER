from openai import OpenAI
from retriever.query_retriever import load_faiss_and_metadata, retrieve_top_k_chunks

# ✅ Initialize OpenAI client
client = OpenAI()

def build_context_from_chunks(results):
    """
    Build a context string from retrieved chunks for GPT.
    """
    context = ""
    for r in results:
        context += f"\n[From {r['source']}]\n{r['chunk']}\n"
    return context

def generate_answer_with_gpt(query, context):
    """
    Ask GPT with retrieved context.
    """
    messages = [
        {"role": "system", "content": "You are a helpful assistant that answers based on the provided context."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}\nAnswer using ONLY the context."}
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )
    
    return response.choices[0].message.content

def interactive_qa():
    """
    Phase 4: Interactive Q&A using multi-PDF retrieval.
    """
    # ✅ Load FAISS & metadata
    faiss_index, chunk_metadata = load_faiss_and_metadata()

    print("\n✅ Multi-PDF QA ready! Type 'exit' to quit.\n")
    
    while True:
        query = input("❓ Ask a question: ").strip()
        if query.lower() in ["exit", "quit", "q"]:
            print("👋 Exiting QA session.")
            break
        
        # ✅ Retrieve top-k chunks with source info
        results = retrieve_top_k_chunks(query, faiss_index, chunk_metadata, k=3)
        
        # ✅ Show where the info is coming from
        sources_used = list({r["source"] for r in results})
        print(f"\n🔍 Retrieved from: {', '.join(sources_used)}")

        # ✅ Build context for GPT
        context = build_context_from_chunks(results)

        # ✅ Get GPT answer
        answer = generate_answer_with_gpt(query, context)

        # ✅ Show final answer
        print("\n🤖 Answer:")
        print(answer)
        print("\n" + "-"*50)

if __name__ == "__main__":
    interactive_qa()
