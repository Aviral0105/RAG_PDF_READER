import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from retriever.query_retriever import load_faiss_and_metadata, retrieve_top_k_chunks
from openai import OpenAI

# ✅ Load retriever
faiss_index, chunk_metadata = load_faiss_and_metadata()

# ✅ OpenAI client (auto reads API key)
client = OpenAI()

def generate_answer(query, k=3):
    # 1️⃣ Retrieve top-k chunks
    results = retrieve_top_k_chunks(query, faiss_index, chunk_metadata, k=k)

    # 2️⃣ Combine chunks into context
    context_text = "\n\n".join([chunk for chunk, _ in results])

    # 3️⃣ Build prompt for GPT
    prompt = f"""
    You are a helpful assistant. Use ONLY the context below to answer the question.
    If the context is not enough, reply with "I don't know from the provided PDF."

    Context:
    {context_text}

    Question: {query}
    Answer:
    """

    # 4️⃣ Call OpenAI GPT
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # or "gpt-3.5-turbo"
        messages=[
            {"role": "system", "content": "You are a helpful PDF QA assistant."},
            {"role": "user", "content": prompt}
        ]
    )

    # 5️⃣ Extract GPT's answer
    return response.choices[0].message.content

if __name__ == "__main__":
    print("✅ Phase 4: RAG Answer Generator Ready!")
    while True:
        query = input("\n❓ Ask a question (or 'exit' to quit): ")
        if query.lower() == "exit":
            break

        answer = generate_answer(query)
        print(f"\n🤖 Answer:\n{answer}\n")
