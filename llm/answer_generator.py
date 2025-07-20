from retriever.query_retriever import load_faiss_and_metadata, retrieve_top_k_chunks
from openai import OpenAI

# ✅ Initialize GPT client
client = OpenAI()

def interactive_qa():
    """
    Interactive Question-Answering session with an indexed PDF.
    Loads FAISS, retrieves top-k chunks for each query, and uses GPT for answers.
    """
    # ✅ Load FAISS index + metadata once
    faiss_index, chunk_metadata = load_faiss_and_metadata()
    print("✅ PDF QA ready! Ask questions below (type 'exit' to quit).")

    while True:
        # ✅ Ask for user query
        query = input("\n❓ Ask a question: ")
        if query.lower() == "exit":
            print("👋 Exiting Q&A...")
            break

        # ✅ Retrieve top-k most relevant chunks
        results = retrieve_top_k_chunks(query, faiss_index, chunk_metadata, k=3)
        context_text = "\n\n".join([chunk for chunk, _ in results])

        # ✅ Build GPT prompt with context
        prompt = f"""
        You are a helpful assistant. Use ONLY the context below to answer the question.
        If the context is not enough, reply with "I don't know from the provided PDF."

        Context:
        {context_text}

        Question: {query}
        Answer:
        """

        # ✅ Get GPT response
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Or gpt-3.5-turbo if preferred
            messages=[
                {"role": "system", "content": "You are a helpful PDF QA assistant."},
                {"role": "user", "content": prompt}
            ]
        )

        # ✅ Print GPT answer
        print("\n🤖 Answer:", response.choices[0].message.content)

# ✅ If running standalone (Phase 4 only)
if __name__ == "__main__":
    interactive_qa()
