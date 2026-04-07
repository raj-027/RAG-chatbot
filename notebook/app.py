from openai import OpenAI

# import from our modules
from data_loader import DataLoader
from embedding import EmbeddingManager
from vectorstore import VectorStore
from search import RAGRetriever


# -----------------------------
# STEP 1: Load Data
# -----------------------------
loader = DataLoader("sample.pdf")
documents = loader.load()


# -----------------------------
# STEP 2: Generate Embeddings
# -----------------------------
embedder = EmbeddingManager()
texts = [doc.page_content for doc in documents]
embeddings = embedder.generate_embeddings(texts)


# -----------------------------
# STEP 3: Store in Vector DB
# -----------------------------
vectorstore = VectorStore()
vectorstore.add_documents(documents, embeddings)


# -----------------------------
# STEP 4: Retriever
# -----------------------------
retriever = RAGRetriever(vectorstore, embedder)


# -----------------------------
# STEP 5: LLM (LM Studio)
# -----------------------------
llm = OpenAI(
    base_url="http://10.130.234.15:1234/v1",
    api_key="lm-studio"
)


# -----------------------------
# STEP 6: RAG PIPELINE
# -----------------------------
def rag_pipeline(query: str):
    results = retriever.retrieve(query, top_k=3)

    if not results:
        return "No relevant context found"

    context = "\n\n".join([doc['content'] for doc in results])

    prompt = f"""
You are a helpful assistant.
Answer ONLY from the given context.

Context:
{context}

Question: {query}

Answer:
"""

    response = llm.chat.completions.create(
        model="qwen2_5-vl-3b-abliterated-caption-it_i1",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    return response.choices[0].message.content
    # -----------------------------
# STEP 7: RUN LOOP
# -----------------------------
if __name__ == "__main__":
    print("RAG system ready! Type 'exit' to quit.")

    while True:
        query = input("Ask: ")

        if query.lower() == "exit":
            break

        answer = rag_pipeline(query)
        print("Answer:", answer, "")