class RAGRetriever:
    def __init__(self, vector_store, embedding_manager):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve(self, query, top_k=5, score_threshold=0.0):
        query_embedding = self.embedding_manager.generate_embeddings([query])[0]

        results = self.vector_store.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k
        )

        retrieved_docs = []

        if results['documents']:
            docs = results['documents'][0]
            metas = results['metadatas'][0]
            dists = results['distances'][0]
            ids = results['ids'][0]

            for i, (doc_id, doc, meta, dist) in enumerate(zip(ids, docs, metas, dists)):
                score = 1 - dist
                if score >= score_threshold:
                    retrieved_docs.append({
                        "id": doc_id,
                        "content": doc,
                        "metadata": meta,
                        "similarity_score": score
                    })

        return retrieved_docs