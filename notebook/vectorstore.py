import os
import uuid
import chromadb

class VectorStore:
    def __init__(self, collection_name: str = "pdf_docs", persist_directory: str = "./vector_db"):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self._init_store()

    def _init_store(self):
        os.makedirs(self.persist_directory, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.collection = self.client.get_or_create_collection(name=self.collection_name)

    def add_documents(self, documents, embeddings):
        ids, docs, metas, embs = [], [], [], []

        for i, (doc, emb) in enumerate(zip(documents, embeddings)):
            ids.append(f"doc_{uuid.uuid4().hex[:8]}_{i}")
            docs.append(doc.page_content)
            metas.append(doc.metadata)
            embs.append(emb.tolist())

        self.collection.add(
            ids=ids,
            documents=docs,
            metadatas=metas,
            embeddings=embs
        )