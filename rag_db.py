# rag_db.py

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class SimpleVectorStore:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.embedder = SentenceTransformer(model_name)
        self.index = None
        self.texts = []
    
    def add_documents(self, docs: list):
        """
        Добавляет документы (строки) в индекс FAISS
        """
        emb = self.embedder.encode(docs, show_progress_bar=False)
        emb = np.array(emb).astype("float32")

        if self.index is None:
            dim = emb.shape[1]
            self.index = faiss.IndexFlatL2(dim)
            self.index.add(emb)
            self.texts.extend(docs)
        else:
            self.index.add(emb)
            self.texts.extend(docs)
    
    def search(self, query: str, k=3):
        """
        Возвращает top-k ближайших документов
        """
        if self.index is None:
            return []
        q = self.embedder.encode([query], show_progress_bar=False)
        q = np.array(q).astype("float32")
        D, I = self.index.search(q, k)
        results = []
        for idx in I[0]:
            results.append(self.texts[idx])
        return results
