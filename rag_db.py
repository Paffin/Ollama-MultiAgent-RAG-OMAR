from typing import List
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class SimpleVectorStore:
    """Векторное хранилище для семантического поиска."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Инициализирует векторное хранилище.
        
        Args:
            model_name: Имя модели для эмбеддингов
        """
        self.embedder = SentenceTransformer(model_name)
        self.index = None
        self.texts = []
        self.max_documents = 1000
        self.max_document_size = 1000000  # 1MB
    
    def add_documents(self, docs: List[str]) -> None:
        """
        Добавляет документы в индекс.
        
        Args:
            docs: Список текстовых документов
        """
        if not docs:
            return
            
        # Проверяем ограничения
        if len(self.texts) + len(docs) > self.max_documents:
            raise ValueError(f"Превышен лимит документов ({self.max_documents})")
            
        for doc in docs:
            if len(doc.encode('utf-8')) > self.max_document_size:
                raise ValueError(f"Документ превышает максимальный размер ({self.max_document_size} байт)")
        
        # Создаем эмбеддинги
        embeddings = self._create_embeddings(docs)
        
        # Добавляем в индекс
        self._add_to_index(embeddings, docs)
    
    def search(self, query: str, k: int = 3) -> List[str]:
        """
        Ищет похожие документы.
        
        Args:
            query: Поисковый запрос
            k: Количество результатов
            
        Returns:
            Список найденных документов
        """
        if not self.index:
            return []
            
        query_embedding = self._create_embeddings([query])
        distances, indices = self.index.search(query_embedding, k)
        
        return [self.texts[idx] for idx in indices[0]]
    
    def _create_embeddings(self, texts: List[str]) -> np.ndarray:
        """Создает эмбеддинги для текстов."""
        embeddings = self.embedder.encode(texts, show_progress_bar=False)
        return np.array(embeddings).astype("float32")
    
    def _add_to_index(self, embeddings: np.ndarray, docs: List[str]) -> None:
        """Добавляет эмбеддинги и документы в индекс."""
        if self.index is None:
            self._create_new_index(embeddings)
        else:
            self.index.add(embeddings)
        self.texts.extend(docs)
    
    def _create_new_index(self, embeddings: np.ndarray) -> None:
        """Создает новый индекс FAISS."""
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
    
    def clear(self) -> None:
        """Очищает хранилище."""
        self.index = None
        self.texts = []
