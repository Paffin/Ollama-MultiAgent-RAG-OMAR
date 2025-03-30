from typing import List, Dict, Any
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class SimpleVectorStore:
    """Векторное хранилище для семантического поиска."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Инициализирует векторное хранилище.
        
        Args:
            model_name: Имя модели для эмбеддингов
        """
        try:
            self.embedder = SentenceTransformer(model_name)
            self.index = None
            self.texts = []
            self.metadata = []
            self.max_documents = 1000
            self.max_document_size = 1000000  # 1MB
            self.last_update = None
            logger.info(f"Векторное хранилище инициализировано с моделью {model_name}")
        except Exception as e:
            logger.error(f"Ошибка при инициализации векторного хранилища: {e}")
            raise
    
    def add_documents(self, docs: List[str], metadata: List[Dict[str, Any]] = None) -> None:
        """
        Добавляет документы в индекс.
        
        Args:
            docs: Список текстовых документов
            metadata: Метаданные для каждого документа
        """
        if not docs:
            return
            
        try:
            # Проверяем ограничения
            if len(self.texts) + len(docs) > self.max_documents:
                raise ValueError(f"Превышен лимит документов ({self.max_documents})")
                
            for doc in docs:
                if len(doc.encode('utf-8')) > self.max_document_size:
                    raise ValueError(f"Документ превышает максимальный размер ({self.max_document_size} байт)")
            
            # Создаем эмбеддинги
            embeddings = self._create_embeddings(docs)
            
            # Добавляем в индекс
            self._add_to_index(embeddings, docs, metadata)
            
            self.last_update = datetime.now()
            logger.info(f"Добавлено {len(docs)} документов в хранилище")
        except Exception as e:
            logger.error(f"Ошибка при добавлении документов: {e}")
            raise
    
    def search(self, query: str, k: int = 3, metadata_filter: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Ищет похожие документы.
        
        Args:
            query: Поисковый запрос
            k: Количество результатов
            metadata_filter: Фильтр по метаданным
            
        Returns:
            Список найденных документов с метаданными
        """
        if not self.index:
            return []
            
        try:
            query_embedding = self._create_embeddings([query])
            distances, indices = self.index.search(query_embedding, k)
            
            results = []
            for i, idx in enumerate(indices[0]):
                if idx < len(self.texts):
                    result = {
                        "text": self.texts[idx],
                        "distance": float(distances[0][i]),
                        "metadata": self.metadata[idx] if idx < len(self.metadata) else {}
                    }
                    if metadata_filter:
                        if all(result["metadata"].get(k) == v for k, v in metadata_filter.items()):
                            results.append(result)
                    else:
                        results.append(result)
            
            logger.info(f"Найдено {len(results)} документов для запроса")
            return results
        except Exception as e:
            logger.error(f"Ошибка при поиске документов: {e}")
            return []
    
    def _create_embeddings(self, texts: List[str]) -> np.ndarray:
        """Создает эмбеддинги для текстов."""
        try:
            embeddings = self.embedder.encode(texts, show_progress_bar=False)
            return np.array(embeddings).astype("float32")
        except Exception as e:
            logger.error(f"Ошибка при создании эмбеддингов: {e}")
            raise
    
    def _add_to_index(self, embeddings: np.ndarray, docs: List[str], metadata: List[Dict[str, Any]] = None) -> None:
        """Добавляет эмбеддинги и документы в индекс."""
        try:
            if self.index is None:
                self._create_new_index(embeddings)
            else:
                self.index.add(embeddings)
            self.texts.extend(docs)
            if metadata:
                self.metadata.extend(metadata)
            else:
                self.metadata.extend([{} for _ in docs])
        except Exception as e:
            logger.error(f"Ошибка при добавлении в индекс: {e}")
            raise
    
    def _create_new_index(self, embeddings: np.ndarray) -> None:
        """Создает новый индекс FAISS."""
        try:
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(embeddings)
        except Exception as e:
            logger.error(f"Ошибка при создании индекса: {e}")
            raise
    
    def clear(self) -> None:
        """Очищает хранилище."""
        try:
            self.index = None
            self.texts = []
            self.metadata = []
            self.last_update = None
            logger.info("Хранилище очищено")
        except Exception as e:
            logger.error(f"Ошибка при очистке хранилища: {e}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Возвращает статистику хранилища."""
        return {
            "total_documents": len(self.texts),
            "index_size": self.index.ntotal if self.index else 0,
            "last_update": self.last_update,
            "model_name": self.embedder.__class__.__name__
        }
