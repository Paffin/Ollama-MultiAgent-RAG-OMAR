import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import re

class SimpleVectorStore:
    def __init__(self, dimension: int = 768):
        """
        Инициализация хранилища векторов.
        
        Args:
            dimension: Размерность векторов
        """
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.documents: List[str] = []
        self.metadata: List[Dict[str, Any]] = []
        
    def add_documents(self, documents: List[str], metadata: List[Dict[str, Any]] = None) -> None:
        """
        Добавление документов в хранилище.
        
        Args:
            documents: Список текстовых документов
            metadata: Список метаданных для каждого документа
            
        Raises:
            ValueError: Если документ пустой или содержит только пробелы
            TypeError: Если документ не является строкой
        """
        if not documents:
            return
            
        if metadata is None:
            metadata = [{} for _ in documents]
            
        if len(documents) != len(metadata):
            raise ValueError("Количество документов и метаданных должно совпадать")
            
        for i, doc in enumerate(documents):
            if not isinstance(doc, str):
                raise TypeError(f"Документ {i} должен быть строкой")
                
            # Очищаем документ от лишних пробелов
            doc = doc.strip()
            
            if not doc:
                raise ValueError(f"Документ {i} не может быть пустым")
                
            # Проверяем на минимальную длину
            if len(doc) < 10:
                raise ValueError(f"Документ {i} слишком короткий (минимум 10 символов)")
                
            # Проверяем на максимальную длину
            if len(doc) > 10000:
                raise ValueError(f"Документ {i} слишком длинный (максимум 10000 символов)")
                
            # Проверяем на наличие вредоносного кода
            if re.search(r'<script|javascript:|eval\(|exec\(|system\(|os\.|subprocess\.', doc, re.I):
                raise ValueError(f"Документ {i} содержит потенциально вредоносный код")
                
            self.documents.append(doc)
            self.metadata.append(metadata[i])
            
    def similarity_search(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """
        Поиск похожих документов.
        
        Args:
            query: Текст запроса
            k: Количество возвращаемых документов
            
        Returns:
            Список документов с метаданными
            
        Raises:
            ValueError: Если запрос пустой или хранилище пусто
        """
        if not query or not query.strip():
            raise ValueError("Запрос не может быть пустым")
            
        if not self.documents:
            raise ValueError("Хранилище документов пусто")
            
        # Ограничиваем k размером хранилища
        k = min(k, len(self.documents))
        
        # Получаем вектор запроса (заглушка)
        query_vector = np.random.rand(1, self.dimension).astype('float32')
        
        # Ищем ближайшие векторы
        distances, indices = self.index.search(query_vector, k)
        
        # Формируем результаты
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.documents):
                results.append({
                    'document': self.documents[idx],
                    'metadata': self.metadata[idx],
                    'score': float(distances[0][i])
                })
                
        return results
        
    def clear(self) -> None:
        """Очистка хранилища."""
        self.index = faiss.IndexFlatL2(self.dimension)
        self.documents = []
        self.metadata = []
        
    def __len__(self) -> int:
        """Количество документов в хранилище."""
        return len(self.documents)
