from typing import List, Dict, Any
import numpy as np
from utils.logger import Logger

class SimpleVectorStore:
    """Простое векторное хранилище"""
    
    def __init__(self):
        self.vectors = []
        self.metadata = []
        self.logger = Logger()
        
    def add(self, vector: np.ndarray, metadata: Dict[str, Any]) -> None:
        """
        Добавление вектора и метаданных
        
        Args:
            vector: Вектор для добавления
            metadata: Метаданные
        """
        try:
            self.vectors.append(vector)
            self.metadata.append(metadata)
            self.logger.info(f"Добавлен вектор с метаданными: {metadata}")
        except Exception as e:
            self.logger.error(f"Ошибка при добавлении вектора: {str(e)}")
            raise
            
    def search(self, query_vector: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """
        Поиск ближайших векторов
        
        Args:
            query_vector: Вектор запроса
            k: Количество результатов
            
        Returns:
            Список результатов с метаданными
        """
        try:
            if not self.vectors:
                return []
                
            # Вычисляем косинусное сходство
            similarities = [
                np.dot(query_vector, v) / (np.linalg.norm(query_vector) * np.linalg.norm(v))
                for v in self.vectors
            ]
            
            # Получаем индексы k ближайших векторов
            indices = np.argsort(similarities)[-k:][::-1]
            
            # Формируем результаты
            results = []
            for idx in indices:
                results.append({
                    'metadata': self.metadata[idx],
                    'similarity': similarities[idx]
                })
                
            return results
            
        except Exception as e:
            self.logger.error(f"Ошибка при поиске векторов: {str(e)}")
            raise 