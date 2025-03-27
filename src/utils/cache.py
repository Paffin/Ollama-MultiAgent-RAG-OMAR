from typing import Any, Optional, Dict
import time
from utils.logger import Logger

class Cache:
    """Класс для кэширования данных"""
    
    def __init__(self, ttl_seconds: int = 3600, max_size_mb: int = 100):
        """
        Инициализация кэша
        
        Args:
            ttl_seconds: Время жизни кэша в секундах
            max_size_mb: Максимальный размер кэша в МБ
        """
        self.logger = Logger()
        self.ttl = ttl_seconds
        self.max_size = max_size_mb * 1024 * 1024  # Конвертируем в байты
        self.cache: Dict[str, Dict[str, Any]] = {}
        
    def get(self, key: str) -> Optional[Any]:
        """
        Получение значения из кэша
        
        Args:
            key: Ключ кэша
            
        Returns:
            Значение из кэша или None
        """
        try:
            if key not in self.cache:
                return None
                
            cache_entry = self.cache[key]
            if time.time() - cache_entry['timestamp'] > self.ttl:
                del self.cache[key]
                return None
                
            return cache_entry['value']
            
        except Exception as e:
            self.logger.error(f"Ошибка при получении значения из кэша: {str(e)}")
            return None
            
    def set(self, key: str, value: Any) -> bool:
        """
        Установка значения в кэш
        
        Args:
            key: Ключ кэша
            value: Значение для кэширования
            
        Returns:
            True если значение установлено
        """
        try:
            # Проверяем размер кэша
            if len(self.cache) >= self.max_size:
                self._cleanup()
                
            self.cache[key] = {
                'value': value,
                'timestamp': time.time()
            }
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка при установке значения в кэш: {str(e)}")
            return False
            
    def delete(self, key: str) -> bool:
        """
        Удаление значения из кэша
        
        Args:
            key: Ключ кэша
            
        Returns:
            True если значение удалено
        """
        try:
            if key in self.cache:
                del self.cache[key]
                return True
            return False
            
        except Exception as e:
            self.logger.error(f"Ошибка при удалении значения из кэша: {str(e)}")
            return False
            
    def clear(self) -> None:
        """Очистка кэша"""
        try:
            self.cache.clear()
            self.logger.info("Кэш очищен")
        except Exception as e:
            self.logger.error(f"Ошибка при очистке кэша: {str(e)}")
            
    def _cleanup(self) -> None:
        """Очистка устаревших значений"""
        try:
            current_time = time.time()
            expired_keys = [
                key for key, entry in self.cache.items()
                if current_time - entry['timestamp'] > self.ttl
            ]
            
            for key in expired_keys:
                del self.cache[key]
                
            self.logger.info(f"Удалено {len(expired_keys)} устаревших записей из кэша")
            
        except Exception as e:
            self.logger.error(f"Ошибка при очистке устаревших значений: {str(e)}") 