from typing import Any, Optional, Callable
import hashlib
import json
from pathlib import Path
from datetime import datetime, timedelta
from functools import wraps
import pickle

class Cache:
    def __init__(self, cache_dir: str = "cache", ttl: int = 3600):
        """
        Инициализация кэша
        
        Args:
            cache_dir: Директория для кэша
            ttl: Время жизни кэша в секундах
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl = ttl
        
    def _get_cache_key(self, func: Callable, args: tuple, kwargs: dict) -> str:
        """Генерация ключа кэша"""
        # Создаем строковое представление аргументов
        args_str = json.dumps(args, sort_keys=True)
        kwargs_str = json.dumps(kwargs, sort_keys=True)
        
        # Создаем уникальный ключ
        key = f"{func.__name__}_{args_str}_{kwargs_str}"
        return hashlib.md5(key.encode()).hexdigest()
        
    def _get_cache_path(self, key: str) -> Path:
        """Получение пути к файлу кэша"""
        return self.cache_dir / f"{key}.cache"
        
    def get(self, key: str) -> Optional[Any]:
        """
        Получение значения из кэша
        
        Args:
            key: Ключ кэша
            
        Returns:
            Значение из кэша или None
        """
        cache_path = self._get_cache_path(key)
        if not cache_path.exists():
            return None
            
        try:
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)
                
            # Проверяем время жизни кэша
            if datetime.now() - cache_data['timestamp'] > timedelta(seconds=self.ttl):
                self.delete(key)
                return None
                
            return cache_data['value']
            
        except Exception:
            return None
            
    def set(self, key: str, value: Any) -> bool:
        """
        Сохранение значения в кэш
        
        Args:
            key: Ключ кэша
            value: Значение для кэширования
            
        Returns:
            True если успешно, False в противном случае
        """
        try:
            cache_data = {
                'value': value,
                'timestamp': datetime.now()
            }
            
            cache_path = self._get_cache_path(key)
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
                
            return True
            
        except Exception:
            return False
            
    def delete(self, key: str) -> bool:
        """
        Удаление значения из кэша
        
        Args:
            key: Ключ кэша
            
        Returns:
            True если успешно, False в противном случае
        """
        try:
            cache_path = self._get_cache_path(key)
            if cache_path.exists():
                cache_path.unlink()
            return True
        except Exception:
            return False
            
    def clear(self) -> bool:
        """
        Очистка всего кэша
        
        Returns:
            True если успешно, False в противном случае
        """
        try:
            for cache_file in self.cache_dir.glob("*.cache"):
                cache_file.unlink()
            return True
        except Exception:
            return False
            
    def cached(self, ttl: Optional[int] = None):
        """
        Декоратор для кэширования результатов функции
        
        Args:
            ttl: Время жизни кэша в секундах (опционально)
        """
        def decorator(func: Callable):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Используем переданное время жизни или значение по умолчанию
                cache_ttl = ttl if ttl is not None else self.ttl
                
                # Создаем временный кэш с указанным временем жизни
                temp_cache = Cache(self.cache_dir, cache_ttl)
                
                # Генерируем ключ кэша
                cache_key = self._get_cache_key(func, args, kwargs)
                
                # Пытаемся получить значение из кэша
                cached_value = temp_cache.get(cache_key)
                if cached_value is not None:
                    return cached_value
                    
                # Если значения нет в кэше, выполняем функцию
                result = func(*args, **kwargs)
                
                # Сохраняем результат в кэш
                temp_cache.set(cache_key, result)
                
                return result
            return wrapper
        return decorator 