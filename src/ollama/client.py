import aiohttp
import json
from typing import Dict, Any, Optional
from utils.logger import Logger

class OllamaClient:
    """Клиент для работы с Ollama API"""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        """
        Инициализация клиента
        
        Args:
            base_url: Базовый URL Ollama API
        """
        self.base_url = base_url.rstrip('/')
        self.logger = Logger()
        
    async def generate(
        self,
        model: str,
        prompt: str,
        options: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Генерация текста с помощью модели
        
        Args:
            model: Название модели
            prompt: Текст запроса
            options: Дополнительные параметры
            
        Returns:
            Сгенерированный текст
        """
        try:
            url = f"{self.base_url}/api/generate"
            
            # Формируем payload
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False
            }
            
            # Добавляем дополнительные параметры
            if options:
                payload["options"] = options
                
            # Отправляем запрос
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"Ошибка API: {error_text}")
                        
                    data = await response.json()
                    return data.get("response", "")
                    
        except Exception as e:
            self.logger.error(f"Ошибка при генерации текста: {str(e)}")
            raise
            
    async def list_models(self) -> list:
        """
        Получение списка доступных моделей
        
        Returns:
            Список имен моделей
        """
        try:
            url = f"{self.base_url}/api/tags"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"Ошибка API: {error_text}")
                        
                    data = await response.json()
                    return [model['name'] for model in data.get("models", [])]
                    
        except Exception as e:
            self.logger.error(f"Ошибка при получении списка моделей: {str(e)}")
            raise
            
    async def pull_model(self, model: str) -> bool:
        """
        Загрузка модели
        
        Args:
            model: Название модели
            
        Returns:
            True если модель успешно загружена
        """
        try:
            url = f"{self.base_url}/api/pull"
            
            payload = {
                "name": model
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"Ошибка API: {error_text}")
                        
                    return True
                    
        except Exception as e:
            self.logger.error(f"Ошибка при загрузке модели: {str(e)}")
            raise
            
    async def delete_model(self, model: str) -> bool:
        """
        Удаление модели
        
        Args:
            model: Название модели
            
        Returns:
            True если модель успешно удалена
        """
        try:
            url = f"{self.base_url}/api/delete"
            
            payload = {
                "name": model
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.delete(url, json=payload) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"Ошибка API: {error_text}")
                        
                    return True
                    
        except Exception as e:
            self.logger.error(f"Ошибка при удалении модели: {str(e)}")
            raise 