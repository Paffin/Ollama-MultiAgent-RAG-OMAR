from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from utils.logger import Logger

class BaseAgent(ABC):
    """Базовый класс для всех агентов"""
    
    def __init__(self, name: str, model_name: str, client: Any):
        """
        Инициализация агента
        
        Args:
            name: Имя агента
            model_name: Название модели
            client: Клиент для работы с моделью
        """
        self.name = name
        self.model_name = model_name
        self.client = client
        self.logger = Logger()
        
    @abstractmethod
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Обработка входных данных
        
        Args:
            input_data: Входные данные
            
        Returns:
            Результат обработки
        """
        pass
        
    @abstractmethod
    async def validate(self, input_data: Dict[str, Any]) -> bool:
        """
        Валидация входных данных
        
        Args:
            input_data: Входные данные
            
        Returns:
            True если данные валидны
        """
        pass
        
    def log_info(self, message: str, **kwargs) -> None:
        """Логирование информационного сообщения"""
        self.logger.info(f"[{self.name}] {message}", **kwargs)
        
    def log_error(self, message: str, **kwargs) -> None:
        """Логирование ошибки"""
        self.logger.error(f"[{self.name}] {message}", **kwargs)
        
    def log_debug(self, message: str, **kwargs) -> None:
        """Логирование отладочного сообщения"""
        self.logger.debug(f"[{self.name}] {message}", **kwargs)
        
    def log_warning(self, message: str, **kwargs) -> None:
        """Логирование предупреждения"""
        self.logger.warning(f"[{self.name}] {message}", **kwargs) 