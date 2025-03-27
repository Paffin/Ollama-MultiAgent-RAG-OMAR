from typing import Dict, Any, List, Optional
from utils.logger import Logger

class TextValidator:
    """Валидатор текстовых данных"""
    
    def __init__(self, min_length: int = 1, max_length: int = 10000):
        """
        Инициализация валидатора
        
        Args:
            min_length: Минимальная длина текста
            max_length: Максимальная длина текста
        """
        self.min_length = min_length
        self.max_length = max_length
        self.logger = Logger()
        
    def validate(self, text: str) -> bool:
        """
        Валидация текста
        
        Args:
            text: Текст для проверки
            
        Returns:
            True если текст валиден
        """
        try:
            if not isinstance(text, str):
                self.logger.error("Текст должен быть строкой")
                return False
                
            if len(text) < self.min_length:
                self.logger.error(f"Текст слишком короткий (минимум {self.min_length} символов)")
                return False
                
            if len(text) > self.max_length:
                self.logger.error(f"Текст слишком длинный (максимум {self.max_length} символов)")
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка при валидации текста: {str(e)}")
            return False

class DataProcessor:
    """Обработчик данных"""
    
    def __init__(self):
        self.logger = Logger()
        self.text_validator = TextValidator()
        
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Обработка данных
        
        Args:
            data: Входные данные
            
        Returns:
            Обработанные данные
        """
        try:
            processed_data = data.copy()
            
            # Валидация текстовых полей
            for key, value in processed_data.items():
                if isinstance(value, str):
                    if not self.text_validator.validate(value):
                        self.logger.error(f"Ошибка валидации поля {key}")
                        raise ValueError(f"Некорректное значение поля {key}")
                        
            self.logger.info("Данные успешно обработаны")
            return processed_data
            
        except Exception as e:
            self.logger.error(f"Ошибка при обработке данных: {str(e)}")
            raise 