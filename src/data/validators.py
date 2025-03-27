from typing import Dict, Any, List, Optional
from utils.logger import Logger

class DataValidator:
    """Валидатор данных"""
    
    def __init__(self):
        self.logger = Logger()
        self.validation_rules = {}
        
    def add_rule(self, field: str, rule: callable) -> None:
        """
        Добавление правила валидации
        
        Args:
            field: Поле для валидации
            rule: Функция валидации
        """
        try:
            self.validation_rules[field] = rule
            self.logger.info(f"Добавлено правило валидации для поля {field}")
        except Exception as e:
            self.logger.error(f"Ошибка при добавлении правила валидации: {str(e)}")
            
    def validate(self, data: Dict[str, Any]) -> bool:
        """
        Валидация данных
        
        Args:
            data: Данные для проверки
            
        Returns:
            True если данные валидны
        """
        try:
            for field, rule in self.validation_rules.items():
                if field not in data:
                    self.logger.error(f"Отсутствует обязательное поле: {field}")
                    return False
                    
                if not rule(data[field]):
                    self.logger.error(f"Ошибка валидации поля {field}")
                    return False
                    
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка при валидации данных: {str(e)}")
            return False 