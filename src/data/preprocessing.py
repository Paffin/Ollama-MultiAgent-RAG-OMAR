from typing import Dict, Any, List, Optional
from utils.logger import Logger

class DataPreprocessor:
    """Предварительная обработка данных"""
    
    def __init__(self):
        self.logger = Logger()
        self.preprocessing_rules = {}
        
    def add_rule(self, field: str, rule: callable) -> None:
        """
        Добавление правила предварительной обработки
        
        Args:
            field: Поле для обработки
            rule: Функция обработки
        """
        try:
            self.preprocessing_rules[field] = rule
            self.logger.info(f"Добавлено правило предварительной обработки для поля {field}")
        except Exception as e:
            self.logger.error(f"Ошибка при добавлении правила предварительной обработки: {str(e)}")
            
    def preprocess(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Предварительная обработка данных
        
        Args:
            data: Данные для обработки
            
        Returns:
            Обработанные данные
        """
        try:
            processed_data = data.copy()
            
            for field, rule in self.preprocessing_rules.items():
                if field in processed_data:
                    processed_data[field] = rule(processed_data[field])
                    
            self.logger.info("Данные успешно предварительно обработаны")
            return processed_data
            
        except Exception as e:
            self.logger.error(f"Ошибка при предварительной обработке данных: {str(e)}")
            raise 