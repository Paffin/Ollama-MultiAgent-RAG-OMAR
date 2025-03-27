from typing import Any, Dict
from .base import BaseAgent

class PlannerAgent(BaseAgent):
    """Агент для планирования и анализа запросов"""
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Обработка входных данных
        
        Args:
            input_data: Входные данные
            
        Returns:
            Результат обработки
        """
        try:
            # Проверяем валидность входных данных
            if not await self.validate(input_data):
                raise ValueError("Некорректные входные данные")
                
            # Получаем запрос
            query = input_data.get('query', '')
            
            # Генерируем план действий
            prompt = f"Проанализируй запрос и создай план действий: {query}"
            response = await self.client.generate(
                model=self.model_name,
                prompt=prompt
            )
            
            # Форматируем результат
            result = {
                'plan': response,
                'query': query,
                'status': 'success'
            }
            
            self.log_info("План действий успешно сгенерирован")
            return result
            
        except Exception as e:
            self.log_error(f"Ошибка при обработке запроса: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }
            
    async def validate(self, input_data: Dict[str, Any]) -> bool:
        """
        Валидация входных данных
        
        Args:
            input_data: Входные данные
            
        Returns:
            True если данные валидны
        """
        if not isinstance(input_data, dict):
            return False
            
        query = input_data.get('query', '')
        if not query or not isinstance(query, str):
            return False
            
        return True 