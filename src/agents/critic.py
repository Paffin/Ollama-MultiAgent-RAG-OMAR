from typing import Any, Dict
from .base import BaseAgent

class CriticAgent(BaseAgent):
    """Агент для анализа качества ответов"""
    
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
                
            # Получаем результат выполнения
            execution_result = input_data.get('execution_result', '')
            
            # Анализируем качество
            prompt = f"Проанализируй качество следующего ответа и укажи возможные улучшения: {execution_result}"
            response = await self.client.generate(
                model=self.model_name,
                prompt=prompt
            )
            
            # Форматируем результат
            result = {
                'critique': response,
                'execution_result': execution_result,
                'status': 'success'
            }
            
            self.log_info("Анализ качества выполнен")
            return result
            
        except Exception as e:
            self.log_error(f"Ошибка при анализе качества: {str(e)}")
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
            
        execution_result = input_data.get('execution_result', '')
        if not execution_result or not isinstance(execution_result, str):
            return False
            
        return True 