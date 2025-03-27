from typing import Any, Dict
from .base import BaseAgent

class PraiseAgent(BaseAgent):
    """Агент для оценки сильных сторон"""
    
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
                
            # Получаем результат выполнения и критику
            execution_result = input_data.get('execution_result', '')
            critique = input_data.get('critique', '')
            
            # Оцениваем сильные стороны
            prompt = f"Оцени сильные стороны следующего ответа, учитывая критику: {execution_result}\nКритика: {critique}"
            response = await self.client.generate(
                model=self.model_name,
                prompt=prompt
            )
            
            # Форматируем результат
            result = {
                'praise': response,
                'execution_result': execution_result,
                'critique': critique,
                'status': 'success'
            }
            
            self.log_info("Оценка сильных сторон выполнена")
            return result
            
        except Exception as e:
            self.log_error(f"Ошибка при оценке сильных сторон: {str(e)}")
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
        critique = input_data.get('critique', '')
        
        if not execution_result or not isinstance(execution_result, str):
            return False
            
        if not critique or not isinstance(critique, str):
            return False
            
        return True 