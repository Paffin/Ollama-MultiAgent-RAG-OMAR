from typing import Any, Dict
from .base import BaseAgent

class ArbiterAgent(BaseAgent):
    """Агент для принятия решений о доработке"""
    
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
                
            # Получаем все результаты
            execution_result = input_data.get('execution_result', '')
            critique = input_data.get('critique', '')
            praise = input_data.get('praise', '')
            
            # Принимаем решение о доработке
            prompt = f"""На основе следующей информации прими решение о необходимости доработки:
            Результат выполнения: {execution_result}
            Критика: {critique}
            Сильные стороны: {praise}
            
            Решение должно быть в формате:
            - Необходима доработка: да/нет
            - Причина: краткое описание
            - Рекомендации: список конкретных действий"""
            
            response = await self.client.generate(
                model=self.model_name,
                prompt=prompt
            )
            
            # Форматируем результат
            result = {
                'decision': response,
                'execution_result': execution_result,
                'critique': critique,
                'praise': praise,
                'status': 'success'
            }
            
            self.log_info("Решение о доработке принято")
            return result
            
        except Exception as e:
            self.log_error(f"Ошибка при принятии решения: {str(e)}")
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
        praise = input_data.get('praise', '')
        
        if not execution_result or not isinstance(execution_result, str):
            return False
            
        if not critique or not isinstance(critique, str):
            return False
            
        if not praise or not isinstance(praise, str):
            return False
            
        return True