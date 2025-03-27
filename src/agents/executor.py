from typing import Any, Dict
from .base import BaseAgent

class ExecutorAgent(BaseAgent):
    """Агент для выполнения инструкций"""
    
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
                
            # Получаем план действий
            plan = input_data.get('plan', '')
            
            # Выполняем инструкции
            prompt = f"Выполни следующие инструкции: {plan}"
            response = await self.client.generate(
                model=self.model_name,
                prompt=prompt
            )
            
            # Форматируем результат
            result = {
                'execution_result': response,
                'plan': plan,
                'status': 'success'
            }
            
            self.log_info("Инструкции успешно выполнены")
            return result
            
        except Exception as e:
            self.log_error(f"Ошибка при выполнении инструкций: {str(e)}")
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
            
        plan = input_data.get('plan', '')
        if not plan or not isinstance(plan, str):
            return False
            
        return True 