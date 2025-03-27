from typing import Dict, Any, List
from datetime import datetime
from utils.logger import Logger

class AgentAnalytics:
    """Аналитика работы агентов"""
    
    def __init__(self):
        self.logger = Logger()
        self.performance_data = []
        self.error_data = []
        self.resource_usage = []
        
    def record_performance(self, agent_name: str, action: str, duration: float, success: bool) -> None:
        """
        Запись данных о производительности
        
        Args:
            agent_name: Имя агента
            action: Выполненное действие
            duration: Длительность выполнения
            success: Успешность выполнения
        """
        try:
            self.performance_data.append({
                'timestamp': datetime.now(),
                'agent': agent_name,
                'action': action,
                'duration': duration,
                'success': success
            })
            self.logger.info(f"Записаны данные о производительности агента {agent_name}")
        except Exception as e:
            self.logger.error(f"Ошибка при записи данных о производительности: {str(e)}")
            
    def record_error(self, agent_name: str, error_type: str, error_message: str) -> None:
        """
        Запись данных об ошибках
        
        Args:
            agent_name: Имя агента
            error_type: Тип ошибки
            error_message: Сообщение об ошибке
        """
        try:
            self.error_data.append({
                'timestamp': datetime.now(),
                'agent': agent_name,
                'error_type': error_type,
                'message': error_message
            })
            self.logger.info(f"Записаны данные об ошибке агента {agent_name}")
        except Exception as e:
            self.logger.error(f"Ошибка при записи данных об ошибке: {str(e)}")
            
    def record_resource_usage(self, agent_name: str, resource_type: str, amount: float) -> None:
        """
        Запись данных об использовании ресурсов
        
        Args:
            agent_name: Имя агента
            resource_type: Тип ресурса
            amount: Количество использованного ресурса
        """
        try:
            self.resource_usage.append({
                'timestamp': datetime.now(),
                'agent': agent_name,
                'resource_type': resource_type,
                'amount': amount
            })
            self.logger.info(f"Записаны данные об использовании ресурсов агентом {agent_name}")
        except Exception as e:
            self.logger.error(f"Ошибка при записи данных об использовании ресурсов: {str(e)}")
            
    def get_performance_report(self) -> Dict[str, Any]:
        """
        Получение отчета о производительности
        
        Returns:
            Словарь с данными о производительности
        """
        try:
            if not self.performance_data:
                return {}
                
            # Анализ данных
            total_actions = len(self.performance_data)
            successful_actions = sum(1 for d in self.performance_data if d['success'])
            avg_duration = sum(d['duration'] for d in self.performance_data) / total_actions
            
            return {
                'total_actions': total_actions,
                'success_rate': successful_actions / total_actions,
                'average_duration': avg_duration,
                'data': self.performance_data
            }
            
        except Exception as e:
            self.logger.error(f"Ошибка при формировании отчета о производительности: {str(e)}")
            return {} 