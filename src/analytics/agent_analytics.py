from typing import Dict, Any, List
from datetime import datetime
from utils.logger import Logger

class AgentUsageStats:
    """Статистика использования агента"""
    
    def __init__(self):
        self.total_calls = 0
        self.successful_calls = 0
        self.failed_calls = 0
        self.total_duration = 0
        self.avg_duration = 0
        self.last_call_time = None
        self.error_count = 0
        
    def update(self, success: bool, duration: float, error: bool = False) -> None:
        """
        Обновление статистики
        
        Args:
            success: Успешность вызова
            duration: Длительность выполнения
            error: Наличие ошибки
        """
        self.total_calls += 1
        if success:
            self.successful_calls += 1
        else:
            self.failed_calls += 1
            
        self.total_duration += duration
        self.avg_duration = self.total_duration / self.total_calls
        self.last_call_time = datetime.now()
        
        if error:
            self.error_count += 1
            
    def get_stats(self) -> Dict[str, Any]:
        """
        Получение статистики
        
        Returns:
            Словарь со статистикой
        """
        return {
            'total_calls': self.total_calls,
            'successful_calls': self.successful_calls,
            'failed_calls': self.failed_calls,
            'success_rate': self.successful_calls / self.total_calls if self.total_calls > 0 else 0,
            'avg_duration': self.avg_duration,
            'last_call_time': self.last_call_time,
            'error_count': self.error_count
        }

class AgentAnalytics:
    """Аналитика работы агентов"""
    
    def __init__(self):
        self.logger = Logger()
        self.performance_data = []
        self.error_data = []
        self.resource_usage = []
        self.agent_stats = {}
        
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
            # Обновляем статистику агента
            if agent_name not in self.agent_stats:
                self.agent_stats[agent_name] = AgentUsageStats()
            self.agent_stats[agent_name].update(success, duration)
            
            # Записываем данные о производительности
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
            # Обновляем статистику ошибок агента
            if agent_name in self.agent_stats:
                self.agent_stats[agent_name].update(False, 0, error=True)
                
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
            
    def get_agent_stats(self, agent_name: str = None) -> Dict[str, Any]:
        """
        Получение статистики агента
        
        Args:
            agent_name: Имя агента (если None, возвращает статистику всех агентов)
            
        Returns:
            Словарь со статистикой
        """
        try:
            if agent_name:
                if agent_name not in self.agent_stats:
                    return {}
                return self.agent_stats[agent_name].get_stats()
            else:
                return {name: stats.get_stats() for name, stats in self.agent_stats.items()}
                
        except Exception as e:
            self.logger.error(f"Ошибка при получении статистики агента: {str(e)}")
            return {} 