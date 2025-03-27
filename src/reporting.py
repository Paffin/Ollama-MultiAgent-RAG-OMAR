from typing import Dict, Any, List
from datetime import datetime
from utils.logger import Logger

class ReportGenerator:
    """Генератор отчетов"""
    
    def __init__(self, analytics):
        self.logger = Logger()
        self.analytics = analytics
        
    def generate_performance_report(self) -> Dict[str, Any]:
        """
        Генерация отчета о производительности
        
        Returns:
            Отчет о производительности
        """
        try:
            performance_data = self.analytics.get_performance_report()
            
            report = {
                'timestamp': datetime.now(),
                'type': 'performance',
                'data': performance_data
            }
            
            self.logger.info("Отчет о производительности сгенерирован")
            return report
            
        except Exception as e:
            self.logger.error(f"Ошибка при генерации отчета о производительности: {str(e)}")
            return {}
            
    def generate_error_report(self) -> Dict[str, Any]:
        """
        Генерация отчета об ошибках
        
        Returns:
            Отчет об ошибках
        """
        try:
            error_data = self.analytics.error_data
            
            report = {
                'timestamp': datetime.now(),
                'type': 'errors',
                'data': error_data
            }
            
            self.logger.info("Отчет об ошибках сгенерирован")
            return report
            
        except Exception as e:
            self.logger.error(f"Ошибка при генерации отчета об ошибках: {str(e)}")
            return {}
            
    def generate_resource_report(self) -> Dict[str, Any]:
        """
        Генерация отчета об использовании ресурсов
        
        Returns:
            Отчет об использовании ресурсов
        """
        try:
            resource_data = self.analytics.resource_usage
            
            report = {
                'timestamp': datetime.now(),
                'type': 'resources',
                'data': resource_data
            }
            
            self.logger.info("Отчет об использовании ресурсов сгенерирован")
            return report
            
        except Exception as e:
            self.logger.error(f"Ошибка при генерации отчета об использовании ресурсов: {str(e)}")
            return {} 