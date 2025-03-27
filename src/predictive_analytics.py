from typing import Dict, Any, List
from datetime import datetime, timedelta
from utils.logger import Logger

class PredictiveAnalytics:
    """Предсказательная аналитика"""
    
    def __init__(self):
        self.logger = Logger()
        self.historical_data = []
        
    def add_data_point(self, data: Dict[str, Any]) -> None:
        """
        Добавление точки данных
        
        Args:
            data: Данные для добавления
        """
        try:
            data['timestamp'] = datetime.now()
            self.historical_data.append(data)
            self.logger.info("Точка данных добавлена")
        except Exception as e:
            self.logger.error(f"Ошибка при добавлении точки данных: {str(e)}")
            
    def predict_performance(self, days: int = 7) -> Dict[str, Any]:
        """
        Предсказание производительности
        
        Args:
            days: Количество дней для предсказания
            
        Returns:
            Предсказание производительности
        """
        try:
            if not self.historical_data:
                return {}
                
            # Простой алгоритм предсказания на основе средних значений
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Фильтруем данные за указанный период
            period_data = [
                d for d in self.historical_data
                if start_date <= d['timestamp'] <= end_date
            ]
            
            if not period_data:
                return {}
                
            # Вычисляем средние значения
            avg_values = {}
            for key in period_data[0].keys():
                if key != 'timestamp':
                    values = [d[key] for d in period_data if key in d]
                    if values:
                        avg_values[key] = sum(values) / len(values)
                        
            prediction = {
                'timestamp': datetime.now(),
                'period_days': days,
                'predictions': avg_values
            }
            
            self.logger.info(f"Сгенерировано предсказание на {days} дней")
            return prediction
            
        except Exception as e:
            self.logger.error(f"Ошибка при генерации предсказания: {str(e)}")
            return {}
            
    def analyze_trends(self) -> Dict[str, Any]:
        """
        Анализ трендов
        
        Returns:
            Результаты анализа трендов
        """
        try:
            if not self.historical_data:
                return {}
                
            # Анализ трендов на основе последних 30 дней
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            
            period_data = [
                d for d in self.historical_data
                if start_date <= d['timestamp'] <= end_date
            ]
            
            if not period_data:
                return {}
                
            # Определяем тренды
            trends = {}
            for key in period_data[0].keys():
                if key != 'timestamp':
                    values = [d[key] for d in period_data if key in d]
                    if len(values) >= 2:
                        # Простой анализ тренда
                        if values[-1] > values[0]:
                            trends[key] = 'increasing'
                        elif values[-1] < values[0]:
                            trends[key] = 'decreasing'
                        else:
                            trends[key] = 'stable'
                            
            analysis = {
                'timestamp': datetime.now(),
                'period_days': 30,
                'trends': trends
            }
            
            self.logger.info("Анализ трендов выполнен")
            return analysis
            
        except Exception as e:
            self.logger.error(f"Ошибка при анализе трендов: {str(e)}")
            return {} 