from typing import Dict, Any, List
from datetime import datetime
from utils.logger import Logger

class NotificationSystem:
    """Система уведомлений"""
    
    def __init__(self):
        self.logger = Logger()
        self.notifications = []
        
    def send_notification(self, message: str, level: str = "info") -> None:
        """
        Отправка уведомления
        
        Args:
            message: Текст уведомления
            level: Уровень важности
        """
        try:
            notification = {
                'timestamp': datetime.now(),
                'message': message,
                'level': level
            }
            self.notifications.append(notification)
            self.logger.info(f"Отправлено уведомление: {message}")
        except Exception as e:
            self.logger.error(f"Ошибка при отправке уведомления: {str(e)}")
            
    def get_notifications(self, level: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Получение уведомлений
        
        Args:
            level: Фильтр по уровню важности
            
        Returns:
            Список уведомлений
        """
        try:
            if level:
                return [n for n in self.notifications if n['level'] == level]
            return self.notifications
        except Exception as e:
            self.logger.error(f"Ошибка при получении уведомлений: {str(e)}")
            return []
            
    def clear_notifications(self) -> None:
        """Очистка уведомлений"""
        try:
            self.notifications.clear()
            self.logger.info("Уведомления очищены")
        except Exception as e:
            self.logger.error(f"Ошибка при очистке уведомлений: {str(e)}") 