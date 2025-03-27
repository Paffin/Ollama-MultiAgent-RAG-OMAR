from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum
from utils.logger import Logger

class NotificationType(Enum):
    """Типы уведомлений"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    SUCCESS = "success"
    DEBUG = "debug"

class NotificationPriority(Enum):
    """Приоритеты уведомлений"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

class Notification:
    """Класс уведомления"""
    
    def __init__(
        self,
        message: str,
        notification_type: NotificationType,
        priority: NotificationPriority = NotificationPriority.MEDIUM,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Инициализация уведомления
        
        Args:
            message: Текст уведомления
            notification_type: Тип уведомления
            priority: Приоритет уведомления
            metadata: Дополнительные данные
        """
        self.message = message
        self.type = notification_type
        self.priority = priority
        self.metadata = metadata or {}
        self.timestamp = datetime.now()
        self.read = False
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Преобразование уведомления в словарь
        
        Returns:
            Словарь с данными уведомления
        """
        return {
            'message': self.message,
            'type': self.type.value,
            'priority': self.priority.value,
            'metadata': self.metadata,
            'timestamp': self.timestamp.isoformat(),
            'read': self.read
        }

class NotificationSystem:
    """Система уведомлений"""
    
    def __init__(self):
        self.logger = Logger()
        self.notifications: List[Notification] = []
        
    def send_notification(
        self,
        message: str,
        notification_type: NotificationType,
        priority: NotificationPriority = NotificationPriority.MEDIUM,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Отправка уведомления
        
        Args:
            message: Текст уведомления
            notification_type: Тип уведомления
            priority: Приоритет уведомления
            metadata: Дополнительные данные
        """
        try:
            notification = Notification(
                message=message,
                notification_type=notification_type,
                priority=priority,
                metadata=metadata
            )
            self.notifications.append(notification)
            self.logger.info(f"Отправлено уведомление: {message}")
        except Exception as e:
            self.logger.error(f"Ошибка при отправке уведомления: {str(e)}")
            
    def get_notifications(
        self,
        notification_type: Optional[NotificationType] = None,
        priority: Optional[NotificationPriority] = None,
        unread_only: bool = False
    ) -> List[Notification]:
        """
        Получение уведомлений
        
        Args:
            notification_type: Фильтр по типу
            priority: Фильтр по приоритету
            unread_only: Только непрочитанные
            
        Returns:
            Список уведомлений
        """
        try:
            filtered_notifications = self.notifications
            
            if notification_type:
                filtered_notifications = [
                    n for n in filtered_notifications
                    if n.type == notification_type
                ]
                
            if priority:
                filtered_notifications = [
                    n for n in filtered_notifications
                    if n.priority == priority
                ]
                
            if unread_only:
                filtered_notifications = [
                    n for n in filtered_notifications
                    if not n.read
                ]
                
            return filtered_notifications
            
        except Exception as e:
            self.logger.error(f"Ошибка при получении уведомлений: {str(e)}")
            return []
            
    def mark_as_read(self, notification_id: int) -> bool:
        """
        Отметить уведомление как прочитанное
        
        Args:
            notification_id: ID уведомления
            
        Returns:
            True если уведомление найдено и обновлено
        """
        try:
            if 0 <= notification_id < len(self.notifications):
                self.notifications[notification_id].read = True
                return True
            return False
        except Exception as e:
            self.logger.error(f"Ошибка при отметке уведомления как прочитанного: {str(e)}")
            return False
            
    def clear_notifications(self) -> None:
        """Очистка уведомлений"""
        try:
            self.notifications.clear()
            self.logger.info("Уведомления очищены")
        except Exception as e:
            self.logger.error(f"Ошибка при очистке уведомлений: {str(e)}") 