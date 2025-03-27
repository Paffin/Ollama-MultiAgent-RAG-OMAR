class OMARError(Exception):
    """Базовый класс для всех исключений OMAR"""
    pass

class ValidationError(OMARError):
    """Ошибка валидации данных"""
    pass

class ProcessingError(OMARError):
    """Ошибка обработки данных"""
    pass

class ConfigurationError(OMARError):
    """Ошибка конфигурации"""
    pass

class CacheError(OMARError):
    """Ошибка кэширования"""
    pass

class AgentError(OMARError):
    """Ошибка работы агента"""
    pass

class AnalyticsError(OMARError):
    """Ошибка аналитики"""
    pass

class NotificationError(OMARError):
    """Ошибка уведомлений"""
    pass

class DataFormatError(OMARError):
    """Ошибка формата данных"""
    pass

class TransformationError(OMARError):
    """Ошибка трансформации данных"""
    pass

class PredictionError(OMARError):
    """Ошибка предсказания"""
    pass

class ReportError(OMARError):
    """Ошибка генерации отчета"""
    pass

class OllamaError(OMARError):
    """Ошибка работы с Ollama"""
    pass

class RAGError(OMARError):
    """Ошибка работы с RAG"""
    pass

def handle_error(error: Exception, context: str = "") -> None:
    """
    Обработка исключений
    
    Args:
        error: Исключение
        context: Контекст ошибки
    """
    from .logger import Logger
    
    logger = Logger()
    
    if isinstance(error, OMARError):
        logger.error(f"{context}: {str(error)}")
    else:
        logger.exception(f"Неожиданная ошибка в {context}: {str(error)}")
        
    # Можно добавить дополнительную логику обработки ошибок
    # Например, отправку уведомлений или сохранение в базу данных 