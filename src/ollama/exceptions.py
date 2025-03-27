class OllamaError(Exception):
    """Базовый класс для исключений Ollama"""
    pass

class OllamaConnectionError(OllamaError):
    """Ошибка подключения к Ollama"""
    pass

class OllamaModelError(OllamaError):
    """Ошибка работы с моделью"""
    pass

class OllamaAPIError(OllamaError):
    """Ошибка API Ollama"""
    pass

class OllamaValidationError(OllamaError):
    """Ошибка валидации данных"""
    pass 