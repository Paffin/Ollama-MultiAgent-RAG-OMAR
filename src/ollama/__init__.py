"""
Клиент Ollama для OMAR
"""

from .client import OllamaClient
from .models import OllamaModel
from .exceptions import OllamaError

__all__ = [
    'OllamaClient',
    'OllamaModel',
    'OllamaError'
] 