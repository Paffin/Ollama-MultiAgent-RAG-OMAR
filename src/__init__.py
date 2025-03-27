"""
OMAR - MultiAgent System with RAG and Analytics
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .utils.config import ConfigManager
from .utils.logger import Logger
from .utils.cache import Cache
from .utils.validators import DataValidator, ValidationRule, ValidationResult
from .utils.exceptions import (
    OMARError,
    ValidationError,
    ProcessingError,
    ConfigurationError,
    CacheError,
    AgentError,
    AnalyticsError,
    NotificationError,
    DataFormatError,
    TransformationError,
    PredictionError,
    ReportError,
    OllamaError,
    RAGError,
    handle_error
) 