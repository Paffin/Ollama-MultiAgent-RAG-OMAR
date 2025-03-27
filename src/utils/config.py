from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Type, TypeVar, Union, List, Set
import json
import os
from pathlib import Path
from enum import Enum

class ConfigError(Exception):
    """Базовый класс для ошибок конфигурации"""
    pass

class ValidationError(ConfigError):
    """Ошибка валидации конфигурации"""
    pass

class LoadError(ConfigError):
    """Ошибка загрузки конфигурации"""
    pass

class SaveError(ConfigError):
    """Ошибка сохранения конфигурации"""
    pass

class ConfigSection(str, Enum):
    """Секции конфигурации"""
    OLLAMA = "ollama"
    AGENTS = "agents"
    DATA = "data"
    ANALYTICS = "analytics"
    NOTIFICATIONS = "notifications"

T = TypeVar('T')

@dataclass
class OllamaConfig:
    """Конфигурация Ollama"""
    base_url: str = "http://localhost:11434"
    timeout: int = 30
    models: Dict[str, str] = field(default_factory=lambda: {
        "planner": "mistral:7b",
        "executor": "mistral:7b",
        "critic": "mistral:7b",
        "praise": "mistral:7b",
        "arbiter": "mistral:7b"
    })

    def validate(self) -> None:
        """Валидация конфигурации"""
        if not self.base_url.startswith(('http://', 'https://')):
            raise ValidationError("base_url должен начинаться с http:// или https://")
        if self.timeout <= 0:
            raise ValidationError("timeout должен быть положительным числом")
        if not self.models:
            raise ValidationError("models не может быть пустым")

@dataclass
class AgentConfig:
    """Конфигурация агентов"""
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 2048
    context_window: int = 4096

    def validate(self) -> None:
        """Валидация конфигурации"""
        if not 0 <= self.temperature <= 2:
            raise ValidationError("temperature должен быть в диапазоне [0, 2]")
        if not 0 <= self.top_p <= 1:
            raise ValidationError("top_p должен быть в диапазоне [0, 1]")
        if self.max_tokens <= 0:
            raise ValidationError("max_tokens должен быть положительным числом")
        if self.context_window <= 0:
            raise ValidationError("context_window должен быть положительным числом")

@dataclass
class DataConfig:
    """Конфигурация обработки данных"""
    chunk_size: int = 1000
    min_text_length: int = 10
    max_text_length: int = 10000
    supported_formats: Set[str] = field(default_factory=lambda: {"csv", "json", "yaml", "xml"})

    def validate(self) -> None:
        """Валидация конфигурации"""
        if self.chunk_size <= 0:
            raise ValidationError("chunk_size должен быть положительным числом")
        if self.min_text_length < 0:
            raise ValidationError("min_text_length не может быть отрицательным")
        if self.max_text_length <= self.min_text_length:
            raise ValidationError("max_text_length должен быть больше min_text_length")
        if not self.supported_formats:
            raise ValidationError("supported_formats не может быть пустым")

@dataclass
class AnalyticsConfig:
    """Конфигурация аналитики"""
    metrics_history_size: int = 100
    prediction_window: int = 5
    confidence_threshold: float = 0.8

    def validate(self) -> None:
        """Валидация конфигурации"""
        if self.metrics_history_size <= 0:
            raise ValidationError("metrics_history_size должен быть положительным числом")
        if self.prediction_window <= 0:
            raise ValidationError("prediction_window должен быть положительным числом")
        if not 0 <= self.confidence_threshold <= 1:
            raise ValidationError("confidence_threshold должен быть в диапазоне [0, 1]")

@dataclass
class NotificationConfig:
    """Конфигурация уведомлений"""
    max_history: int = 1000
    priority_levels: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5])
    categories: List[str] = field(default_factory=lambda: ["info", "warning", "error", "success"])

    def validate(self) -> None:
        """Валидация конфигурации"""
        if self.max_history <= 0:
            raise ValidationError("max_history должен быть положительным числом")
        if not self.priority_levels:
            raise ValidationError("priority_levels не может быть пустым")
        if not self.categories:
            raise ValidationError("categories не может быть пустым")

@dataclass
class AppConfig:
    theme: str = "light"
    ollama: OllamaConfig = OllamaConfig()
    agent: AgentConfig = AgentConfig()
    data: DataConfig = DataConfig()

@dataclass
class ConfigManager:
    """Менеджер конфигурации"""
    config_path: str = "config/settings.json"
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    agents: AgentConfig = field(default_factory=AgentConfig)
    data: DataConfig = field(default_factory=DataConfig)
    analytics: AnalyticsConfig = field(default_factory=AnalyticsConfig)
    notifications: NotificationConfig = field(default_factory=NotificationConfig)
    
    def __post_init__(self) -> None:
        """Инициализация после создания объекта"""
        self.load_config()
        self._validate_all()
        
    def _validate_all(self) -> None:
        """Валидация всех конфигураций"""
        try:
            self.ollama.validate()
            self.agents.validate()
            self.data.validate()
            self.analytics.validate()
            self.notifications.validate()
        except ValidationError as e:
            raise ValidationError(f"Ошибка валидации конфигурации: {e}")
        
    def _get_config_class(self, section: ConfigSection) -> Type[T]:
        """Получение класса конфигурации для секции"""
        config_classes = {
            ConfigSection.OLLAMA: OllamaConfig,
            ConfigSection.AGENTS: AgentConfig,
            ConfigSection.DATA: DataConfig,
            ConfigSection.ANALYTICS: AnalyticsConfig,
            ConfigSection.NOTIFICATIONS: NotificationConfig
        }
        return config_classes[section]
        
    def load_config(self) -> None:
        """Загрузка конфигурации из файла"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                    
                for section in ConfigSection:
                    if section in config_data:
                        config_class = self._get_config_class(section)
                        setattr(self, section, config_class(**config_data[section]))
                        
        except json.JSONDecodeError as e:
            raise LoadError(f"Ошибка декодирования JSON: {e}")
        except Exception as e:
            raise LoadError(f"Ошибка при загрузке конфигурации: {e}")
            
    def save_config(self) -> None:
        """Сохранение конфигурации в файл"""
        try:
            config_dir = os.path.dirname(self.config_path)
            if not os.path.exists(config_dir):
                os.makedirs(config_dir)
                
            config_data = {
                section: getattr(self, section).__dict__
                for section in ConfigSection
            }
            
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=4, ensure_ascii=False)
                
        except Exception as e:
            raise SaveError(f"Ошибка при сохранении конфигурации: {e}")
            
    def update_config(self, section: ConfigSection, **kwargs) -> None:
        """
        Обновление конфигурации
        
        Args:
            section: Раздел конфигурации
            **kwargs: Параметры для обновления
            
        Raises:
            ValidationError: При ошибке валидации
            SaveError: При ошибке сохранения
        """
        try:
            config = getattr(self, section)
            for key, value in kwargs.items():
                setattr(config, key, value)
            config.validate()
            self.save_config()
            
        except ValidationError as e:
            raise ValidationError(f"Ошибка валидации при обновлении {section}: {e}")
        except SaveError as e:
            raise SaveError(f"Ошибка сохранения при обновлении {section}: {e}")
        except Exception as e:
            raise ConfigError(f"Неожиданная ошибка при обновлении {section}: {e}")
            
    def get_config(self, section: ConfigSection) -> Dict[str, Any]:
        """
        Получение конфигурации раздела
        
        Args:
            section: Раздел конфигурации
            
        Returns:
            Словарь с конфигурацией
            
        Raises:
            ConfigError: При ошибке получения конфигурации
        """
        try:
            return getattr(self, section).__dict__
        except Exception as e:
            raise ConfigError(f"Ошибка при получении конфигурации {section}: {e}") 