from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import json
import os
from pathlib import Path

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

@dataclass
class AgentConfig:
    """Конфигурация агентов"""
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 2048
    context_window: int = 4096

@dataclass
class DataConfig:
    """Конфигурация обработки данных"""
    chunk_size: int = 1000
    min_text_length: int = 10
    max_text_length: int = 10000
    supported_formats: list = field(default_factory=lambda: ["csv", "json", "yaml", "xml"])

@dataclass
class AnalyticsConfig:
    """Конфигурация аналитики"""
    metrics_history_size: int = 100
    prediction_window: int = 5
    confidence_threshold: float = 0.8

@dataclass
class NotificationConfig:
    """Конфигурация уведомлений"""
    max_history: int = 1000
    priority_levels: list = field(default_factory=lambda: [1, 2, 3, 4, 5])
    categories: list = field(default_factory=lambda: ["info", "warning", "error", "success"])

@dataclass
class AppConfig:
    theme: str = "light"
    ollama: OllamaConfig = OllamaConfig()
    agent: AgentConfig = AgentConfig()
    data: DataConfig = DataConfig()

class ConfigManager:
    """Менеджер конфигурации"""
    config_path: str = "config/settings.json"
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    agents: AgentConfig = field(default_factory=AgentConfig)
    data: DataConfig = field(default_factory=DataConfig)
    analytics: AnalyticsConfig = field(default_factory=AnalyticsConfig)
    notifications: NotificationConfig = field(default_factory=NotificationConfig)
    
    def __post_init__(self):
        """Инициализация после создания объекта"""
        self.load_config()
        
    def load_config(self) -> None:
        """Загрузка конфигурации из файла"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                    
                # Обновляем конфигурацию из файла
                if 'ollama' in config_data:
                    self.ollama = OllamaConfig(**config_data['ollama'])
                if 'agents' in config_data:
                    self.agents = AgentConfig(**config_data['agents'])
                if 'data' in config_data:
                    self.data = DataConfig(**config_data['data'])
                if 'analytics' in config_data:
                    self.analytics = AnalyticsConfig(**config_data['analytics'])
                if 'notifications' in config_data:
                    self.notifications = NotificationConfig(**config_data['notifications'])
                    
        except Exception as e:
            print(f"Ошибка при загрузке конфигурации: {e}")
            
    def save_config(self) -> None:
        """Сохранение конфигурации в файл"""
        try:
            # Создаем директорию для конфигурации, если она не существует
            config_dir = os.path.dirname(self.config_path)
            if not os.path.exists(config_dir):
                os.makedirs(config_dir)
                
            # Преобразуем конфигурацию в словарь
            config_data = {
                'ollama': self.ollama.__dict__,
                'agents': self.agents.__dict__,
                'data': self.data.__dict__,
                'analytics': self.analytics.__dict__,
                'notifications': self.notifications.__dict__
            }
            
            # Сохраняем в файл
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=4, ensure_ascii=False)
                
        except Exception as e:
            print(f"Ошибка при сохранении конфигурации: {e}")
            
    def update_config(self, section: str, **kwargs) -> None:
        """
        Обновление конфигурации
        
        Args:
            section: Раздел конфигурации
            **kwargs: Параметры для обновления
        """
        try:
            if section == 'ollama':
                for key, value in kwargs.items():
                    setattr(self.ollama, key, value)
            elif section == 'agents':
                for key, value in kwargs.items():
                    setattr(self.agents, key, value)
            elif section == 'data':
                for key, value in kwargs.items():
                    setattr(self.data, key, value)
            elif section == 'analytics':
                for key, value in kwargs.items():
                    setattr(self.analytics, key, value)
            elif section == 'notifications':
                for key, value in kwargs.items():
                    setattr(self.notifications, key, value)
            else:
                raise ValueError(f"Неизвестный раздел конфигурации: {section}")
                
            self.save_config()
            
        except Exception as e:
            print(f"Ошибка при обновлении конфигурации: {e}")
            
    def get_config(self, section: str) -> Dict[str, Any]:
        """
        Получение конфигурации раздела
        
        Args:
            section: Раздел конфигурации
            
        Returns:
            Словарь с конфигурацией
        """
        try:
            if section == 'ollama':
                return self.ollama.__dict__
            elif section == 'agents':
                return self.agents.__dict__
            elif section == 'data':
                return self.data.__dict__
            elif section == 'analytics':
                return self.analytics.__dict__
            elif section == 'notifications':
                return self.notifications.__dict__
            else:
                raise ValueError(f"Неизвестный раздел конфигурации: {section}")
                
        except Exception as e:
            print(f"Ошибка при получении конфигурации: {e}")
            return {} 