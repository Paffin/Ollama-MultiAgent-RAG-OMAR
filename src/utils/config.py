from dataclasses import dataclass
from typing import Dict, Any, Optional
import json
from pathlib import Path

@dataclass
class OllamaConfig:
    url: str = "http://localhost:11434"
    temperature: float = 0.8
    top_p: float = 0.9
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    num_ctx: int = 10000
    num_predict: int = 512

@dataclass
class AgentConfig:
    max_iterations: int = 5
    min_quality: float = 0.6
    quality_threshold: float = 0.8
    min_improvement: float = 0.1
    stagnation_threshold: float = 0.2
    improvement_threshold: float = 0.1
    max_stagnation: int = 2

@dataclass
class DataConfig:
    chunk_size: int = 1000
    min_text_length: int = 10
    max_text_length: int = 1000
    supported_formats: list = None

    def __post_init__(self):
        self.supported_formats = ['csv', 'json', 'yaml', 'xml', 'parquet']

@dataclass
class AppConfig:
    theme: str = "light"
    ollama: OllamaConfig = OllamaConfig()
    agent: AgentConfig = AgentConfig()
    data: DataConfig = DataConfig()

class ConfigManager:
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "config/settings.json"
        self.config = self._load_config()
        
    def _load_config(self) -> AppConfig:
        """Загрузка конфигурации из файла"""
        try:
            if Path(self.config_path).exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config_dict = json.load(f)
                    return self._dict_to_config(config_dict)
            return AppConfig()
        except Exception as e:
            print(f"Ошибка загрузки конфигурации: {e}")
            return AppConfig()
            
    def _dict_to_config(self, config_dict: Dict[str, Any]) -> AppConfig:
        """Преобразование словаря в объект конфигурации"""
        ollama_config = OllamaConfig(**config_dict.get('ollama', {}))
        agent_config = AgentConfig(**config_dict.get('agent', {}))
        data_config = DataConfig(**config_dict.get('data', {}))
        
        return AppConfig(
            theme=config_dict.get('theme', 'light'),
            ollama=ollama_config,
            agent=agent_config,
            data=data_config
        )
        
    def save_config(self) -> None:
        """Сохранение конфигурации в файл"""
        try:
            config_dict = {
                'theme': self.config.theme,
                'ollama': self.config.ollama.__dict__,
                'agent': self.config.agent.__dict__,
                'data': self.config.data.__dict__
            }
            
            Path(self.config_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2)
        except Exception as e:
            print(f"Ошибка сохранения конфигурации: {e}")
            
    def update_config(self, **kwargs) -> None:
        """Обновление конфигурации"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        self.save_config() 