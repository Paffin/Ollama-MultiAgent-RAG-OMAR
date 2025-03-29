import json
import os
from typing import Dict, Optional

class Settings:
    def __init__(self, settings_file: str = "agent_settings.json"):
        self.settings_file = settings_file
        self.settings = self._load_settings()

    def _load_settings(self) -> Dict:
        """Загрузка настроек из файла"""
        if os.path.exists(self.settings_file):
            try:
                with open(self.settings_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Ошибка при загрузке настроек: {e}")
                return self._get_default_settings()
        return self._get_default_settings()

    def _get_default_settings(self) -> Dict:
        """Получение настроек по умолчанию"""
        return {
            "models": {
                "planner": "mistral",
                "executor": "mistral",
                "critic": "mistral",
                "praise": "mistral",
                "arbiter": "mistral"
            },
            "ollama": {
                "temperature": 0.8,
                "top_p": 0.9,
                "presence_penalty": 0.0,
                "frequency_penalty": 0.0,
                "num_ctx": 4096,
                "num_predict": 4096,
                "max_iterations": 2
            }
        }

    def save_settings(self):
        """Сохранение настроек в файл"""
        try:
            with open(self.settings_file, 'w', encoding='utf-8') as f:
                json.dump(self.settings, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"Ошибка при сохранении настроек: {e}")

    def get_model(self, agent_name: str) -> str:
        """Получение модели для агента"""
        return self.settings["models"].get(agent_name, "mistral")

    def set_model(self, agent_name: str, model: str):
        """Установка модели для агента"""
        self.settings["models"][agent_name] = model
        self.save_settings()

    def get_ollama_settings(self) -> Dict:
        """Получение настроек Ollama"""
        return self.settings["ollama"]

    def update_ollama_settings(self, settings: Dict):
        """Обновление настроек Ollama"""
        self.settings["ollama"].update(settings)
        self.save_settings()

    def get_max_iterations(self) -> int:
        """Получение максимального количества итераций"""
        return self.settings["ollama"].get("max_iterations", 2)

    def set_max_iterations(self, value: int):
        """Установка максимального количества итераций"""
        self.settings["ollama"]["max_iterations"] = value
        self.save_settings() 