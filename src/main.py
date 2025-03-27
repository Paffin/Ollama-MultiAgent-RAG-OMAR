import sys
import asyncio
from pathlib import Path
import requests
from typing import Dict, Any
import time

# Настройка для Windows
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# Подключаем nest_asyncio
import nest_asyncio
nest_asyncio.apply()

# Импорты из нашего пакета
from utils.config import ConfigManager
from utils.logger import Logger
from utils.exceptions import handle_error
from agents import (
    PlannerAgent,
    ExecutorAgent,
    CriticAgent,
    PraiseAgent,
    ArbiterAgent
)
from ollama.client import OllamaClient
from rag_db import SimpleVectorStore
from prompts import (
    PLANNER_PROMPT,
    EXECUTOR_PROMPT,
    CRITIC_PROMPT,
    PRAISE_PROMPT,
    ARBITER_PROMPT
)
from analytics import AgentAnalytics
from data import DataProcessor, DataValidator, DataPreprocessor
from notifications import NotificationSystem
from reporting import ReportGenerator
from predictive_analytics import PredictiveAnalytics
from transformations import StreamingProcessor

def check_ollama_server(url: str, max_retries: int = 3, timeout: int = 5) -> bool:
    """
    Проверка доступности сервера Ollama.
    
    Args:
        url: URL сервера Ollama
        max_retries: Максимальное количество попыток
        timeout: Таймаут запроса в секундах
        
    Returns:
        bool: True если сервер доступен
    """
    for attempt in range(max_retries):
        try:
            response = requests.get(f"{url}/api/version", timeout=timeout)
            response.raise_for_status()
            return True
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                return False
            time.sleep(1)
    return False

def validate_config(config: Dict[str, Any]) -> bool:
    """
    Валидация конфигурации.
    
    Args:
        config: Конфигурация для проверки
        
    Returns:
        bool: True если конфигурация валидна
    """
    required_fields = ['ollama', 'agents', 'data', 'analytics']
    for field in required_fields:
        if field not in config:
            return False
    return True

def init_systems():
    """Инициализация всех систем"""
    try:
        # Загрузка конфигурации
        config = ConfigManager()
        
        # Валидация конфигурации
        if not validate_config(config.config):
            raise ValueError("Некорректная конфигурация")
        
        # Инициализация логгера
        logger = Logger()
        logger.info("Инициализация систем")
        
        # Проверка доступности сервера Ollama
        ollama_url = config.config.ollama.url
        if not check_ollama_server(ollama_url):
            raise ConnectionError(f"Сервер Ollama недоступен по адресу: {ollama_url}")
        
        # Инициализация клиента Ollama
        try:
            ollama_client = OllamaClient(ollama_url)
        except Exception as e:
            raise ConnectionError(f"Ошибка при подключении к Ollama: {e}")
        
        # Инициализация хранилища векторов
        try:
            vector_store = SimpleVectorStore()
        except Exception as e:
            raise RuntimeError(f"Ошибка при инициализации хранилища векторов: {e}")
        
        # Инициализация агентов
        agents = {}
        agent_configs = {
            "planner": (PlannerAgent, "Planner"),
            "executor": (ExecutorAgent, "Executor"),
            "critic": (CriticAgent, "Critic"),
            "praise": (PraiseAgent, "Praise"),
            "arbiter": (ArbiterAgent, "Arbiter")
        }
        
        for agent_type, (agent_class, name) in agent_configs.items():
            try:
                agents[agent_type] = agent_class(
                    name=name,
                    model_name=config.config.agents.get(agent_type, {}).get('model', 'llama2'),
                    client=ollama_client
                )
            except Exception as e:
                raise RuntimeError(f"Ошибка при инициализации агента {name}: {e}")
        
        # Инициализация систем обработки данных
        try:
            data_processor = DataProcessor()
            data_validator = DataValidator()
            data_preprocessor = DataPreprocessor()
            streaming_processor = StreamingProcessor()
        except Exception as e:
            raise RuntimeError(f"Ошибка при инициализации систем обработки данных: {e}")
        
        # Инициализация аналитики
        try:
            agent_analytics = AgentAnalytics()
        except Exception as e:
            raise RuntimeError(f"Ошибка при инициализации аналитики: {e}")
        
        # Инициализация системы уведомлений
        try:
            notification_system = NotificationSystem()
        except Exception as e:
            raise RuntimeError(f"Ошибка при инициализации системы уведомлений: {e}")
        
        # Инициализация генератора отчетов
        try:
            report_generator = ReportGenerator(agent_analytics)
        except Exception as e:
            raise RuntimeError(f"Ошибка при инициализации генератора отчетов: {e}")
        
        # Инициализация предсказательной аналитики
        try:
            predictive_analytics = PredictiveAnalytics()
        except Exception as e:
            raise RuntimeError(f"Ошибка при инициализации предсказательной аналитики: {e}")
        
        logger.info("Все системы успешно инициализированы")
        
        return {
            "config": config,
            "logger": logger,
            "ollama_client": ollama_client,
            "vector_store": vector_store,
            "agents": agents,
            "data_processor": data_processor,
            "data_validator": data_validator,
            "data_preprocessor": data_preprocessor,
            "streaming_processor": streaming_processor,
            "agent_analytics": agent_analytics,
            "notification_system": notification_system,
            "report_generator": report_generator,
            "predictive_analytics": predictive_analytics
        }
        
    except Exception as e:
        handle_error(e, "Инициализация систем")
        raise

def main():
    """Главная функция приложения"""
    try:
        # Инициализация систем
        systems = init_systems()
        
        # Импорт и запуск Streamlit приложения
        import streamlit_app
        streamlit_app.run(systems)
        
    except Exception as e:
        handle_error(e, "Запуск приложения")
        sys.exit(1)

if __name__ == "__main__":
    main() 