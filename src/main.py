import sys
import asyncio
from pathlib import Path

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
from ollama_client import OllamaClient
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

def init_systems():
    """Инициализация всех систем"""
    try:
        # Загрузка конфигурации
        config = ConfigManager()
        
        # Инициализация логгера
        logger = Logger()
        logger.info("Инициализация систем")
        
        # Инициализация клиента Ollama
        ollama_client = OllamaClient(config.config.ollama.url)
        
        # Инициализация хранилища векторов
        vector_store = SimpleVectorStore()
        
        # Инициализация агентов
        agents = {
            "planner": PlannerAgent(
                name="Planner",
                model_name="llama2",
                client=ollama_client
            ),
            "executor": ExecutorAgent(
                name="Executor",
                model_name="llama2",
                client=ollama_client
            ),
            "critic": CriticAgent(
                name="Critic",
                model_name="llama2",
                client=ollama_client
            ),
            "praise": PraiseAgent(
                name="Praise",
                model_name="llama2",
                client=ollama_client
            ),
            "arbiter": ArbiterAgent(
                name="Arbiter",
                model_name="llama2",
                client=ollama_client
            )
        }
        
        # Инициализация систем обработки данных
        data_processor = DataProcessor()
        data_validator = DataValidator()
        data_preprocessor = DataPreprocessor()
        streaming_processor = StreamingProcessor()
        
        # Инициализация аналитики
        agent_analytics = AgentAnalytics()
        
        # Инициализация системы уведомлений
        notification_system = NotificationSystem()
        
        # Инициализация генератора отчетов
        report_generator = ReportGenerator(agent_analytics)
        
        # Инициализация предсказательной аналитики
        predictive_analytics = PredictiveAnalytics()
        
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