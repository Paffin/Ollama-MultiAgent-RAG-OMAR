import sys
import asyncio
from pathlib import Path
import requests
from typing import Dict, Any, List
import time
import streamlit as st

# Настройка для Windows
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# Подключаем nest_asyncio
import nest_asyncio
nest_asyncio.apply()

# Добавляем путь к src в PYTHONPATH
src_path = str(Path(__file__).parent)
if src_path not in sys.path:
    sys.path.append(src_path)

# Импорты из нашего пакета
from utils.config import ConfigManager
from utils.logger import Logger
from utils.exceptions import handle_error, ConfigurationError
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
from utils.cache import Cache
from streamlite.components import (
    AgentChain,
    AnalyticsDashboard,
    DataProcessingPanel,
    NotificationPanel,
    SettingsPanel,
    AgentInteractionPanel
)

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
    """Инициализация систем"""
    try:
        # Инициализация конфигурации
        config = ConfigManager()
        
        # Валидация конфигурации
        if not validate_config(config.get_config_dict()):
            raise ConfigurationError("Ошибка валидации конфигурации")
            
        # Инициализация логгера
        logger = Logger()
        
        # Инициализация кэша
        cache = Cache(
            ttl_seconds=config.cache.ttl_seconds,
            max_size_mb=config.cache.max_size_mb
        )
        
        # Инициализация системы уведомлений
        notifications = NotificationSystem()
        
        # Инициализация аналитики
        analytics = AgentAnalytics()
        
        # Инициализация обработчиков данных
        data_processor = DataProcessor()
        data_validator = DataValidator()
        data_preprocessor = DataPreprocessor()
        
        # Добавляем правила валидации
        def validate_text(text: str) -> bool:
            if not isinstance(text, str):
                return False
            if len(text.strip()) == 0:
                return False
            if len(text) > 10000:
                return False
            return True
            
        data_validator.add_rule("text", validate_text)
        
        # Инициализация Ollama клиента
        ollama_client = OllamaClient(
            base_url=config.ollama.base_url
        )
        
        return {
            'config': config,
            'logger': logger,
            'cache': cache,
            'notifications': notifications,
            'analytics': analytics,
            'data_processor': data_processor,
            'data_validator': data_validator,
            'data_preprocessor': data_preprocessor,
            'ollama_client': ollama_client
        }
        
    except Exception as e:
        handle_error(e, "Инициализация систем")
        raise

async def get_available_models(ollama_client: OllamaClient) -> List[str]:
    """
    Получение списка доступных моделей
    
    Args:
        ollama_client: Клиент Ollama
        
    Returns:
        Список доступных моделей
    """
    try:
        return await ollama_client.list_models()
    except Exception as e:
        st.error(f"Ошибка при получении списка моделей: {str(e)}")
        return []

def select_models(available_models: List[str], config: ConfigManager) -> Dict[str, str]:
    """
    Выбор моделей через интерфейс
    
    Args:
        available_models: Список доступных моделей
        config: Конфигурация
        
    Returns:
        Словарь с выбранными моделями
    """
    st.subheader("Выбор моделей для агентов")
    
    selected_models = {}
    for agent_type in ["planner", "executor", "critic", "praise", "arbiter"]:
        current_model = config.ollama.models.get(agent_type)
        selected_model = st.selectbox(
            f"Модель для {agent_type}",
            options=available_models,
            index=available_models.index(current_model) if current_model in available_models else 0
        )
        selected_models[agent_type] = selected_model
    
    return selected_models

def main():
    """Главная функция приложения"""
    try:
        # Инициализация систем
        systems = init_systems()
        
        # Настройка страницы
        st.set_page_config(
            page_title="OMAR - Multi-Agent RAG System",
            page_icon="🤖",
            layout="wide"
        )
        
        # Инициализация состояния сессии
        if 'agent_chain' not in st.session_state:
            st.session_state.agent_chain = []
        if 'notifications' not in st.session_state:
            st.session_state.notifications = []
            
        # Заголовок
        st.title("OMAR - Multi-Agent RAG System")
        
        # Получение списка доступных моделей
        available_models = asyncio.run(get_available_models(systems['ollama_client']))
        
        if not available_models:
            st.error("Не удалось получить список моделей. Убедитесь, что сервер Ollama запущен.")
            return
            
        # Выбор моделей
        selected_models = select_models(available_models, systems['config'])
        
        # Обновление конфигурации
        systems['config'].ollama.models = selected_models
        
        # Инициализация компонентов
        agent_chain = AgentChain(st.session_state.agent_chain)
        agent_stats = systems['analytics'].get_agent_stats()
        analytics_data = {
            'total_stats': {
                'total_calls': sum(stats['total_calls'] for stats in agent_stats.values()),
                'successful_calls': sum(stats['successful_calls'] for stats in agent_stats.values()),
                'failed_calls': sum(stats['failed_calls'] for stats in agent_stats.values()),
                'total_duration': sum(stats['total_duration'] for stats in agent_stats.values()),
                'avg_duration': sum(stats['avg_duration'] for stats in agent_stats.values()) / len(agent_stats) if agent_stats else 0,
                'error_count': sum(stats['error_count'] for stats in agent_stats.values())
            },
            'efficiency_scores': {
                name: stats['success_rate'] * 100 for name, stats in agent_stats.items()
            }
        }
        analytics_dashboard = AnalyticsDashboard(analytics_data)
        data_panel = DataProcessingPanel(
            systems['data_processor'],
            systems['data_validator'],
            systems['data_preprocessor']
        )
        notification_panel = NotificationPanel(systems['notifications'])
        settings_panel = SettingsPanel(systems['config'])
        agent_panel = AgentInteractionPanel(systems)
        
        # Отображение компонентов
        with st.sidebar:
            settings_panel.render()
            notification_panel.render()
        
        # Основной контент
        agent_panel.render()
        agent_chain.render()
        analytics_dashboard.render()
        data_panel.render()
            
    except Exception as e:
        handle_error(e, "Запуск приложения")
        sys.exit(1)

if __name__ == "__main__":
    main() 