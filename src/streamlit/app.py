import streamlit as st
from datetime import datetime
from typing import Dict, Any
import sys
from pathlib import Path

# Добавляем путь к src в PYTHONPATH
src_path = str(Path(__file__).parent.parent)
if src_path not in sys.path:
    sys.path.append(src_path)

from src.streamlit.components import (
    AgentChain,
    AnalyticsDashboard,
    DataProcessingPanel,
    NotificationPanel,
    SettingsPanel
)
from utils.logger import Logger
from utils.exceptions import ValidationError

logger = Logger()

def validate_input_data(data: str) -> bool:
    """
    Валидация входных данных
    
    Args:
        data: Входные данные
        
    Returns:
        bool: True если данные валидны
    """
    if not data or not isinstance(data, str):
        return False
    if len(data.strip()) == 0:
        return False
    return True

def run_app(systems: Dict[str, Any] = None) -> None:
    """
    Запуск Streamlit приложения
    
    Args:
        systems: Словарь с инициализированными системами
    """
    try:
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
        
        # Инициализация компонентов
        agent_chain = AgentChain(st.session_state.agent_chain)
        analytics_dashboard = AnalyticsDashboard(systems['analytics'].get_all_stats())
        data_panel = DataProcessingPanel(
            systems['data_processor'],
            systems['data_validator'],
            systems['data_preprocessor']
        )
        notification_panel = NotificationPanel(st.session_state.notifications)
        settings_panel = SettingsPanel(systems['config'])
        
        # Отображение компонентов
        with st.sidebar:
            settings_panel.render()
            notification_panel.render()
        
        with st.main():
            agent_chain.render()
            analytics_dashboard.render()
            data_panel.render()
            
    except Exception as e:
        logger.error(f"Ошибка в Streamlit приложении: {str(e)}")
        st.error(f"Произошла ошибка: {str(e)}")

def log_chain(agent_name: str, step_type: str, content: str):
    """Логирование шага в цепочке агентов"""
    if 'agent_chain' not in st.session_state:
        st.session_state.agent_chain = []
        
    st.session_state.agent_chain.append({
        'agent': agent_name,
        'type': step_type,
        'content': content,
        'timestamp': datetime.now()
    })
    
def add_notification(message: str, type: str, source: str, priority: int):
    """Добавление уведомления"""
    if 'notifications' not in st.session_state:
        st.session_state.notifications = []
        
    st.session_state.notifications.append({
        'message': message,
        'type': type,
        'source': source,
        'priority': priority,
        'timestamp': datetime.now()
    })

if __name__ == "__main__":
    run_app() 