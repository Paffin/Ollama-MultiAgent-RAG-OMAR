import streamlit as st
from datetime import datetime
from typing import Dict, Any
from .components import (
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

def run(systems: Dict[str, Any] = None) -> None:
    """
    Запуск Streamlit приложения
    
    Args:
        systems: Словарь с инициализированными системами
    """
    st.set_page_config(
        page_title="OMAR - Multi-Agent RAG System",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("OMAR - Multi-Agent RAG System")
    
    # Инициализация компонентов
    agent_chain = AgentChain(systems)
    analytics_dashboard = AnalyticsDashboard(systems)
    data_panel = DataProcessingPanel(systems)
    notification_panel = NotificationPanel(systems)
    settings_panel = SettingsPanel(systems)
    
    # Отображение компонентов
    with st.sidebar:
        settings_panel.render()
        notification_panel.render()
    
    with st.main():
        agent_chain.render()
        analytics_dashboard.render()
        data_panel.render()

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
    run() 