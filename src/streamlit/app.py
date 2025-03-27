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
    try:
        # Настройка страницы
        st.set_page_config(
            page_title="OMAR - MultiAgent System",
            page_icon="🤖",
            layout="wide"
        )
        
        # Инициализация состояния сессии
        if 'agent_chain' not in st.session_state:
            st.session_state.agent_chain = []
        if 'agent_metrics' not in st.session_state:
            st.session_state.agent_metrics = {}
        if 'notifications' not in st.session_state:
            st.session_state.notifications = []
            
        # Заголовок
        st.title("OMAR - MultiAgent System with RAG and Analytics")
        
        # Боковая панель
        with st.sidebar:
            st.header("Настройки")
            theme = st.selectbox(
                "Тема",
                ["light", "dark"],
                index=0
            )
            
        # Основной контент
        st.header("Статус систем")
        
        # Статус Ollama
        st.subheader("Ollama")
        try:
            ollama_status = "🟢 Доступен" if systems['ollama_client'].check_connection() else "🔴 Недоступен"
        except Exception as e:
            logger.error(f"Ошибка проверки статуса Ollama: {str(e)}")
            ollama_status = "🔴 Ошибка подключения"
        st.write(f"Статус: {ollama_status}")
        
        # Статус кэша
        st.subheader("Кэш")
        cache_status = "🟢 Включен" if systems['cache'].enabled else "🔴 Выключен"
        st.write(f"Статус: {cache_status}")
        
        # Статистика уведомлений
        st.subheader("Уведомления")
        notifications = systems['notifications']
        st.write(f"Всего: {len(notifications.get_notifications())}")
        st.write(f"Непрочитанных: {len(notifications.get_unread_notifications())}")
        
        # Статистика аналитики
        st.subheader("Аналитика")
        analytics = systems['analytics']
        try:
            st.write("Метрики агентов:")
            for agent, stats in analytics.get_agent_stats().items():
                st.write(f"- {agent}: {stats}")
        except Exception as e:
            logger.error(f"Ошибка получения метрик агентов: {str(e)}")
            st.error("Ошибка загрузки метрик")
            
        # Обработка данных
        st.subheader("Обработка данных")
        data_processor = systems['data_processor']
        st.write("Поддерживаемые форматы:")
        for fmt in data_processor.supported_formats:
            st.write(f"- {fmt}")
        
        # Вкладки
        tab1, tab2, tab3, tab4 = st.tabs(["Основной", "Аналитика", "Данные", "Настройки"])
        
        with tab1:
            # Основной функционал
            user_query = st.text_input("Введите запрос:")
            if st.button("Обработать"):
                if user_query:
                    # Валидация входных данных
                    if not validate_input_data(user_query):
                        st.error("Введите корректный текст запроса")
                        return
                        
                    # Очищаем предыдущую цепочку
                    st.session_state.agent_chain = []
                    
                    # Обработка запроса
                    start_time = datetime.now()
                    
                    # Логируем начало обработки
                    log_chain("system", "start", "Начало обработки запроса")
                    add_notification("Начало обработки запроса", "info", "system", 3)
                    
                    try:
                        # Валидация и обработка данных
                        processed_data = systems['data_processor'].process_text(user_query)
                        log_chain("data_processor", "validation", "Валидация данных успешно завершена")
                        
                        # Обновление метрик агентов
                        systems['analytics'].update_usage_stats(
                            agent_name="planner",
                            success=True,
                            response_time=(datetime.now() - start_time).total_seconds(),
                            quality_score=0.8,
                            resource_usage=0.5
                        )
                        
                        # Отображение результатов
                        st.markdown("### Результаты обработки")
                        st.json(processed_data)
                        
                        # Логируем успешное завершение
                        log_chain("system", "success", "Запрос успешно обработан")
                        add_notification("Запрос успешно обработан", "success", "system", 4)
                        
                    except ValidationError as e:
                        # Логируем ошибку валидации
                        log_chain("system", "error", f"Ошибка валидации: {str(e)}")
                        add_notification(f"Ошибка валидации: {str(e)}", "error", "system", 5)
                        st.error(f"Ошибка валидации: {str(e)}")
                    except Exception as e:
                        # Логируем ошибку
                        log_chain("system", "error", f"Ошибка обработки: {str(e)}")
                        add_notification(f"Ошибка обработки: {str(e)}", "error", "system", 5)
                        st.error(f"Произошла ошибка: {str(e)}")
                    
                    # Отображаем цепочку работы агентов
                    AgentChain(st.session_state.agent_chain).render()
                    
                else:
                    st.warning("Введите текст запроса")
                
        with tab2:
            # Аналитика
            try:
                analytics_data = {
                    'total_stats': systems['analytics'].get_all_stats(),
                    'efficiency_scores': {
                        agent: systems['analytics'].get_efficiency_score(agent)
                        for agent in systems['analytics'].get_all_stats()
                    },
                    'usage_plots': systems['analytics'].generate_usage_plots(),
                    'performance_plots': systems['analytics'].generate_performance_plots()
                }
                AnalyticsDashboard(analytics_data).render()
            except Exception as e:
                logger.error(f"Ошибка отображения аналитики: {str(e)}")
                st.error("Ошибка загрузки аналитики")
            
        with tab3:
            # Обработка данных
            DataProcessingPanel(
                systems['data_processor'],
                systems['data_validator'],
                systems['data_preprocessor']
            ).render()
            
        with tab4:
            # Настройки
            SettingsPanel(systems['config']).render()
            
        # Панель уведомлений
        NotificationPanel(systems['notifications']).render()
        
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