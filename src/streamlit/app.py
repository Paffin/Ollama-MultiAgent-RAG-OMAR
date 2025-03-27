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

def run(systems: Dict[str, Any]):
    """Запуск Streamlit приложения"""
    
    # Настройка страницы
    st.set_page_config(
        page_title="OMAR - MultiAgent System",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Инициализация состояния сессии
    if 'agent_chain' not in st.session_state:
        st.session_state.agent_chain = []
    if 'agent_metrics' not in st.session_state:
        st.session_state.agent_metrics = {}
        
    # Заголовок
    st.title("OMAR - MultiAgent System with RAG and Analytics")
    
    # Вкладки
    tab1, tab2, tab3, tab4 = st.tabs(["Основной", "Аналитика", "Данные", "Настройки"])
    
    with tab1:
        # Основной функционал
        user_query = st.text_input("Введите запрос:")
        if st.button("Обработать"):
            if user_query:
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
                    systems['agent_analytics'].update_usage_stats(
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
        analytics_data = {
            'total_stats': systems['agent_analytics'].get_all_stats(),
            'efficiency_scores': {
                agent: systems['agent_analytics'].get_efficiency_score(agent)
                for agent in systems['agent_analytics'].get_all_stats()
            },
            'usage_plots': systems['agent_analytics'].generate_usage_plots(),
            'performance_plots': systems['agent_analytics'].generate_performance_plots()
        }
        AnalyticsDashboard(analytics_data).render()
        
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
    NotificationPanel(systems['notification_system']).render()
    
def log_chain(agent_name: str, step_type: str, content: str):
    """Логирование шага в цепочке агентов"""
    st.session_state.agent_chain.append({
        'agent': agent_name,
        'type': step_type,
        'content': content,
        'timestamp': datetime.now()
    })
    
def add_notification(message: str, type: str = "info", source: str = "system", priority: int = 3):
    """Добавление уведомления"""
    st.session_state.notification_system.add_notification(
        message=message,
        type=type,
        source=source,
        priority=priority
    ) 