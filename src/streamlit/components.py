import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import pandas as pd
from typing import Dict, Any, List, Optional

class AgentChain:
    """Компонент для отображения цепочки агентов"""
    
    def __init__(self, chain_data: List[Dict[str, Any]]):
        self.chain_data = chain_data
        
    def render(self):
        """Отрисовка компонента"""
        if not self.chain_data:
            st.info("Пока нет шагов в логе.")
            return
            
        st.markdown("### Процесс работы агентов")
        
        # Создаем временную шкалу
        timeline = go.Figure()
        
        # Добавляем шаги
        for i, step in enumerate(self.chain_data):
            timeline.add_trace(go.Scatter(
                x=[step['timestamp']],
                y=[i],
                mode='markers+text',
                name=step['agent'],
                text=[f"{step['type']}: {step['content'][:50]}..."],
                textposition="top center",
                marker=dict(size=10)
            ))
        
        # Настраиваем внешний вид
        timeline.update_layout(
            title="Временная шкала работы агентов",
            showlegend=True,
            height=400,
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            xaxis=dict(title="Время")
        )
        
        st.plotly_chart(timeline, use_container_width=True)
        
        # Отображаем детали шагов
        st.markdown("#### Детали шагов")
        for step in self.chain_data:
            with st.expander(f"{step['agent']} - {step['type']} ({step['timestamp'].strftime('%H:%M:%S')})"):
                st.text(step['content'])

class AnalyticsDashboard:
    """Компонент для отображения аналитики"""
    
    def __init__(self, analytics_data: Dict[str, Any]):
        self.analytics_data = analytics_data
        
    def render(self):
        """Отрисовка компонента"""
        st.markdown("### Аналитика")
        
        # Общая статистика
        st.markdown("#### Общая статистика")
        if self.analytics_data.get('total_stats'):
            stats_df = pd.DataFrame([
                {
                    'Агент': agent,
                    'Всего запросов': stats.total_requests,
                    'Успешных': stats.successful_requests,
                    'Неудачных': stats.failed_requests,
                    'Среднее время ответа': f"{stats.avg_response_time:.2f} сек",
                    'Среднее качество': f"{stats.avg_quality_score:.2f}",
                    'Эффективность': f"{self.analytics_data['efficiency_scores'].get(agent, 0):.2f}"
                }
                for agent, stats in self.analytics_data['total_stats'].items()
            ])
            st.table(stats_df)
            
            # График эффективности
            fig_efficiency = px.bar(
                stats_df,
                x='Агент',
                y='Эффективность',
                title='Эффективность агентов'
            )
            st.plotly_chart(fig_efficiency)
        
        # Графики использования
        if self.analytics_data.get('usage_plots'):
            st.markdown("#### Графики использования")
            for plot_name, plot in self.analytics_data['usage_plots'].items():
                st.plotly_chart(plot)
        
        # Графики производительности
        if self.analytics_data.get('performance_plots'):
            st.markdown("#### Графики производительности")
            for plot_name, plot in self.analytics_data['performance_plots'].items():
                st.plotly_chart(plot)

class DataProcessingPanel:
    """Компонент для обработки данных"""
    
    def __init__(self, data_processor: Any, data_validator: Any, data_preprocessor: Any):
        self.data_processor = data_processor
        self.data_validator = data_validator
        self.data_preprocessor = data_preprocessor
        
    def render(self):
        """Отрисовка компонента"""
        st.markdown("### Обработка данных")
        
        # Загрузка данных
        uploaded_file = st.file_uploader("Выберите файл", type=['csv', 'json', 'yaml', 'xml', 'parquet'])
        if uploaded_file:
            try:
                # Обработка файла
                data = self.data_processor.process_file(uploaded_file)
                st.dataframe(data)
                
                # Валидация данных
                if st.button("Валидировать данные"):
                    validation_result = self.data_validator.validate_data(data)
                    if validation_result.is_valid:
                        st.success("Данные прошли валидацию")
                    else:
                        st.error("Ошибки валидации:")
                        for error in validation_result.errors:
                            st.error(error)
                
                # Предварительная обработка
                if st.button("Предварительная обработка"):
                    processed_data = self.data_preprocessor.preprocess_data(data)
                    st.dataframe(processed_data)
                    
            except Exception as e:
                st.error(f"Ошибка при обработке файла: {e}")

class NotificationPanel:
    """Компонент для отображения уведомлений"""
    
    def __init__(self, notification_system: Any):
        self.notification_system = notification_system
        
    def render(self):
        """Отрисовка компонента"""
        st.markdown("### Уведомления")
        
        # Получение уведомлений
        notifications = self.notification_system.get_notifications()
        
        # Отображение уведомлений
        for notification in notifications:
            with st.container():
                if notification.type == "error":
                    st.error(notification.message)
                elif notification.type == "warning":
                    st.warning(notification.message)
                elif notification.type == "success":
                    st.success(notification.message)
                else:
                    st.info(notification.message)

class SettingsPanel:
    """Компонент для настроек"""
    
    def __init__(self, config_manager: Any):
        self.config_manager = config_manager
        
    def render(self):
        """Отрисовка компонента"""
        st.markdown("### Настройки")
        
        # Настройки темы
        theme = st.selectbox("Тема", ["light", "dark"])
        self.config_manager.update_config(theme=theme)
        
        # Настройки Ollama
        st.markdown("#### Настройки Ollama")
        ollama_config = self.config_manager.config.ollama
        ollama_config.temperature = st.slider(
            "Temperature",
            0.0,
            2.0,
            ollama_config.temperature,
            0.1
        )
        ollama_config.top_p = st.slider(
            "Top P",
            0.0,
            1.0,
            ollama_config.top_p,
            0.05
        )
        
        # Настройки агентов
        st.markdown("#### Настройки агентов")
        agent_config = self.config_manager.config.agent
        agent_config.max_iterations = st.number_input(
            "Максимальное количество итераций",
            1,
            10,
            agent_config.max_iterations,
            1
        )
        agent_config.min_quality = st.slider(
            "Минимальное качество",
            0.0,
            1.0,
            agent_config.min_quality,
            0.05
        )
        
        # Сохранение настроек
        if st.button("Сохранить настройки"):
            self.config_manager.save_config()
            st.success("Настройки сохранены") 