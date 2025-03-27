import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import pandas as pd
from typing import Dict, Any, List, Optional

class AgentChain:
    """Компонент для отображения цепочки работы агентов"""
    def __init__(self, chain: List[Dict[str, Any]]):
        self.chain = chain
        
    def render(self):
        """Отображение цепочки"""
        st.subheader("Цепочка работы агентов")
        for step in self.chain:
            with st.expander(f"{step['agent']} - {step['type']}"):
                st.write(step['content'])
                st.caption(f"Время: {step['timestamp']}")

class AnalyticsDashboard:
    """Компонент для отображения аналитики"""
    def __init__(self, data: Dict[str, Any]):
        self.data = data
        
    def render(self):
        """Отображение аналитики"""
        st.subheader("Аналитика")
        
        # Общая статистика
        st.write("### Общая статистика")
        st.json(self.data['total_stats'])
        
        # Оценки эффективности
        st.write("### Оценки эффективности")
        for agent, score in self.data['efficiency_scores'].items():
            st.write(f"{agent}: {score}")
            
        # Графики использования
        st.write("### Графики использования")
        for plot in self.data['usage_plots']:
            st.plotly_chart(plot)
            
        # Графики производительности
        st.write("### Графики производительности")
        for plot in self.data['performance_plots']:
            st.plotly_chart(plot)

class DataProcessingPanel:
    """Компонент для обработки данных"""
    def __init__(self, processor, validator, preprocessor):
        self.processor = processor
        self.validator = validator
        self.preprocessor = preprocessor
        
    def render(self):
        """Отображение панели обработки данных"""
        st.subheader("Обработка данных")
        
        # Загрузка файла
        uploaded_file = st.file_uploader("Выберите файл", type=['csv', 'json', 'yaml', 'xml'])
        
        if uploaded_file is not None:
            try:
                # Обработка файла
                data = self.processor.process_file(uploaded_file)
                
                # Валидация
                validation_result = self.validator.validate(data)
                if validation_result:
                    st.success("Данные успешно валидированы")
                else:
                    st.error("Ошибка валидации данных")
                    
                # Предварительная обработка
                processed_data = self.preprocessor.preprocess(data)
                
                # Отображение результатов
                st.write("### Результаты обработки")
                st.json(processed_data)
                
            except Exception as e:
                st.error(f"Ошибка при обработке файла: {str(e)}")

class NotificationPanel:
    """Компонент для отображения уведомлений"""
    def __init__(self, notification_system):
        self.notification_system = notification_system
        
    def render(self):
        """Отображение уведомлений"""
        st.subheader("Уведомления")
        
        # Получаем уведомления
        notifications = self.notification_system.get_notifications()
        
        # Фильтры
        col1, col2 = st.columns(2)
        with col1:
            notification_type = st.selectbox(
                "Тип уведомления",
                ["Все", "INFO", "WARNING", "ERROR", "SUCCESS"]
            )
        with col2:
            priority = st.selectbox(
                "Приоритет",
                ["Все", "LOW", "MEDIUM", "HIGH", "CRITICAL"]
            )
            
        # Отображение уведомлений
        for notification in notifications:
            if (notification_type == "Все" or notification.type == notification_type) and \
               (priority == "Все" or notification.priority == priority):
                with st.expander(f"{notification.type} - {notification.message}"):
                    st.write(f"Источник: {notification.source}")
                    st.write(f"Приоритет: {notification.priority}")
                    st.write(f"Время: {notification.timestamp}")
                    if notification.metadata:
                        st.write("Метаданные:")
                        st.json(notification.metadata)

class SettingsPanel:
    """Компонент для отображения настроек"""
    def __init__(self, config):
        self.config = config
        
    def render(self):
        """Отображение настроек"""
        st.subheader("Настройки")
        
        # Настройки Ollama
        st.write("### Настройки Ollama")
        ollama_url = st.text_input(
            "URL сервера Ollama",
            value=self.config.ollama.base_url
        )
        timeout = st.number_input(
            "Таймаут (секунды)",
            min_value=1,
            max_value=60,
            value=self.config.ollama.timeout
        )
        
        # Настройки агентов
        st.write("### Настройки агентов")
        temperature = st.slider(
            "Температура",
            min_value=0.0,
            max_value=2.0,
            value=self.config.agents.temperature,
            step=0.1
        )
        top_p = st.slider(
            "Top P",
            min_value=0.0,
            max_value=1.0,
            value=self.config.agents.top_p,
            step=0.1
        )
        
        # Настройки обработки данных
        st.write("### Настройки обработки данных")
        chunk_size = st.number_input(
            "Размер чанка",
            min_value=100,
            max_value=10000,
            value=self.config.data.chunk_size,
            step=100
        )
        
        # Кнопка сохранения
        if st.button("Сохранить настройки"):
            try:
                self.config.update_config(
                    ConfigSection.OLLAMA,
                    base_url=ollama_url,
                    timeout=timeout
                )
                self.config.update_config(
                    ConfigSection.AGENTS,
                    temperature=temperature,
                    top_p=top_p
                )
                self.config.update_config(
                    ConfigSection.DATA,
                    chunk_size=chunk_size
                )
                st.success("Настройки успешно сохранены")
            except Exception as e:
                st.error(f"Ошибка при сохранении настроек: {str(e)}") 