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
                st.caption(f"Время: {step['timestamp'].strftime('%H:%M:%S')}")

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
        
        # Эффективность агентов
        st.write("### Эффективность агентов")
        for agent, score in self.data['efficiency_scores'].items():
            st.metric(agent, f"{score:.2f}")
            
        # Графики использования
        st.write("### Использование агентов")
        for plot in self.data['usage_plots']:
            st.plotly_chart(plot)
            
        # Графики производительности
        st.write("### Производительность")
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
        
        # Ввод данных
        data = st.text_area("Введите данные для обработки")
        
        if st.button("Обработать"):
            if data:
                try:
                    # Предварительная обработка
                    processed = self.preprocessor.preprocess(data)
                    st.write("### Предварительная обработка")
                    st.json(processed)
                    
                    # Валидация
                    validation_result = self.validator.validate(processed)
                    st.write("### Результаты валидации")
                    st.json(validation_result)
                    
                    # Обработка
                    result = self.processor.process(processed)
                    st.write("### Результаты обработки")
                    st.json(result)
                    
                except Exception as e:
                    st.error(f"Ошибка обработки: {str(e)}")
            else:
                st.warning("Введите данные для обработки")

class NotificationPanel:
    """Компонент для отображения уведомлений"""
    def __init__(self, notifications):
        self.notifications = notifications
        
    def render(self):
        """Отображение уведомлений"""
        st.subheader("Уведомления")
        
        # Фильтры
        col1, col2 = st.columns(2)
        with col1:
            notification_type = st.selectbox(
                "Тип уведомления",
                ["Все", "INFO", "WARNING", "ERROR", "SUCCESS", "DEBUG"]
            )
        with col2:
            priority = st.selectbox(
                "Приоритет",
                ["Все", "LOW", "MEDIUM", "HIGH", "CRITICAL"]
            )
            
        # Список уведомлений
        notifications = self.notifications.get_notifications()
        
        if notification_type != "Все":
            notifications = [n for n in notifications if n.type == notification_type]
            
        if priority != "Все":
            notifications = [n for n in notifications if n.priority == priority]
            
        for notification in notifications:
            with st.expander(f"{notification.message} ({notification.type})"):
                st.write(f"Источник: {notification.source}")
                st.write(f"Приоритет: {notification.priority}")
                st.write(f"Время: {notification.timestamp.strftime('%H:%M:%S')}")
                
                if not notification.read:
                    if st.button("Отметить как прочитанное", key=f"read_{notification.id}"):
                        self.notifications.mark_as_read(notification.id)
                        st.experimental_rerun()

class SettingsPanel:
    """Компонент для управления настройками"""
    def __init__(self, config):
        self.config = config
        
    def render(self):
        """Отображение настроек"""
        st.subheader("Настройки")
        
        # Настройки Ollama
        st.write("### Настройки Ollama")
        base_url = st.text_input(
            "Base URL",
            value=self.config.get('ollama', {}).get('base_url', ''),
            key="ollama_base_url"
        )
        
        # Настройки кэша
        st.write("### Настройки кэша")
        cache_enabled = st.checkbox(
            "Включить кэш",
            value=self.config.get('cache', {}).get('enabled', True),
            key="cache_enabled"
        )
        
        if cache_enabled:
            ttl = st.number_input(
                "Время жизни кэша (секунды)",
                min_value=1,
                value=self.config.get('cache', {}).get('ttl', 3600),
                key="cache_ttl"
            )
            
            max_size = st.number_input(
                "Максимальный размер кэша (МБ)",
                min_value=1,
                value=self.config.get('cache', {}).get('max_size', 100),
                key="cache_max_size"
            )
            
        # Настройки логирования
        st.write("### Настройки логирования")
        log_level = st.selectbox(
            "Уровень логирования",
            ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            index=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"].index(
                self.config.get('logging', {}).get('level', 'INFO')
            ),
            key="log_level"
        )
        
        if st.button("Сохранить настройки"):
            # Обновляем конфигурацию
            self.config.set('ollama.base_url', base_url)
            self.config.set('cache.enabled', cache_enabled)
            if cache_enabled:
                self.config.set('cache.ttl', ttl)
                self.config.set('cache.max_size', max_size)
            self.config.set('logging.level', log_level)
            
            st.success("Настройки сохранены")
            st.experimental_rerun() 