import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import pandas as pd
from typing import Dict, Any, List, Optional
from utils.exceptions import ValidationError

class AgentChain:
    """Компонент для отображения цепочки работы агентов"""
    def __init__(self, chain: List[Dict[str, Any]]):
        self.chain = chain
        
    def render(self):
        """Отображение цепочки"""
        if not self.chain:
            st.info("Цепочка агентов пуста")
            return
            
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
        if not self.data:
            st.warning("Нет данных для отображения")
            return
            
        st.subheader("Аналитика")
        
        # Общая статистика
        if 'total_stats' in self.data and self.data['total_stats']:
            st.write("### Общая статистика")
            st.json(self.data['total_stats'])
        else:
            st.info("Нет общей статистики")
        
        # Эффективность агентов
        if 'efficiency_scores' in self.data and self.data['efficiency_scores']:
            st.write("### Эффективность агентов")
            for agent, score in self.data['efficiency_scores'].items():
                st.metric(agent, f"{score:.2f}")
        else:
            st.info("Нет данных об эффективности")
            
        # Графики использования
        if 'usage_plots' in self.data and self.data['usage_plots']:
            st.write("### Использование агентов")
            for plot in self.data['usage_plots']:
                try:
                    st.plotly_chart(plot)
                except Exception as e:
                    st.error(f"Ошибка отображения графика: {str(e)}")
        else:
            st.info("Нет графиков использования")
            
        # Графики производительности
        if 'performance_plots' in self.data and self.data['performance_plots']:
            st.write("### Производительность")
            for plot in self.data['performance_plots']:
                try:
                    st.plotly_chart(plot)
                except Exception as e:
                    st.error(f"Ошибка отображения графика: {str(e)}")
        else:
            st.info("Нет графиков производительности")

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
            if not data:
                st.warning("Введите данные для обработки")
                return
                
            try:
                # Валидация входных данных
                if not isinstance(data, str):
                    raise ValidationError("Входные данные должны быть строкой")
                    
                if len(data.strip()) == 0:
                    raise ValidationError("Входные данные не могут быть пустыми")
                    
                # Предварительная обработка
                processed = self.preprocessor.preprocess({"text": data})
                st.write("### Предварительная обработка")
                st.json(processed)
                
                # Валидация
                validation_result = self.validator.validate(processed)
                st.write("### Результаты валидации")
                if validation_result["status"] == "ERROR":
                    st.error("Ошибки валидации:")
                    for field, error in validation_result["errors"].items():
                        st.error(f"{field}: {error['message']}")
                else:
                    st.success("Валидация успешна")
                
                # Обработка
                result = self.processor.process(processed)
                st.write("### Результаты обработки")
                st.json(result)
                
            except ValidationError as e:
                st.error(f"Ошибка валидации: {str(e)}")
            except Exception as e:
                st.error(f"Ошибка обработки: {str(e)}")

class NotificationPanel:
    """Компонент для отображения уведомлений"""
    def __init__(self, notifications):
        self.notifications = notifications
        self.items_per_page = 10
        
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
            
        # Пагинация
        total_pages = (len(notifications) + self.items_per_page - 1) // self.items_per_page
        current_page = st.number_input("Страница", min_value=1, max_value=max(1, total_pages), value=1)
        
        start_idx = (current_page - 1) * self.items_per_page
        end_idx = start_idx + self.items_per_page
        page_notifications = notifications[start_idx:end_idx]
        
        if not page_notifications:
            st.info("Нет уведомлений для отображения")
            return
            
        for notification in page_notifications:
            with st.expander(f"{notification.message} ({notification.type})"):
                st.write(f"Источник: {notification.source}")
                st.write(f"Приоритет: {notification.priority}")
                st.write(f"Время: {notification.timestamp.strftime('%H:%M:%S')}")
                
                if not notification.read:
                    if st.button("Отметить как прочитанное", key=f"read_{notification.id}"):
                        self.notifications.mark_as_read(notification.id)
                        st.rerun()

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
            value=self.config.ollama.base_url,
            key="ollama_base_url"
        )
        
        # Отображение выбранных моделей
        st.write("#### Выбранные модели")
        for agent_type, model in self.config.ollama.models.items():
            st.write(f"{agent_type}: {model}")
        
        # Настройки кэша
        st.write("### Настройки кэша")
        cache_enabled = st.checkbox(
            "Включить кэш",
            value=self.config.cache.enabled,
            key="cache_enabled"
        )
        
        if cache_enabled:
            ttl = st.number_input(
                "Время жизни кэша (секунды)",
                min_value=1,
                value=self.config.cache.ttl_seconds,
                key="cache_ttl"
            )
            
            max_size = st.number_input(
                "Максимальный размер кэша (МБ)",
                min_value=1,
                value=self.config.cache.max_size_mb,
                key="cache_max_size"
            )
            
        # Настройки логирования
        st.write("### Настройки логирования")
        log_level = st.selectbox(
            "Уровень логирования",
            ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            index=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"].index(
                self.config.logging.level
            ),
            key="log_level"
        )
        
        if st.button("Сохранить настройки"):
            try:
                # Обновляем конфигурацию
                from utils.config import ConfigSection
                self.config.update_config(ConfigSection.OLLAMA, base_url=base_url)
                self.config.update_config(ConfigSection.CACHE, enabled=cache_enabled)
                if cache_enabled:
                    self.config.update_config(ConfigSection.CACHE, ttl_seconds=ttl, max_size_mb=max_size)
                self.config.update_config(ConfigSection.LOGGING, level=log_level)
                
                st.success("Настройки сохранены")
                st.rerun()
            except Exception as e:
                st.error(f"Ошибка сохранения настроек: {str(e)}") 