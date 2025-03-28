import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import pandas as pd
from typing import Dict, Any, List, Optional
from utils.exceptions import ValidationError
from agents import (
    PlannerAgent,
    ExecutorAgent,
    CriticAgent,
    PraiseAgent,
    ArbiterAgent
)

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

class AgentInteractionPanel:
    """Компонент для взаимодействия с агентами"""
    def __init__(self, systems):
        self.systems = systems
        self.agents = {
            "planner": PlannerAgent(
                name="planner",
                model_name=systems['config'].ollama.models.get("planner"),
                client=systems['ollama_client']
            ),
            "executor": ExecutorAgent(
                name="executor",
                model_name=systems['config'].ollama.models.get("executor"),
                client=systems['ollama_client']
            ),
            "critic": CriticAgent(
                name="critic",
                model_name=systems['config'].ollama.models.get("critic"),
                client=systems['ollama_client']
            ),
            "praise": PraiseAgent(
                name="praise",
                model_name=systems['config'].ollama.models.get("praise"),
                client=systems['ollama_client']
            ),
            "arbiter": ArbiterAgent(
                name="arbiter",
                model_name=systems['config'].ollama.models.get("arbiter"),
                client=systems['ollama_client']
            )
        }
        
    def render(self):
        """Отображение панели взаимодействия с агентами"""
        st.subheader("Взаимодействие с агентами")
        
        # Ввод запроса
        query = st.text_area("Введите ваш запрос")
        
        if st.button("Обработать запрос"):
            if not query:
                st.warning("Введите запрос")
                return
                
            try:
                import asyncio
                
                # Планирование
                with st.spinner("Планирование..."):
                    plan = asyncio.run(self.agents["planner"].process({"query": query}))
                    st.write("### План действий")
                    st.write(plan)
                    
                # Выполнение
                with st.spinner("Выполнение..."):
                    result = asyncio.run(self.agents["executor"].process({"plan": plan}))
                    st.write("### Результат выполнения")
                    st.write(result)
                    
                # Критика
                with st.spinner("Анализ результата..."):
                    critique = asyncio.run(self.agents["critic"].process({"result": result}))
                    st.write("### Критический анализ")
                    st.write(critique)
                    
                # Похвала
                with st.spinner("Оценка качества..."):
                    praise = asyncio.run(self.agents["praise"].process({"result": result}))
                    st.write("### Положительные аспекты")
                    st.write(praise)
                    
                # Арбитраж
                with st.spinner("Финальная оценка..."):
                    final_verdict = asyncio.run(self.agents["arbiter"].process({
                        "query": query,
                        "plan": plan,
                        "result": result,
                        "critique": critique,
                        "praise": praise
                    }))
                    st.write("### Финальное решение")
                    st.write(final_verdict)
                    
                # Добавляем в цепочку
                st.session_state.agent_chain.append({
                    "agent": "planner",
                    "type": "plan",
                    "content": plan,
                    "timestamp": datetime.now()
                })
                st.session_state.agent_chain.append({
                    "agent": "executor",
                    "type": "result",
                    "content": result,
                    "timestamp": datetime.now()
                })
                st.session_state.agent_chain.append({
                    "agent": "critic",
                    "type": "critique",
                    "content": critique,
                    "timestamp": datetime.now()
                })
                st.session_state.agent_chain.append({
                    "agent": "praise",
                    "type": "praise",
                    "content": praise,
                    "timestamp": datetime.now()
                })
                st.session_state.agent_chain.append({
                    "agent": "arbiter",
                    "type": "verdict",
                    "content": final_verdict,
                    "timestamp": datetime.now()
                })
                
                st.success("Запрос успешно обработан")
                st.rerun()
                
            except Exception as e:
                st.error(f"Ошибка при обработке запроса: {str(e)}")
                self.systems['logger'].error(f"Ошибка при обработке запроса: {str(e)}") 