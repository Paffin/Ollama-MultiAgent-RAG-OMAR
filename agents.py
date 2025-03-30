from typing import List, Dict, Union, Generator, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import re
import time
import threading
import logging
import os
import urllib.parse
from ollama_client import OllamaClient
from rag_db import SimpleVectorStore
from windows_tools import (
    run_system_command, list_directory,
    PlaywrightBrowser,
    llava_analyze_screenshot_via_ollama_llava
)
from system_prompts import (
    PLANNER_PROMPT,
    EXECUTOR_PROMPT,
    CRITIC_PROMPT,
    PRAISE_PROMPT,
    ARBITER_PROMPT
)

logger = logging.getLogger(__name__)

@dataclass
class AgentMetrics:
    """Метрики агента для отслеживания производительности."""
    tokens: int = 0
    api_calls: int = 0
    processing_time: float = 0.0
    memory_usage: int = 0
    steps_completed: int = 0
    total_steps: int = 0

@dataclass
class ProgressPoint:
    """Точка прогресса для отслеживания выполнения задачи."""
    time: float
    progress: float
    status: str

class AgentStatus(Enum):
    """Статусы агента."""
    IDLE = "idle"
    PLANNING = "planning"
    EXECUTING = "executing"
    CRITICIZING = "criticizing"
    PRAISING = "praising"
    ARBITRATING = "arbitrating"
    COMPLETED = "completed"
    ERROR = "error"

    @property
    def value(self) -> str:
        return super().value

    def __str__(self) -> str:
        return self.value

class AgentState:
    """Состояние агента с отслеживанием прогресса и метрик."""
    
    def __init__(self):
        """Инициализирует состояние агента."""
        self.status = AgentStatus.IDLE
        self.current_task = ""
        self.error = None
        self.progress = 0.0
        self.start_time = None
        self.end_time = None
        self.metrics = AgentMetrics()
        self._last_update = time.time()
        self._update_interval = 0.1
        self._progress_history = []
        self._lock = threading.Lock()
        self._status_transitions = {
            AgentStatus.IDLE: [AgentStatus.PLANNING, AgentStatus.EXECUTING, 
                             AgentStatus.CRITICIZING, AgentStatus.PRAISING, 
                             AgentStatus.ARBITRATING],
            AgentStatus.PLANNING: [AgentStatus.COMPLETED, AgentStatus.ERROR],
            AgentStatus.EXECUTING: [AgentStatus.COMPLETED, AgentStatus.ERROR],
            AgentStatus.CRITICIZING: [AgentStatus.COMPLETED, AgentStatus.ERROR],
            AgentStatus.PRAISING: [AgentStatus.COMPLETED, AgentStatus.ERROR],
            AgentStatus.ARBITRATING: [AgentStatus.COMPLETED, AgentStatus.ERROR],
            AgentStatus.COMPLETED: [AgentStatus.IDLE],
            AgentStatus.ERROR: [AgentStatus.IDLE]
        }

    def start_task(self, task: str, total_steps: int = 0) -> None:
        """Начинает новую задачу."""
        with self._lock:
            if not task:
                raise ValueError("Задача не может быть пустой")
                
            # Проверяем возможность перехода
            if self.status != AgentStatus.IDLE:
                raise ValueError(f"Невозможно начать новую задачу в статусе {self.status}")
                
            self.current_task = task
            self.error = None
            self.progress = 0.0
            self.start_time = time.time()
            self.end_time = None
            self.metrics = AgentMetrics(total_steps=total_steps)
            self._progress_history = []
            self._last_update = time.time()
            self._update_progress_history()

    def complete_task(self) -> None:
        """Завершает текущую задачу."""
        with self._lock:
            if self.status == AgentStatus.COMPLETED:
                return
                
            if self.status == AgentStatus.ERROR:
                raise ValueError("Невозможно завершить задачу с ошибкой")
                
            if self.status == AgentStatus.IDLE:
                raise ValueError("Невозможно завершить незапущенную задачу")
                
            self._transition_to(AgentStatus.COMPLETED)
            self.progress = 1.0
            self.end_time = time.time()
            self.metrics.processing_time = self.end_time - self.start_time
            self._update_progress_history()

    def set_error(self, error: str) -> None:
        """Устанавливает ошибку."""
        with self._lock:
            if not error:
                raise ValueError("Описание ошибки не может быть пустым")
                
            self._transition_to(AgentStatus.ERROR)
            self.error = error
            self.end_time = time.time()
            self.metrics.processing_time = self.end_time - self.start_time
            self._update_progress_history()

    def update_status(self, new_status: AgentStatus) -> None:
        """Обновляет статус агента."""
        with self._lock:
            if not isinstance(new_status, AgentStatus):
                raise ValueError("Некорректный тип статуса")
                
            self._transition_to(new_status)
            self._update_progress_history()

    def _transition_to(self, new_status: AgentStatus) -> None:
        """Выполняет переход в новый статус с проверкой."""
        if new_status not in self._status_transitions[self.status]:
            raise ValueError(
                f"Недопустимый переход из {self.status} в {new_status}. "
                f"Разрешенные переходы: {self._status_transitions[self.status]}"
            )
        self.status = new_status

    def update_metrics(self, tokens: int = 0, api_calls: int = 0, memory_usage: int = 0) -> None:
        """Обновляет метрики агента."""
        with self._lock:
            if tokens < 0 or api_calls < 0 or memory_usage < 0:
                raise ValueError("Метрики не могут быть отрицательными")
                
            self.metrics.tokens += tokens
            self.metrics.api_calls += api_calls
            self.metrics.memory_usage = max(self.metrics.memory_usage, memory_usage)

    def update_progress(self, progress: float, status: Optional[AgentStatus] = None) -> None:
        """Обновляет прогресс выполнения."""
        current_time = time.time()
        if current_time - self._last_update < self._update_interval:
            return
            
        with self._lock:
            if not 0 <= progress <= 1:
                raise ValueError("Прогресс должен быть в диапазоне [0, 1]")
                
            self.progress = progress
            if status:
                self._transition_to(status)
            self._last_update = current_time
            self._update_progress_history()

    def increment_steps(self, count: int = 1) -> None:
        """Увеличивает счетчик выполненных шагов."""
        with self._lock:
            if count < 1:
                raise ValueError("Количество шагов должно быть положительным")
                
            self.metrics.steps_completed += count
            if self.metrics.total_steps > 0:
                self.progress = min(1.0, self.metrics.steps_completed / self.metrics.total_steps)
                self._update_progress_history()

    def _update_progress_history(self) -> None:
        """Обновляет историю прогресса."""
        current_time = time.time()
        point = {
            "time": current_time,
            "progress": self.progress,
            "status": self.status.value,
            "task": self.current_task,
            "error": self.error,
            "metrics": {
                "tokens": self.metrics.tokens,
                "api_calls": self.metrics.api_calls,
                "steps": f"{self.metrics.steps_completed}/{self.metrics.total_steps}",
                "memory": self.metrics.memory_usage
            }
        }
        self._progress_history.append(point)
        
        # Ограничиваем историю последними 100 точками
        if len(self._progress_history) > 100:
            self._progress_history = self._progress_history[-100:]

    def get_progress_history(self) -> List[Dict[str, Any]]:
        """Возвращает историю прогресса."""
        with self._lock:
            return self._progress_history.copy()

    def get_metrics(self) -> Dict[str, Any]:
        """Возвращает текущие метрики."""
        with self._lock:
            return {
                "tokens": self.metrics.tokens,
                "api_calls": self.metrics.api_calls,
                "processing_time": self.metrics.processing_time,
                "memory_usage": self.metrics.memory_usage,
                "steps_completed": self.metrics.steps_completed,
                "total_steps": self.metrics.total_steps
            }

    def get_status_info(self) -> Dict[str, Any]:
        """Возвращает информацию о статусе."""
        with self._lock:
            return {
                "status": self.status.value,
                "current_task": self.current_task,
                "progress": self.progress,
                "error": self.error,
                "start_time": self.start_time,
                "end_time": self.end_time,
                "processing_time": self.metrics.processing_time,
                "steps_completed": self.metrics.steps_completed,
                "total_steps": self.metrics.total_steps
            }

class BaseAgent:
    """Базовый класс для всех агентов в системе.
    
    Этот класс предоставляет базовую функциональность для всех агентов:
    - Управление состоянием
    - История сообщений
    - Построение промптов
    - Обработка ошибок
    - Метрики производительности
    """
    
    def __init__(self, name: str, system_prompt: str, model_name: str, client: OllamaClient):
        """Инициализирует базового агента.
        
        Args:
            name: Имя агента
            system_prompt: Системный промпт
            model_name: Имя модели LLM
            client: Клиент Ollama
            
        Raises:
            ValueError: Если параметры некорректны
        """
        if not name or not isinstance(name, str):
            raise ValueError("Имя агента должно быть непустой строкой")
        if not system_prompt or not isinstance(system_prompt, str):
            raise ValueError("Системный промпт должен быть непустой строкой")
        if not model_name or not isinstance(model_name, str):
            raise ValueError("Имя модели должно быть непустой строкой")
        if not isinstance(client, OllamaClient):
            raise ValueError("Клиент должен быть экземпляром OllamaClient")
            
        self.name = name
        self.system_prompt = system_prompt
        self.model_name = model_name
        self.client = client
        self.state = AgentState()
        self.message_history: List[Dict[str, str]] = []
        self._browser: Optional[PlaywrightBrowser] = None
    
    def add_message(self, role: str, content: str) -> None:
        """Добавляет сообщение в историю диалога.
        
        Args:
            role: Роль отправителя (user/assistant)
            content: Содержание сообщения
            
        Raises:
            ValueError: Если параметры некорректны
        """
        if not role or not isinstance(role, str):
            raise ValueError("Роль должна быть непустой строкой")
        if not content or not isinstance(content, str):
            raise ValueError("Содержание должно быть непустой строкой")
            
        self.message_history.append({"role": role, "content": content})
    
    def build_prompt(self, user_query: str) -> str:
        """Создает промпт для модели.
        
        Args:
            user_query: Запрос пользователя
            
        Returns:
            Полный промпт для модели
            
        Raises:
            ValueError: Если запрос пустой
        """
        if not user_query or not isinstance(user_query, str):
            raise ValueError("Запрос пользователя должен быть непустой строкой")
            
        prompt_parts = [
            self.system_prompt,
            "\nИстория сообщений:",
            *[f"{msg['role']}: {msg['content']}" for msg in self.message_history],
            f"\nЗапрос пользователя: {user_query}"
        ]
        return "\n".join(prompt_parts)

    def update_state(self, status: AgentStatus, task: str, total_steps: int = 0) -> None:
        """Обновляет состояние агента.
        
        Args:
            status: Новый статус
            task: Описание текущей задачи
            total_steps: Общее количество шагов
            
        Raises:
            ValueError: Если параметры некорректны
        """
        if not isinstance(status, AgentStatus):
            raise ValueError("Статус должен быть экземпляром AgentStatus")
        if not task or not isinstance(task, str):
            raise ValueError("Задача должна быть непустой строкой")
        if not isinstance(total_steps, int) or total_steps < 0:
            raise ValueError("Количество шагов должно быть неотрицательным целым числом")
            
        if self._should_create_new_state():
            self._create_new_state()
        self.state.start_task(task, total_steps)
        self.state.status = status

    def _should_create_new_state(self) -> bool:
        """Проверяет, нужно ли создать новое состояние.
        
        Returns:
            True если нужно создать новое состояние
        """
        return (
            self.state.status == AgentStatus.COMPLETED or
            self.state.status == AgentStatus.ERROR
        )

    def _create_new_state(self) -> None:
        """Создает новое состояние агента."""
        self.state = AgentState()

    def complete_state(self) -> None:
        """Завершает текущее состояние агента."""
        if self.state:
            self.state.complete_task()

    def clear_history(self) -> None:
        """Очищает историю диалога и контекст агента."""
        self.message_history = []

    def __del__(self):
        """Очищает ресурсы при удалении."""
        if self._browser:
            self._browser.close()

    def update_progress(self, progress: float, status: Optional[AgentStatus] = None) -> None:
        """Обновляет прогресс выполнения.
        
        Args:
            progress: Значение прогресса (0.0 - 1.0)
            status: Новый статус (опционально)
            
        Raises:
            ValueError: Если параметры некорректны
        """
        if not isinstance(progress, float) or not 0 <= progress <= 1:
            raise ValueError("Прогресс должен быть числом от 0 до 1")
        if status is not None and not isinstance(status, AgentStatus):
            raise ValueError("Статус должен быть экземпляром AgentStatus")
            
        if self.state:
            self.state.update_progress(progress, status)

    def increment_steps(self, count: int = 1) -> None:
        """Увеличивает счетчик выполненных шагов.
        
        Args:
            count: Количество шагов для добавления
            
        Raises:
            ValueError: Если параметр некорректен
        """
        if not isinstance(count, int) or count < 1:
            raise ValueError("Количество шагов должно быть положительным целым числом")
            
        if self.state:
            self.state.increment_steps(count)

    def update_metrics(self, tokens: int = 0, api_calls: int = 0, memory_usage: int = 0) -> None:
        """Обновляет метрики агента."""
        if self.state:
            self.state.update_metrics(tokens, api_calls, memory_usage)

    def _parse_validation_response(self, response: str) -> Dict[str, Any]:
        """Парсит ответ валидации."""
        try:
            # Ищем оценку
            score_match = re.search(r"Оценка:\s*(\d+)", response)
            score = int(score_match.group(1)) if score_match else 0
            
            # Ищем проблемы
            issues_match = re.findall(r"Проблема:\s*(.*?)(?=Рекомендация:|$)", response, re.DOTALL)
            issues = [issue.strip() for issue in issues_match if issue.strip()]
            
            # Ищем рекомендации
            suggestions_match = re.findall(r"Рекомендация:\s*(.*?)(?=Оценка:|$)", response, re.DOTALL)
            suggestions = [suggestion.strip() for suggestion in suggestions_match if suggestion.strip()]
            
            # Определяем валидность
            is_valid = score >= 70
            
            return {
                "score": score,
                "issues": issues,
                "suggestions": suggestions,
                "is_valid": is_valid
            }
        except Exception as e:
            logger.error(f"Ошибка при парсинге ответа валидации: {e}")
            return {
                "score": 0,
                "issues": [f"Ошибка парсинга: {str(e)}"],
                "suggestions": [],
                "is_valid": False
            }

    def _parse_analysis_response(self, response: str) -> Dict[str, Any]:
        """Парсит ответ анализа."""
        try:
            # Ищем оценку
            score_match = re.search(r"score:\s*(\d+)", response)
            score = int(score_match.group(1)) if score_match else 0
            
            # Ищем сильные стороны
            strengths_match = re.findall(r"strengths:\s*\[(.*?)\]", response)
            strengths = [s.strip() for s in strengths_match[0].split(",")] if strengths_match else []
            
            # Ищем слабые стороны
            weaknesses_match = re.findall(r"weaknesses:\s*\[(.*?)\]", response)
            weaknesses = [w.strip() for w in weaknesses_match[0].split(",")] if weaknesses_match else []
            
            # Ищем предложения
            suggestions_match = re.findall(r"suggestions:\s*\[(.*?)\]", response)
            suggestions = [s.strip() for s in suggestions_match[0].split(",")] if suggestions_match else []
            
            # Ищем достижения
            achievements_match = re.findall(r"achievements:\s*\[(.*?)\]", response)
            achievements = [a.strip() for a in achievements_match[0].split(",")] if achievements_match else []
            
            # Ищем влияние
            impact_match = re.search(r"impact:\s*(\d+)", response)
            impact = int(impact_match.group(1)) if impact_match else 0
            
            # Ищем уроки
            lessons_match = re.findall(r"lessons:\s*\[(.*?)\]", response)
            lessons = [l.strip() for l in lessons_match[0].split(",")] if lessons_match else []
            
            # Ищем инновации
            innovations_match = re.findall(r"innovations:\s*\[(.*?)\]", response)
            innovations = [i.strip() for i in innovations_match[0].split(",")] if innovations_match else []
            
            # Ищем преимущества
            benefits_match = re.findall(r"benefits:\s*\[(.*?)\]", response)
            benefits = [b.strip() for b in benefits_match[0].split(",")] if benefits_match else []
            
            # Ищем потенциал
            potential_match = re.findall(r"potential:\s*\[(.*?)\]", response)
            potential = [p.strip() for p in potential_match[0].split(",")] if potential_match else []
            
            # Ищем эффективность
            efficiency_match = re.findall(r"efficiency:\s*\[(.*?)\]", response)
            efficiency = [e.strip() for e in efficiency_match[0].split(",")] if efficiency_match else []
            
            # Ищем улучшения
            improvements_match = re.findall(r"improvements:\s*\[(.*?)\]", response)
            improvements = [i.strip() for i in improvements_match[0].split(",")] if improvements_match else []
            
            # Ищем рекомендации
            recommendations_match = re.findall(r"recommendations:\s*\[(.*?)\]", response)
            recommendations = [r.strip() for r in recommendations_match[0].split(",")] if recommendations_match else []
            
            # Ищем ключевые моменты
            key_points_match = re.findall(r"key_points:\s*\[(.*?)\]", response)
            key_points = [k.strip() for k in key_points_match[0].split(",")] if key_points_match else []
            
            # Ищем серьезность
            severity_match = re.search(r"severity:\s*(\d+)", response)
            severity = int(severity_match.group(1)) if severity_match else 0
            
            return {
                "score": score,
                "strengths": strengths,
                "weaknesses": weaknesses,
                "suggestions": suggestions,
                "achievements": achievements,
                "impact": impact,
                "lessons": lessons,
                "innovations": innovations,
                "benefits": benefits,
                "potential": potential,
                "efficiency": efficiency,
                "improvements": improvements,
                "recommendations": recommendations,
                "key_points": key_points,
                "severity": severity
            }
        except Exception as e:
            logger.error(f"Ошибка при парсинге ответа анализа: {e}")
            return {
                "score": 0,
                "strengths": [],
                "weaknesses": [],
                "suggestions": [],
                "achievements": [],
                "impact": 0,
                "lessons": [],
                "innovations": [],
                "benefits": [],
                "potential": [],
                "efficiency": [],
                "improvements": [],
                "recommendations": [],
                "key_points": [],
                "severity": 0
            }

    def _parse_relevance_response(self, response: str) -> Dict[str, Any]:
        """Парсит ответ проверки релевантности."""
        try:
            # Ищем флаг релевантности
            is_relevant_match = re.search(r"is_relevant:\s*(true|false)", response.lower())
            is_relevant = is_relevant_match.group(1) == "true" if is_relevant_match else False
            
            # Ищем предложения
            suggestions_match = re.findall(r"suggestions:\s*\[(.*?)\]", response)
            suggestions = [s.strip() for s in suggestions_match[0].split(",")] if suggestions_match else []
            
            return {
                "is_relevant": is_relevant,
                "suggestions": suggestions
            }
        except Exception as e:
            logger.error(f"Ошибка при парсинге ответа релевантности: {e}")
            return {
                "is_relevant": False,
                "suggestions": []
            }

class PlannerAgent(BaseAgent):
    """Агент для планирования и анализа запросов пользователя."""
    
    def __init__(self, name: str, system_prompt: str, model_name: str, client: OllamaClient):
        super().__init__(name, system_prompt or PLANNER_PROMPT, model_name, client)

    def create_detailed_plan(self, user_query: str) -> str:
        """Создает детальный план действий на основе запроса пользователя."""
        self.update_state(AgentStatus.PLANNING, "Creating detailed plan", total_steps=1)
        try:
            plan = self._generate_plan(user_query)
            self.complete_state()
            return plan.strip()
        except Exception as e:
            self.state.set_error(str(e))
            raise

    def _generate_plan(self, user_query: str) -> str:
        """Генерирует план действий с помощью LLM."""
        prompt = self.build_prompt(user_query)
        plan = self.client.generate(prompt=prompt, model=self.model_name, stream=False)
        return ''.join(plan) if not isinstance(plan, str) else plan

    def analyze_query(self, user_query: str) -> str:
        """Анализирует запрос пользователя и определяет необходимые действия."""
        self.update_state(AgentStatus.PLANNING, "Analyzing query", total_steps=1)
        try:
            analysis = self._generate_analysis(user_query)
            self.complete_state()
            return analysis.strip() or user_query
        except Exception as e:
            self.state.set_error(str(e))
            raise

    def _generate_analysis(self, user_query: str) -> str:
        """Генерирует анализ запроса с помощью LLM."""
        prompt = self._build_analysis_prompt(user_query)
        analysis = self.client.generate(prompt=prompt, model=self.model_name, stream=False)
        return ''.join(analysis) if not isinstance(analysis, str) else analysis

    def _build_analysis_prompt(self, user_query: str) -> str:
        """Создает промпт для анализа запроса."""
        return (
            f"Проанализируй следующий запрос и определи необходимые действия. "
            f"Учитывай следующие аспекты:\n"
            f"1. Тип запроса (информационный, практический, аналитический)\n"
            f"2. Необходимость поиска информации (локальная/интернет)\n"
            f"3. Необходимость взаимодействия с веб-интерфейсом\n"
            f"4. Необходимость анализа визуального контента\n"
            f"5. Необходимость выполнения системных команд\n\n"
            f"Запрос: {user_query}\n\n"
            f"Ответь в формате:\n"
            f"Тип запроса: <тип>\n"
            f"Необходимые действия:\n"
            f"- <действие 1>\n"
            f"- <действие 2>\n"
            f"Рекомендуемые инструменты:\n"
            f"- <инструмент 1>: <причина>\n"
            f"- <инструмент 2>: <причина>\n"
            f"Финальная инструкция: <инструкция>"
        )

    def generate_instruction(
        self, 
        user_query: str,
        vector_store: SimpleVectorStore,
        stream=False,
        **ollama_opts
    ) -> Union[str, Generator[str, None, None]]:
        """Генерирует инструкцию на основе запроса пользователя."""
        self.update_state(AgentStatus.PLANNING, "Анализ запроса", total_steps=4)
        try:
            # Шаг 1: Анализ запроса
            self.update_progress(0.25, AgentStatus.PLANNING)
            analysis = self.analyze_query(user_query)
            self.increment_steps()
            
            # Шаг 2: Создание детального плана
            self.update_progress(0.5, AgentStatus.PLANNING)
            detailed_plan = self.create_detailed_plan(user_query)
            self.increment_steps()
            
            # Шаг 3: Определение плана выполнения
            self.update_progress(0.75, AgentStatus.PLANNING)
            plan = self._determine_execution_plan(user_query, vector_store)
            self.increment_steps()
            
            # Шаг 4: Формирование финальной инструкции
            self.update_progress(1.0, AgentStatus.PLANNING)
            instruction = self._build_final_instruction(analysis, detailed_plan, plan)
            self.increment_steps()
            
            self.complete_state()
            
            if not stream:
                return instruction
            else:
                def gen_plan():
                    yield instruction
                return gen_plan()
                
        except Exception as e:
            self.state.set_error(str(e))
            raise

    def _validate_plan(self, plan: str, user_query: str) -> Dict[str, Any]:
        """Проверяет качество плана."""
        try:
            validation_prompt = f"""
            Проверьте качество следующего плана действий для запроса пользователя:
            
            Запрос пользователя:
            {user_query}
            
            План действий:
            {plan}
            
            Оцените план по следующим критериям:
            1. Полнота (все ли необходимые шаги включены)
            2. Логичность (шаги в правильном порядке)
            3. Выполнимость (все шаги реализуемы)
            4. Релевантность (соответствие запросу)
            5. Эффективность (оптимальность решения)
            
            Предоставьте ответ в формате:
            Оценка: [0-100]
            Проблема: [описание проблемы]
            Рекомендация: [конкретное предложение по улучшению]
            """
            
            response = self.client.generate(
                prompt=validation_prompt,
                model=self.model_name,
                stream=False
            )
            
            return self._parse_validation_response(response)
            
        except Exception as e:
            logger.error(f"Ошибка при валидации плана: {e}")
            return {
                "score": 0,
                "issues": [f"Ошибка валидации: {str(e)}"],
                "suggestions": [],
                "is_valid": False
            }

    def _improve_plan(self, plan: str, issues: List[str]) -> str:
        """Улучшает план на основе выявленных проблем."""
        try:
            improvement_prompt = f"""
            Улучшите следующий план действий, учитывая выявленные проблемы:
            
            Исходный план:
            {plan}
            
            Выявленные проблемы:
            {chr(10).join(f"- {issue}" for issue in issues)}
            
            Предоставь улучшенную версию плана, которая:
            1. Устраняет все выявленные проблемы
            2. Сохраняет логическую структуру
            3. Добавляет недостающие шаги
            4. Улучшает эффективность
            5. Учитывает граничные случаи
            """
            
            improved_plan = self.client.generate(
                prompt=improvement_prompt,
                model=self.model_name,
                stream=False
            )
            
            return improved_plan
            
        except Exception as e:
            logger.error(f"Ошибка при улучшении плана: {e}")
            return plan

    def _check_relevance(self, instruction: str, user_query: str) -> Dict[str, Any]:
        """Проверяет релевантность инструкции запросу."""
        relevance_prompt = f"""
        Проверь релевантность следующей инструкции запросу: "{user_query}"
        
        Инструкция:
        {instruction}
        
        Проверь:
        1. Соответствие запросу
        2. Актуальность информации
        3. Полноту ответа
        4. Практическую применимость
        
        Ответь в формате:
        is_relevant: true/false
        suggestions: [список предложений по улучшению]
        """
        
        response = self.client.generate(
            prompt=relevance_prompt,
            model=self.model_name,
            stream=False
        )
        return self._parse_relevance_response(response)

    def _adjust_for_relevance(self, instruction: str, suggestions: List[str]) -> str:
        """Корректирует инструкцию для повышения релевантности."""
        adjustment_prompt = f"""
        Улучши следующую инструкцию, учитывая предложения:
        
        Инструкция:
        {instruction}
        
        Предложения:
        {', '.join(suggestions)}
        
        Предоставь улучшенную версию инструкции.
        """
        
        improved_instruction = self.client.generate(
            prompt=adjustment_prompt,
            model=self.model_name,
            stream=False
        )
        return improved_instruction

    def _determine_execution_plan(self, user_query: str, vector_store: SimpleVectorStore) -> str:
        """Определяет план выполнения на основе запроса и доступных данных."""
        if any(cmd in user_query.lower() for cmd in ["ducksearch:", "browser:", "visual:", "cmd:", "ls:"]):
            return user_query
            
        local_hits = vector_store.search(user_query, k=1)
        return f"ducksearch: {user_query}" if not local_hits else user_query

    def _build_final_instruction(self, analysis: str, detailed_plan: str, plan: str) -> str:
        """Создает финальную инструкцию из всех компонентов."""
        return (
            f"Анализ запроса:\n{analysis}\n\n"
            f"План действий:\n{detailed_plan}\n\n"
            f"Инструкция для выполнения:\n{plan}"
        )

class ExecutorAgent(BaseAgent):
    """Агент для выполнения инструкций и команд.
    
    Этот класс отвечает за:
    - Выполнение команд и инструкций
    - Валидацию результатов
    - Улучшение результатов
    - Проверку релевантности
    - Обработку ошибок
    """
    
    def __init__(self, name: str, system_prompt: str, model_name: str, client: OllamaClient):
        """Инициализирует исполнительного агента.
        
        Args:
            name: Имя агента
            system_prompt: Системный промпт
            model_name: Имя модели LLM
            client: Клиент Ollama
            
        Raises:
            ValueError: Если параметры некорректны
        """
        super().__init__(name, system_prompt, model_name, client)
        
    def execute_instruction(
        self,
        instruction: str,
        vector_store: Optional[SimpleVectorStore] = None,
        stream: bool = False,
        **kwargs
    ) -> Union[str, Generator[str, None, None]]:
        """Выполняет инструкцию."""
        if not instruction or not isinstance(instruction, str):
            raise ValueError("Инструкция должна быть непустой строкой")
            
        try:
            # Обновляем состояние
            self.update_state(AgentStatus.EXECUTING, "Выполнение инструкции")
            
            # Извлекаем команду из инструкции
            command = self._extract_command(instruction)
            if not command:
                return "Ошибка: не удалось извлечь команду из инструкции"
            
            # Выполняем команду
            result = self._execute_command(command, vector_store)
            if not result:
                self.update_state(AgentStatus.ERROR, "Пустой результат выполнения")
                return "Ошибка: команда не вернула результат"
            
            # Обновляем состояние
            self.update_state(AgentStatus.COMPLETED, "Инструкция выполнена")
            
            # Возвращаем результат
            if stream:
                return self._stream_result(result)
            return result
            
        except Exception as e:
            error_msg = str(e)
            self.update_state(AgentStatus.ERROR, f"Ошибка выполнения: {error_msg}")
            logger.error(f"Ошибка при выполнении инструкции: {error_msg}")
            return f"Ошибка выполнения: {error_msg}"

    def _extract_command(self, instruction: str) -> Optional[str]:
        """Извлекает команду из инструкции."""
        try:
            # Очищаем инструкцию
            instruction = instruction.strip()
            
            # Ищем команды в тексте
            command_patterns = [
                (r'ducksearch:([^\n]+)', lambda m: f"ducksearch:{m.group(1).strip()}"),
                (r'browser:(url=[^;]+(?:;(?:click|type|screenshot)=[^;]+)*)', lambda m: f"browser:{m.group(1)}"),
                (r'visual:((?:analyze|describe|ocr)=[^;]+(?:;(?:analyze|describe|ocr)=[^;]+)*)', lambda m: f"visual:{m.group(1)}"),
                (r'cmd:([^\n]+)', lambda m: f"cmd:{m.group(1).strip()}"),
                (r'ls:([^\n]+)', lambda m: f"ls:{m.group(1).strip()}")
            ]
            
            for pattern, formatter in command_patterns:
                match = re.search(pattern, instruction, re.IGNORECASE)
                if match:
                    return formatter(match)
            
            # Если не нашли точного совпадения, ищем ключевые слова
            keywords = {
                'ducksearch': 'поиск',
                'browser': 'браузер',
                'visual': 'визуальный',
                'cmd': 'команда',
                'ls': 'директория'
            }
            
            for cmd, keyword in keywords.items():
                if any(word in instruction.lower() for word in [cmd, keyword]):
                    # Очищаем текст от лишнего и формируем команду
                    text = re.sub(r'^.*?(?:' + cmd + '|' + keyword + r')\s*:?\s*', '', instruction, flags=re.IGNORECASE)
                    return f"{cmd}:{text.strip()}"
            
            # Если не нашли команду, возвращаем как поисковый запрос
            return f"ducksearch:{instruction}"
            
        except Exception as e:
            logger.error(f"Ошибка при извлечении команды: {e}")
            return None

    def _execute_command(self, command: str, vector_store: Optional[SimpleVectorStore] = None) -> Optional[str]:
        """Выполняет команду."""
        try:
            # Определяем тип команды
            command_type = command.split(':', 1)[0].lower() if ':' in command else 'ducksearch'
            command_content = command.split(':', 1)[1] if ':' in command else command
            
            # Выполняем соответствующую команду
            if command_type == "ducksearch":
                return self._execute_ducksearch(command_content)
            elif command_type == "browser":
                return self._execute_browser_actions(command_content)
            elif command_type == "visual":
                return self._execute_visual_actions(command_content)
            elif command_type == "cmd":
                return self._execute_shell_command(command_content)
            elif command_type == "ls":
                return self._execute_list_directory(command_content)
            else:
                return self._execute_ducksearch(command)  # Fallback to search
                
        except Exception as e:
            logger.error(f"Ошибка при выполнении команды: {e}")
            return f"Ошибка выполнения: {str(e)}"

    def _execute_ducksearch(self, query: str) -> str:
        """Выполняет поиск через DuckDuckGo с фолбэком на браузерный поиск."""
        try:
            # Проверяем и очищаем запрос
            query = query.strip()
            if not query:
                return "Ошибка: пустой поисковый запрос"
            
            # Сначала пробуем через DuckDuckGo API
            results = self._try_ddg_search(query)
            if results and not results.startswith("Ошибка"):
                return results
                
            # Если DuckDuckGo не сработал, используем браузерный поиск
            logger.info("DuckDuckGo поиск не удался, переключаемся на браузерный поиск")
            return self._try_browser_search(query)
            
        except Exception as e:
            logger.error(f"Ошибка при выполнении поиска: {e}")
            return f"Ошибка поиска: {str(e)}"

    def _try_ddg_search(self, query: str) -> str:
        """Пытается выполнить поиск через DuckDuckGo API."""
        try:
            from duckduckgo_search import DDGS
            from duckduckgo_search.exceptions import DuckDuckGoSearchException
            import time
            
            max_retries = 3
            base_delay = 2
            
            for attempt in range(max_retries):
                try:
                    with DDGS() as ddgs:
                        results = list(ddgs.text(
                            query,
                            max_results=5,
                            region='ru-ru'
                        ))
                        
                        if not results:
                            return "Поиск не дал результатов"
                        
                        formatted_results = []
                        for i, result in enumerate(results, 1):
                            formatted_results.append(
                                f"{i}. {result.get('title', '')}\n"
                                f"   URL: {result.get('link', '')}\n"
                                f"   {result.get('body', '')}\n"
                            )
                        
                        return "\n".join(formatted_results)
                        
                except DuckDuckGoSearchException as ddg_error:
                    if "Ratelimit" in str(ddg_error):
                        if attempt < max_retries - 1:
                            delay = base_delay * (2 ** attempt)
                            logger.warning(f"Rate limit достигнут, ожидание {delay} секунд...")
                            time.sleep(delay)
                            continue
                    logger.error(f"Ошибка DuckDuckGo поиска: {ddg_error}")
                    return f"Ошибка: {str(ddg_error)}"
                    
        except ImportError:
            return "Ошибка: пакет duckduckgo-search не установлен"
        except Exception as e:
            logger.error(f"Ошибка при DuckDuckGo поиске: {e}")
            return f"Ошибка поиска: {str(e)}"

    def _try_browser_search(self, query: str) -> str:
        """Выполняет поиск через браузер."""
        browser = None
        try:
            # Конфигурация таймаутов и повторных попыток
            config = {
                'browser_launch_timeout': 30000,  # 30 секунд
                'page_load_timeout': 45000,      # 45 секунд
                'navigation_timeout': 30000,     # 30 секунд
                'element_timeout': 15000,        # 15 секунд
                'max_retries': 3,
                'base_delay': 2,                 # базовая задержка в секундах
            }
            
            # Инициализация браузера с повторными попытками
            for attempt in range(config['max_retries']):
                try:
                    if not self._browser:
                        self._browser = PlaywrightBrowser(
                            headless=True,
                            timeout=config['browser_launch_timeout']
                        )
                        self._browser.launch()
                    browser = self._browser
                    break
                except Exception as e:
                    if attempt == config['max_retries'] - 1:
                        raise Exception(
                            f"Не удалось запустить браузер после {config['max_retries']} "
                            f"попыток: {e}"
                        )
                    delay = config['base_delay'] * (2 ** attempt)
                    logger.warning(f"Попытка {attempt + 1}: ошибка запуска браузера. "
                                 f"Повтор через {delay} сек.")
                    time.sleep(delay)
            
            # Формируем URL для поиска
            encoded_query = urllib.parse.quote(query)
            search_url = f"https://duckduckgo.com/?q={encoded_query}&kl=ru-ru&k1=-1"
            
            # Переходим на страницу поиска с повторными попытками
            for attempt in range(config['max_retries']):
                try:
                    # Устанавливаем таймауты для навигации
                    browser.page.set_default_timeout(config['navigation_timeout'])
                    browser.page.set_default_navigation_timeout(config['page_load_timeout'])
                    
                    # Переходим по URL
                    browser.goto(search_url)
                    
                    # Ждем загрузки результатов с таймаутом
                    self._wait_for_search_results(
                        browser.page,
                        timeout=config['element_timeout']
                    )
                    
                    # Проверяем наличие результатов
                    results = self._extract_search_results()
                    if results:
                        return "\n".join(results)
                        
                    # Если результатов нет, но нет и ошибок - пробуем еще раз
                    if attempt < config['max_retries'] - 1:
                        delay = config['base_delay'] * (2 ** attempt)
                        logger.warning(f"Попытка {attempt + 1}: результаты не найдены. "
                                     f"Повтор через {delay} сек.")
                        time.sleep(delay)
                        continue
                        
                    return "Поиск не дал результатов"
                    
                except Exception as e:
                    if attempt == config['max_retries'] - 1:
                        raise Exception(f"Не удалось загрузить результаты поиска: {e}")
                    delay = config['base_delay'] * (2 ** attempt)
                    logger.warning(f"Попытка {attempt + 1}: ошибка загрузки. "
                                 f"Повтор через {delay} сек.")
                    time.sleep(delay)
            
            return "Не удалось получить результаты поиска"
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Ошибка при браузерном поиске: {error_msg}")
            return f"Ошибка поиска: {error_msg}"
            
        finally:
            # Очищаем ресурсы в случае ошибки
            if browser and browser != self._browser:
                try:
                    browser.close()
                except Exception as e:
                    logger.error(f"Ошибка при закрытии браузера: {e}")
                    
    def _wait_for_search_results(self, page, timeout: int = 15000) -> None:
        """Ожидает загрузки результатов поиска с таймаутом."""
        try:
            # Список селекторов для проверки
            result_selectors = [
                '.results',
                '.result__body',
                '#links',
                '.serp__results'
            ]
            
            # Ждем появления хотя бы одного селектора
            for selector in result_selectors:
                try:
                    page.wait_for_selector(
                        selector,
                        state='visible',
                        timeout=timeout
                    )
                    logger.debug(f"Найден селектор результатов: {selector}")
                    return
                except Exception:
                    continue
                    
            # Если ни один селектор не найден
            raise TimeoutError("Не найдены селекторы результатов поиска")
            
        except Exception as e:
            logger.error(f"Ошибка при ожидании результатов: {e}")
            raise

    def _extract_search_results(self) -> List[str]:
        """Извлекает результаты поиска со страницы."""
        results = []
        try:
            # Актуальные селекторы DuckDuckGo
            selectors = {
                'result': '.result__body',
                'title': '.result__title',
                'link': '.result__url',
                'snippet': '.result__snippet'
            }
            
            # Получаем все результаты
            result_elements = self._browser.page.query_selector_all(selectors['result'])
            
            # Обрабатываем первые 5 результатов
            for i, result in enumerate(result_elements[:5], 1):
                try:
                    # Извлекаем компоненты результата
                    title_element = result.query_selector(selectors['title'])
                    link_element = result.query_selector(selectors['link'])
                    snippet_element = result.query_selector(selectors['snippet'])
                    
                    # Проверяем наличие всех элементов
                    if all([title_element, link_element, snippet_element]):
                        title = title_element.inner_text().strip()
                        link = link_element.get_attribute('href') or link_element.inner_text().strip()
                        snippet = snippet_element.inner_text().strip()
                        
                        # Форматируем результат
                        formatted_result = (
                            f"{i}. {title}\n"
                            f"   URL: {link}\n"
                            f"   {snippet}\n"
                        )
                        results.append(formatted_result)
                except Exception as e:
                    logger.warning(f"Не удалось извлечь результат {i}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Ошибка при извлечении результатов: {e}")
            
        return results

    def _cleanup_browser(self) -> None:
        """Закрывает браузер и очищает ресурсы."""
        if self._browser:
            try:
                self._browser.close()
            except Exception as e:
                logger.error(f"Ошибка при закрытии браузера: {e}")
            finally:
                self._browser = None

    def _execute_search(self, instruction: str) -> Optional[str]:
        """Выполняет общий поиск.
        
        Args:
            instruction: Инструкция с поисковым запросом
            
        Returns:
            Результаты поиска или None в случае ошибки
        """
        try:
            # Извлекаем поисковый запрос
            query = instruction.replace("search", "").strip()
            if not query:
                return "Ошибка: пустой поисковый запрос"
            
            # Сначала пробуем поиск в векторном хранилище
            vector_results = self._search_vector_store(query)
            
            # Затем выполняем веб-поиск
            web_results = self._try_ddg_search(query)
            
            # Если оба поиска не дали результатов
            if not vector_results and not web_results:
                return "Поиск не дал результатов"
            
            # Форматируем и объединяем результаты
            formatted_results = []
            
            if vector_results:
                formatted_results.append("Результаты из локального хранилища:")
                formatted_results.append("-" * 50)
                formatted_results.append(self._format_vector_store_results(vector_results))
                formatted_results.append("\n")
            
            if web_results:
                formatted_results.append("Результаты веб-поиска:")
                formatted_results.append("-" * 50)
                formatted_results.append(web_results)
            
            return "\n".join(formatted_results)
            
        except Exception as e:
            logger.error(f"Ошибка при выполнении поиска: {e}")
            return f"Ошибка поиска: {str(e)}"
            
    def _search_vector_store(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """Выполняет поиск в векторном хранилище."""
        try:
            if not hasattr(self, '_vector_store'):
                self._vector_store = SimpleVectorStore()
            
            results = self._vector_store.search(query, k=k)
            return results
        except Exception as e:
            logger.error(f"Ошибка при поиске в векторном хранилище: {e}")
            return []
            
    def _format_vector_store_results(self, results: List[Dict[str, Any]]) -> str:
        """Форматирует результаты из векторного хранилища."""
        formatted_results = []
        for i, result in enumerate(results, 1):
            formatted_results.append(
                f"{i}. Релевантность: {1 - result.get('distance', 0):.2f}\n"
                f"   {result.get('text', '')}\n"
            )
        return "\n".join(formatted_results)
    
    def _execute_shell_command(self, instruction: str) -> Optional[str]:
        """Выполняет shell-команду.
        
        Args:
            instruction: Инструкция с shell-командой
            
        Returns:
            Результат выполнения команды или None в случае ошибки
        """
        try:
            # Извлекаем команду
            command = instruction.replace("cmd", "").strip()
            if not command:
                return "Ошибка: пустая команда"
            
            # Проверяем безопасность команды
            if not self._is_safe_command(command):
                return "Ошибка: небезопасная команда"
            
            # Выполняем команду
            result = self._run_shell_command(command)
            if not result:
                return "Команда не вернула результат"
            
            return result
            
        except Exception as e:
            logger.error(f"Ошибка при выполнении shell-команды: {e}")
            return None
    
    def _is_safe_command(self, command: str) -> bool:
        """Проверяет безопасность команды."""
        # Список небезопасных команд
        unsafe_commands = {
            "rm": "Удаление файлов",
            "del": "Удаление файлов",
            "format": "Форматирование диска",
            "mkfs": "Создание файловой системы",
            "dd": "Побитовое копирование",
            "chmod": "Изменение прав доступа",
            "chown": "Изменение владельца",
            "sudo": "Повышение привилегий",
            "su": "Смена пользователя"
        }
        
        # Опасные паттерны
        unsafe_patterns = [
            r">.*",  # Перенаправление вывода
            r">>.*", # Добавление к файлу
            r"\|.*", # Пайпы
            r"&.*",  # Фоновые процессы
            r";.*",  # Последовательное выполнение
            r"&&.*"  # Условное выполнение
        ]
        
        # Проверяем команду
        command = command.lower().strip()
        
        # Проверяем опасные команды
        for unsafe_cmd in unsafe_commands:
            if unsafe_cmd in command.split():
                return False
        
        # Проверяем опасные паттерны
        for pattern in unsafe_patterns:
            if re.search(pattern, command):
                return False
        
        return True
    
    def _run_shell_command(self, command: str) -> Optional[str]:
        """Запускает shell-команду.
        
        Args:
            command: Команда для выполнения
            
        Returns:
            Результат выполнения команды
        """
        try:
            # Здесь должна быть реализация выполнения shell-команд
            # Временная заглушка
            return f"Выполнена команда: {command}"
        except Exception as e:
            logger.error(f"Ошибка при запуске shell-команды: {e}")
            return None

    def _execute_list_directory(self, instruction: str) -> Optional[str]:
        """Выполняет команду просмотра директории.
        
        Args:
            instruction: Инструкция с путем к директории
            
        Returns:
            Содержимое директории или None в случае ошибки
        """
        try:
            # Извлекаем путь
            path = instruction.replace("ls", "").strip()
            if not path:
                path = "."
            
            # Проверяем безопасность пути
            if not self._is_safe_path(path):
                return "Ошибка: небезопасный путь"
            
            # Получаем содержимое директории
            contents = self._get_directory_contents(path)
            if not contents:
                return "Директория пуста или недоступна"
            
            # Форматируем результат
            return self._format_directory_contents(contents)
            
        except Exception as e:
            logger.error(f"Ошибка при просмотре директории: {e}")
            return None
    
    def _is_safe_path(self, path: str) -> bool:
        """Проверяет безопасность пути."""
        try:
            # Нормализуем и получаем абсолютный путь
            abs_path = os.path.abspath(os.path.normpath(path))
            work_dir = os.path.abspath(os.getcwd())
            
            # Проверяем, что путь находится внутри рабочей директории
            if not abs_path.startswith(work_dir):
                return False
            
            # Проверяем на наличие опасных компонентов
            unsafe_components = ["..", "~", "$", "`", "*", "?", "{", "}", "[", "]"]
            return not any(component in path for component in unsafe_components)
        except Exception:
            return False
    
    def _get_directory_contents(self, path: str) -> List[Dict[str, Any]]:
        """Получает содержимое директории.
        
        Args:
            path: Путь к директории
            
        Returns:
            Список элементов директории
        """
        try:
            contents = []
            for item in os.listdir(path):
                full_path = os.path.join(path, item)
                contents.append({
                    "name": item,
                    "type": "directory" if os.path.isdir(full_path) else "file",
                    "size": os.path.getsize(full_path) if os.path.isfile(full_path) else 0,
                    "modified": os.path.getmtime(full_path)
                })
            return contents
        except Exception as e:
            logger.error(f"Ошибка при получении содержимого директории: {e}")
            return []
    
    def _format_directory_contents(self, contents: List[Dict[str, Any]]) -> str:
        """Форматирует содержимое директории.
        
        Args:
            contents: Список элементов директории
            
        Returns:
            Отформатированный текст содержимого
        """
        try:
            formatted_contents = []
            for item in contents:
                size = self._format_size(item["size"])
                modified = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(item["modified"]))
                formatted_contents.append(
                    f"{item['type'][0]} {item['name']:<30} {size:>10} {modified}"
                )
            return "\n".join(formatted_contents)
        except Exception as e:
            logger.error(f"Ошибка при форматировании содержимого директории: {e}")
            return "Ошибка форматирования"
    
    def _format_size(self, size: int) -> str:
        """Форматирует размер файла.
        
        Args:
            size: Размер в байтах
            
        Returns:
            Отформатированный размер
        """
        for unit in ["B", "KB", "MB", "GB"]:
            if size < 1024:
                return f"{size:.1f}{unit}"
            size /= 1024
        return f"{size:.1f}TB"
    
    def _execute_browser_actions(self, actions: str) -> Optional[str]:
        """Выполняет действия в браузере."""
        max_retries = 3
        retry_delay = 1.0
        
        try:
            # Инициализируем браузер если нужно
            if not self._browser:
                for attempt in range(max_retries):
                    try:
                        self._browser = PlaywrightBrowser(headless=True, timeout=30000)
                        self._browser.launch()
                        break
                    except Exception as e:
                        if attempt == max_retries - 1:
                            return f"Ошибка: не удалось инициализировать браузер после {max_retries} попыток: {e}"
                        time.sleep(retry_delay * (2 ** attempt))
            
            results = []
            # Разбираем действия
            for action in actions.split(';'):
                action = action.strip()
                if not action:
                    continue
                
                try:
                    if action.startswith('url='):
                        url = action.split('=', 1)[1]
                        self._browser.goto(url)
                        results.append(f"✓ Открыта страница: {url}")
                        results.append(f"  Заголовок: {self._browser.get_page_title()}")
                        results.append(f"  SSL: {'Да' if self._browser.check_ssl() else 'Нет'}")
                        
                    elif action.startswith('click='):
                        selector = action.split('=', 1)[1]
                        self._browser.click(selector)
                        results.append(f"✓ Клик по элементу: {selector}")
                        
                    elif action.startswith('type='):
                        _, params = action.split('=', 1)
                        selector, text = params.split(':', 1)
                        self._browser.type_text(selector, text)
                        results.append(f"✓ Введен текст в элемент: {selector}")
                        
                    elif action.startswith('screenshot='):
                        path = action.split('=', 1)[1]
                        screenshot_path = self._browser.screenshot(path)
                        results.append(f"✓ Создан скриншот: {screenshot_path}")
                        
                    else:
                        results.append(f"⚠ Неизвестное действие: {action}")
                        
                except Exception as action_error:
                    results.append(f"✗ Ошибка при выполнении действия {action}: {str(action_error)}")
                    
            return "\n".join(results)
            
        except Exception as e:
            logger.error(f"Ошибка при выполнении действий в браузере: {e}")
            return f"Ошибка: {str(e)}"
        finally:
            self._cleanup_browser()
    
    def _execute_visual_actions(self, instruction: str) -> Optional[str]:
        """Выполняет визуальные действия.
        
        Args:
            instruction: Инструкция с действиями
            
        Returns:
            Результат действий или None в случае ошибки
        """
        try:
            # Извлекаем действия
            actions = instruction.replace("visual", "").strip()
            if not actions:
                return "Ошибка: пустая инструкция"
            
            # Выполняем действия
            result = self._perform_visual_actions(actions)
            if not result:
                return "Действия не дали результата"
            
            return result
            
        except Exception as e:
            logger.error(f"Ошибка при выполнении визуальных действий: {e}")
            return None
    
    def _perform_visual_actions(self, actions: str) -> Optional[str]:
        """Выполняет визуальные действия.
        
        Args:
            actions: Строка с действиями в формате:
                    analyze=<path> - анализ изображения
                    describe=<path> - описание изображения
                    ocr=<path> - распознавание текста
                    
        Returns:
            Результат действий или None в случае ошибки
        """
        try:
            results = []
            for action in actions.split(';'):
                action = action.strip()
                if not action:
                    continue
                
                try:
                    # Извлекаем путь и проверяем его
                    action_type, path = self._parse_visual_action(action)
                    if not path:
                        results.append(f"Ошибка: неверный формат действия {action}")
                        continue
                        
                    if not self._is_safe_path(path):
                        results.append(f"Ошибка: небезопасный путь {path}")
                        continue
                        
                    if not os.path.exists(path):
                        results.append(f"Ошибка: файл не существует {path}")
                        continue
                        
                    # Проверяем тип файла
                    if not self._is_valid_image(path):
                        results.append(f"Ошибка: неподдерживаемый тип файла {path}")
                        continue
                    
                    # Выполняем соответствующее действие
                    result = self._execute_visual_action(action_type, path)
                    if result:
                        results.append(result)
                    
                except Exception as action_error:
                    error_msg = str(action_error)
                    logger.error(f"Ошибка при выполнении действия {action}: {error_msg}")
                    results.append(f"Ошибка при выполнении {action}: {error_msg}")
                    continue
            
            if not results:
                return "Не удалось выполнить визуальные действия"
                
            return "\n\n".join(results)
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Ошибка при выполнении визуальных действий: {error_msg}")
            return f"Ошибка: {error_msg}"
            
    def _parse_visual_action(self, action: str) -> Tuple[str, str]:
        """Разбирает действие на тип и путь."""
        try:
            action_type, path = action.split('=', 1)
            if action_type not in ['analyze', 'describe', 'ocr']:
                raise ValueError(f"Неподдерживаемый тип действия: {action_type}")
            return action_type, path
        except Exception as e:
            raise ValueError(f"Неверный формат действия: {e}")
            
    def _is_valid_image(self, path: str) -> bool:
        """Проверяет, является ли файл поддерживаемым изображением."""
        try:
            from PIL import Image
            with Image.open(path) as img:
                return img.format in ['PNG', 'JPEG', 'JPG']
        except Exception:
            return False
            
    def _execute_visual_action(self, action_type: str, path: str) -> Optional[str]:
        """Выполняет конкретное визуальное действие."""
        try:
            prompts = {
                'analyze': "Проанализируйте это изображение и опишите его содержимое.",
                'describe': "Опишите подробно, что вы видите на этом изображении.",
                'ocr': "Прочитайте и извлеките весь текст с этого изображения."
            }
            
            # Устанавливаем таймаут для LLaVA
            timeout = 30  # секунды
            
            # Создаем future для выполнения с таймаутом
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    llava_analyze_screenshot_via_ollama_llava,
                    path,
                    prompts[action_type],
                    model="ollama/llava"
                )
                
                try:
                    result = future.result(timeout=timeout)
                    if not result:
                        raise ValueError("Пустой результат от LLaVA")
                    
                    action_labels = {
                        'analyze': "Анализ изображения",
                        'describe': "Описание изображения",
                        'ocr': "Распознанный текст из"
                    }
                    
                    return f"{action_labels[action_type]} {path}:\n{result}"
                    
                except concurrent.futures.TimeoutError:
                    raise TimeoutError(f"Превышено время ожидания ({timeout}с) при обработке {path}")
                    
        except Exception as e:
            logger.error(f"Ошибка при выполнении {action_type} для {path}: {e}")
            raise
    
    def _execute_general_command(self, instruction: str) -> Optional[str]:
        """Выполняет общую команду.
        
        Args:
            instruction: Инструкция с командой
            
        Returns:
            Результат выполнения команды или None в случае ошибки
        """
        try:
            # Анализируем команду
            command_type = self._analyze_command(instruction)
            
            # Выполняем соответствующее действие
            if command_type == "search":
                return self._execute_search(instruction)
            elif command_type == "browser":
                return self._execute_browser_actions(instruction)
            elif command_type == "visual":
                return self._execute_visual_actions(instruction)
            else:
                return self._execute_shell_command(instruction)
                
        except Exception as e:
            logger.error(f"Ошибка при выполнении общей команды: {e}")
            return None
    
    def _analyze_command(self, instruction: str) -> str:
        """Анализирует тип команды.
        
        Args:
            instruction: Инструкция для анализа
            
        Returns:
            Тип команды
        """
        try:
            # Нормализуем инструкцию
            instruction = instruction.lower().strip()
            
            # Определяем шаблоны команд
            command_patterns = {
                'search': {
                    'patterns': [
                        r'ducksearch:.*',
                        r'найти\s+.*',
                        r'поиск\s+.*',
                        r'search\s+.*',
                        r'find\s+.*',
                        r'query\s+.*'
                    ],
                    'keywords': [
                        'найти', 'поиск', 'search', 'find', 'query',
                        'информация', 'данные', 'узнать'
                    ]
                },
                'browser': {
                    'patterns': [
                        r'browser:.*',
                        r'открыть\s+https?://.*',
                        r'перейти\s+на\s+.*',
                        r'загрузить\s+страницу\s+.*'
                    ],
                    'keywords': [
                        'открыть', 'браузер', 'browser', 'url', 'сайт',
                        'страница', 'веб', 'web', 'http', 'https'
                    ]
                },
                'visual': {
                    'patterns': [
                        r'visual:.*',
                        r'analyze:.*',
                        r'describe:.*',
                        r'ocr:.*',
                        r'показать\s+.*\.(png|jpg|jpeg)',
                        r'распознать\s+.*\.(png|jpg|jpeg)'
                    ],
                    'keywords': [
                        'показать', 'визуальный', 'visual', 'image',
                        'картинка', 'фото', 'изображение', 'скриншот',
                        'анализ', 'распознать', 'прочитать'
                    ]
                },
                'system': {
                    'patterns': [
                        r'cmd:.*',
                        r'ls:.*',
                        r'dir\s+.*',
                        r'выполнить\s+команду\s+.*'
                    ],
                    'keywords': [
                        'команда', 'система', 'выполнить', 'запустить',
                        'директория', 'папка', 'файл', 'cmd', 'ls', 'dir'
                    ]
                }
            }
            
            # Проверяем точные шаблоны
            for cmd_type, patterns in command_patterns.items():
                for pattern in patterns['patterns']:
                    if re.match(pattern, instruction):
                        logger.debug(f"Найден точный шаблон {pattern} для типа {cmd_type}")
                        return cmd_type
            
            # Проверяем ключевые слова и их контекст
            keyword_scores = {cmd_type: 0 for cmd_type in command_patterns}
            
            for cmd_type, patterns in command_patterns.items():
                # Подсчитываем количество ключевых слов
                for keyword in patterns['keywords']:
                    if keyword in instruction:
                        # Увеличиваем счет на основе позиции слова
                        position_weight = 1.0
                        if instruction.startswith(keyword):
                            position_weight = 2.0  # Слово в начале важнее
                        keyword_scores[cmd_type] += position_weight
                        
                        # Учитываем контекст
                        context_before = instruction[:instruction.index(keyword)].strip()
                        context_after = instruction[instruction.index(keyword) + len(keyword):].strip()
                        
                        # Анализируем контекст
                        if self._is_relevant_context(context_before, context_after, cmd_type):
                            keyword_scores[cmd_type] += 0.5
            
            # Выбираем тип с наивысшим счетом
            if any(score > 0 for score in keyword_scores.values()):
                best_type = max(keyword_scores.items(), key=lambda x: x[1])[0]
                logger.debug(f"Выбран тип {best_type} на основе анализа ключевых слов")
                return best_type
            
            # По умолчанию считаем команду поисковой
            logger.debug("Тип команды не определен, используем поиск по умолчанию")
            return 'search'
            
        except Exception as e:
            logger.error(f"Ошибка при анализе команды: {e}")
            return 'search'  # В случае ошибки используем безопасное значение
            
    def _is_relevant_context(self, context_before: str, context_after: str, cmd_type: str) -> bool:
        """Проверяет релевантность контекста для типа команды."""
        try:
            # Словарь контекстных индикаторов для каждого типа
            context_indicators = {
                'search': [
                    'информацию', 'данные', 'результаты', 'статьи',
                    'документы', 'материалы'
                ],
                'browser': [
                    'сайт', 'страницу', 'ссылку', 'веб-сайт',
                    'портал', 'адрес'
                ],
                'visual': [
                    'изображение', 'картинку', 'фото', 'скриншот',
                    'текст на', 'в изображении'
                ],
                'system': [
                    'файл', 'папку', 'директорию', 'скрипт',
                    'программу', 'процесс'
                ]
            }
            
            # Проверяем наличие индикаторов в контексте
            indicators = context_indicators.get(cmd_type, [])
            context = f"{context_before} {context_after}".lower()
            
            return any(indicator in context for indicator in indicators)
            
        except Exception as e:
            logger.error(f"Ошибка при анализе контекста: {e}")
            return False
    
    def _validate_result(self, result: str) -> Dict[str, Any]:
        """Валидирует результат выполнения.
        
        Args:
            result: Результат для валидации
            
        Returns:
            Результаты валидации
        """
        try:
            prompt = f"""Проверьте результат выполнения команды:

Результат: {result}

Проверьте:
1. Корректность результата
2. Наличие ошибок
3. Полноту информации
4. Форматирование

Формат ответа:
score: <число от 0 до 100>
issues: [<список проблем>]
suggestions: [<список предложений>]
is_valid: <true/false>
"""
            response = self.client.generate(prompt, self.model_name)
            return self._parse_validation_response(response)
        except Exception as e:
            logger.error(f"Ошибка при валидации результата: {e}")
            return {
                "score": 0,
                "issues": [f"Ошибка валидации: {str(e)}"],
                "suggestions": [],
                "is_valid": False
            }
    
    def _improve_result(self, result: str, issues: List[str]) -> Optional[str]:
        """Улучшает результат выполнения.
        
        Args:
            result: Исходный результат
            issues: Список проблем
            
        Returns:
            Улучшенный результат или None в случае ошибки
        """
        try:
            prompt = f"""Улучшите результат выполнения команды:

Исходный результат: {result}

Проблемы:
{chr(10).join(f"- {issue}" for issue in issues)}

Улучшите:
1. Исправьте найденные проблемы
2. Добавьте недостающую информацию
3. Улучшите форматирование
4. Сделайте результат более понятным

Формат ответа:
improved_result: <улучшенный результат>
"""
            response = self.client.generate(prompt, self.model_name)
            
            # Извлекаем улучшенный результат
            match = re.search(r"improved_result:\s*(.*)", response, re.DOTALL)
            if match:
                return match.group(1).strip()
            return None
            
        except Exception as e:
            logger.error(f"Ошибка при улучшении результата: {e}")
            return None
    
    def _check_result_relevance(self, result: str) -> Dict[str, Any]:
        """Проверяет релевантность результата.
        
        Args:
            result: Результат для проверки
            
        Returns:
            Результаты проверки
        """
        try:
            prompt = f"""Проверьте релевантность результата:

Результат: {result}

Проверьте:
1. Соответствие запросу
2. Актуальность информации
3. Полезность результата
4. Возможные улучшения

Формат ответа:
is_relevant: <true/false>
issues: [<список проблем>]
suggestions: [<список предложений>]
"""
            response = self.client.generate(prompt, self.model_name)
            
            # Парсим ответ
            is_relevant = "is_relevant: true" in response.lower()
            
            issues_match = re.findall(r"issues:\s*\[(.*?)\]", response)
            issues = [issue.strip() for issue in issues_match[0].split(",")] if issues_match else []
            
            suggestions_match = re.findall(r"suggestions:\s*\[(.*?)\]", response)
            suggestions = [suggestion.strip() for suggestion in suggestions_match[0].split(",")] if suggestions_match else []
            
            return {
                "is_relevant": is_relevant,
                "issues": issues,
                "suggestions": suggestions
            }
            
        except Exception as e:
            logger.error(f"Ошибка при проверке релевантности: {e}")
            return {
                "is_relevant": False,
                "issues": [f"Ошибка проверки: {str(e)}"],
                "suggestions": []
            }
    
    def _stream_result(self, result: str) -> Generator[str, None, None]:
        """Создает потоковый вывод результата.
        
        Args:
            result: Результат для потокового вывода
            
        Yields:
            Части результата
        """
        try:
            # Разбиваем результат на части
            parts = result.split("\n")
            
            # Выводим каждую часть
            for part in parts:
                if part.strip():
                    yield part + "\n"
                    time.sleep(0.1)  # Небольшая задержка для читаемости
                    
        except Exception as e:
            logger.error(f"Ошибка при потоковом выводе: {e}")
            yield f"Ошибка: {str(e)}"

class CriticAgent(BaseAgent):
    """Агент для критического анализа результатов.
    
    Этот класс отвечает за:
    - Анализ качества результатов
    - Анализ релевантности
    - Анализ производительности
    - Генерацию критики
    - Проверку конструктивности
    """
    
    def __init__(self, name: str, system_prompt: str, model_name: str, client: OllamaClient):
        """Инициализирует агента-критика.
        
        Args:
            name: Имя агента
            system_prompt: Системный промпт
            model_name: Имя модели LLM
            client: Клиент Ollama
            
        Raises:
            ValueError: Если параметры некорректны
        """
        super().__init__(name, system_prompt, model_name, client)
    
    def criticize(
        self,
        executor_result: str,
        stream: bool = False,
        **kwargs
    ) -> Union[str, Generator[str, None, None]]:
        """Выполняет критический анализ результата."""
        if not executor_result or not isinstance(executor_result, str):
            raise ValueError("Результат должен быть непустой строкой")
            
        try:
            # Обновляем состояние
            self.update_state(AgentStatus.CRITICIZING, "Критический анализ")
            
            # Анализируем качество
            quality_analysis = self._analyze_quality(executor_result, **kwargs)
            if not quality_analysis:
                self.update_state(AgentStatus.ERROR, "Ошибка анализа качества")
                return "Ошибка: не удалось проанализировать качество"
            
            # Анализируем релевантность
            relevance_analysis = self._analyze_relevance(executor_result, **kwargs)
            if not relevance_analysis:
                self.update_state(AgentStatus.ERROR, "Ошибка анализа релевантности")
                return "Ошибка: не удалось проанализировать релевантность"
            
            # Анализируем производительность
            performance_analysis = self._analyze_performance(executor_result, **kwargs)
            if not performance_analysis:
                self.update_state(AgentStatus.ERROR, "Ошибка анализа производительности")
                return "Ошибка: не удалось проанализировать производительность"
            
            # Генерируем критику
            critique = self._build_critique(
                quality_analysis,
                relevance_analysis,
                performance_analysis,
                **kwargs
            )
            if not critique:
                self.update_state(AgentStatus.ERROR, "Ошибка генерации критики")
                return "Ошибка: не удалось сгенерировать критику"
            
            # Проверяем конструктивность
            if not self._is_constructive(critique):
                critique = self._make_constructive(critique, **kwargs)
            
            # Обновляем состояние
            self.update_state(AgentStatus.COMPLETED, "Критический анализ завершен")
            
            # Возвращаем результат
            if stream:
                return self._stream_result(critique)
            return critique
            
        except Exception as e:
            error_msg = str(e) if not isinstance(e, AgentStatus) else e.value
            self.update_state(AgentStatus.ERROR, f"Ошибка анализа: {error_msg}")
            logger.error(f"Ошибка при критическом анализе: {error_msg}")
            return f"Ошибка анализа: {error_msg}"
            
    def _analyze_quality(self, result: str, **kwargs) -> Dict[str, Any]:
        """Анализирует качество результата."""
        try:
            prompt = self._build_quality_analysis_prompt(result)
            response = self.client.generate(prompt=prompt, model=self.model_name, **kwargs)
            return self._parse_analysis_response(response)
        except Exception as e:
            logger.error(f"Ошибка при анализе качества: {e}")
            return {}
            
    def _analyze_relevance(self, result: str, **kwargs) -> Dict[str, Any]:
        """Анализирует релевантность результата."""
        try:
            prompt = self._build_relevance_analysis_prompt(result)
            response = self.client.generate(prompt=prompt, model=self.model_name, **kwargs)
            return self._parse_analysis_response(response)
        except Exception as e:
            logger.error(f"Ошибка при анализе релевантности: {e}")
            return {}
            
    def _analyze_performance(self, result: str, **kwargs) -> Dict[str, Any]:
        """Анализирует производительность результата."""
        try:
            prompt = self._build_performance_analysis_prompt(result)
            response = self.client.generate(prompt=prompt, model=self.model_name, **kwargs)
            return self._parse_analysis_response(response)
        except Exception as e:
            logger.error(f"Ошибка при анализе производительности: {e}")
            return {}
            
    def _build_critique(
        self,
        quality: Dict[str, Any],
        relevance: Dict[str, Any],
        performance: Dict[str, Any],
        **kwargs
    ) -> Optional[str]:
        """Создает критический анализ на основе результатов."""
        try:
            prompt = f"""Создайте критический анализ на основе результатов:

Качество:
- Оценка: {quality.get('score', 0)}
- Сильные стороны: {', '.join(quality.get('strengths', []))}
- Слабые стороны: {', '.join(quality.get('weaknesses', []))}
- Предложения: {', '.join(quality.get('suggestions', []))}

Релевантность:
- Оценка: {relevance.get('score', 0)}
- Сильные стороны: {', '.join(relevance.get('strengths', []))}
- Слабые стороны: {', '.join(relevance.get('weaknesses', []))}
- Предложения: {', '.join(relevance.get('suggestions', []))}

Производительность:
- Оценка: {performance.get('score', 0)}
- Сильные стороны: {', '.join(performance.get('strengths', []))}
- Слабые стороны: {', '.join(performance.get('weaknesses', []))}
- Предложения: {', '.join(performance.get('suggestions', []))}

Создайте конструктивный критический анализ, который:
1. Отмечает сильные стороны
2. Указывает на слабые стороны
3. Предлагает конкретные улучшения
4. Сохраняет баланс между критикой и поддержкой
"""
            response = self.client.generate(prompt=prompt, model=self.model_name, **kwargs)
            return response.strip()
        except Exception as e:
            logger.error(f"Ошибка при создании критики: {e}")
            return None
            
    def _is_constructive(self, critique: str, **kwargs) -> bool:
        """Проверяет конструктивность критики."""
        try:
            prompt = f"""Проверьте конструктивность критики:

Критика: {critique}

Проверьте:
1. Наличие конкретных предложений
2. Баланс между критикой и поддержкой
3. Профессиональный тон
4. Полезность для улучшения

Формат ответа:
is_constructive: <true/false>
reason: <причина>
"""
            response = self.client.generate(prompt=prompt, model=self.model_name, **kwargs)
            return "is_constructive: true" in response.lower()
        except Exception as e:
            logger.error(f"Ошибка при проверке конструктивности: {e}")
            return False
            
    def _make_constructive(self, critique: str, **kwargs) -> str:
        """Улучшает конструктивность критики."""
        try:
            prompt = f"""Улучшите конструктивность критики:

Исходная критика: {critique}

Улучшите:
1. Добавьте конкретные предложения
2. Усильте баланс между критикой и поддержкой
3. Сделайте тон более профессиональным
4. Усильте полезность для улучшения

Формат ответа:
improved_critique: <улучшенная критика>
"""
            response = self.client.generate(prompt=prompt, model=self.model_name, **kwargs)
            
            # Извлекаем улучшенную критику
            match = re.search(r"improved_critique:\s*(.*)", response, re.DOTALL)
            if match:
                return match.group(1).strip()
            return critique
            
        except Exception as e:
            logger.error(f"Ошибка при улучшении конструктивности: {e}")
            return critique
    
    def _stream_result(self, result: str) -> Generator[str, None, None]:
        """Создает потоковый вывод результата.
        
        Args:
            result: Результат для потокового вывода
            
        Yields:
            Части результата
        """
        try:
            # Разбиваем результат на части
            parts = result.split("\n")
            
            # Выводим каждую часть
            for part in parts:
                if part.strip():
                    yield part + "\n"
                    time.sleep(0.1)  # Небольшая задержка для читаемости
                    
        except Exception as e:
            logger.error(f"Ошибка при потоковом выводе: {e}")
            yield f"Ошибка: {str(e)}"

    def _build_quality_analysis_prompt(self, result: str) -> str:
        """Создает промпт для анализа качества."""
        return f"""Проанализируйте качество результата:

Результат: {result}

Проанализируйте:
1. Точность и достоверность
2. Полноту информации
3. Логичность и последовательность
4. Ясность и понятность
5. Профессиональность

Формат ответа:
score: <число от 0 до 100>
strengths: [<список сильных сторон>]
weaknesses: [<список слабых сторон>]
suggestions: [<список предложений>]
"""

    def _build_relevance_analysis_prompt(self, result: str) -> str:
        """Создает промпт для анализа релевантности."""
        return f"""Проанализируйте релевантность результата:

Результат: {result}

Проанализируйте:
1. Соответствие запросу
2. Актуальность информации
3. Полезность для пользователя
4. Своевременность
5. Контекстность

Формат ответа:
score: <число от 0 до 100>
strengths: [<список сильных сторон>]
weaknesses: [<список слабых сторон>]
suggestions: [<список предложений>]
"""

    def _build_performance_analysis_prompt(self, result: str) -> str:
        """Создает промпт для анализа производительности."""
        return f"""Проанализируйте производительность результата:

Результат: {result}

Проанализируйте:
1. Эффективность
2. Оптимальность
3. Скорость выполнения
4. Использование ресурсов
5. Масштабируемость

Формат ответа:
score: <число от 0 до 100>
strengths: [<список сильных сторон>]
weaknesses: [<список слабых сторон>]
suggestions: [<список предложений>]
"""

class PraiseAgent(BaseAgent):
    """Агент для анализа сильных сторон результатов.
    
    Этот класс отвечает за:
    - Анализ качества результатов
    - Анализ релевантности
    - Анализ производительности
    - Генерацию похвалы
    - Проверку объективности
    """
    
    def __init__(self, name: str, system_prompt: str, model_name: str, client: OllamaClient):
        """Инициализирует агента-похвалы.
        
        Args:
            name: Имя агента
            system_prompt: Системный промпт
            model_name: Имя модели LLM
            client: Клиент Ollama
            
        Raises:
            ValueError: Если параметры некорректны
        """
        super().__init__(name, system_prompt, model_name, client)
    
    def praise(
        self,
        executor_result: str,
        stream: bool = False,
        **kwargs
    ) -> Union[str, Generator[str, None, None]]:
        """Выполняет анализ сильных сторон результата."""
        if not executor_result or not isinstance(executor_result, str):
            raise ValueError("Результат должен быть непустой строкой")
            
        try:
            # Обновляем состояние
            self.update_state(AgentStatus.PRAISING, "Анализ сильных сторон")
            
            # Анализируем качество
            quality_analysis = self._analyze_quality(executor_result, **kwargs)
            if not quality_analysis:
                self.update_state(AgentStatus.ERROR, "Ошибка анализа качества")
                return "Ошибка: не удалось проанализировать качество"
            
            # Анализируем релевантность
            relevance_analysis = self._analyze_relevance(executor_result, **kwargs)
            if not relevance_analysis:
                self.update_state(AgentStatus.ERROR, "Ошибка анализа релевантности")
                return "Ошибка: не удалось проанализировать релевантность"
            
            # Анализируем производительность
            performance_analysis = self._analyze_performance(executor_result, **kwargs)
            if not performance_analysis:
                self.update_state(AgentStatus.ERROR, "Ошибка анализа производительности")
                return "Ошибка: не удалось проанализировать производительность"
            
            # Генерируем похвалу
            praise = self._build_praise(
                quality_analysis,
                relevance_analysis,
                performance_analysis,
                **kwargs
            )
            if not praise:
                self.update_state(AgentStatus.ERROR, "Ошибка генерации похвалы")
                return "Ошибка: не удалось сгенерировать похвалу"
            
            # Проверяем объективность
            if not self._is_objective(praise, **kwargs):
                praise = self._make_objective(praise, **kwargs)
            
            # Обновляем состояние
            self.update_state(AgentStatus.COMPLETED, "Анализ сильных сторон завершен")
            
            # Возвращаем результат
            if stream:
                return self._stream_result(praise)
            return praise
            
        except Exception as e:
            error_msg = str(e) if not isinstance(e, AgentStatus) else e.value
            self.update_state(AgentStatus.ERROR, f"Ошибка анализа: {error_msg}")
            logger.error(f"Ошибка при анализе сильных сторон: {error_msg}")
            return f"Ошибка анализа: {error_msg}"
    
    def _analyze_quality(self, result: str, **kwargs) -> Dict[str, Any]:
        """Анализирует качество результата."""
        try:
            prompt = self._build_quality_analysis_prompt(result)
            response = self.client.generate(prompt=prompt, model=self.model_name, **kwargs)
            return self._parse_analysis_response(response)
        except Exception as e:
            logger.error(f"Ошибка при анализе качества: {e}")
            return {}

    def _build_quality_analysis_prompt(self, result: str) -> str:
        """Создает промпт для анализа качества."""
        return f"""Проанализируйте сильные стороны результата:

Результат: {result}

Проанализируйте:
1. Точность и достоверность
2. Полноту информации
3. Логичность и последовательность
4. Ясность и понятность
5. Профессиональность

Формат ответа:
score: <число от 0 до 100>
strengths: [<список сильных сторон>]
achievements: [<список достижений>]
innovations: [<список инноваций>]
benefits: [<список преимуществ>]
"""

    def _analyze_relevance(self, result: str, **kwargs) -> Dict[str, Any]:
        """Анализирует релевантность результата."""
        try:
            prompt = self._build_relevance_analysis_prompt(result)
            response = self.client.generate(prompt=prompt, model=self.model_name, **kwargs)
            return self._parse_analysis_response(response)
        except Exception as e:
            logger.error(f"Ошибка при анализе релевантности: {e}")
            return {}
    
    def _build_relevance_analysis_prompt(self, result: str) -> str:
        """Создает промпт для анализа релевантности."""
        return f"""Проанализируйте релевантность результата:

Результат: {result}

Проанализируйте:
1. Соответствие запросу
2. Актуальность информации
3. Полезность для пользователя
4. Своевременность
5. Контекстность

Формат ответа:
score: <число от 0 до 100>
strengths: [<список сильных сторон>]
achievements: [<список достижений>]
innovations: [<список инноваций>]
benefits: [<список преимуществ>]
"""
    
    def _analyze_performance(self, result: str, **kwargs) -> Dict[str, Any]:
        """Анализирует производительность результата."""
        try:
            prompt = self._build_performance_analysis_prompt(result)
            response = self.client.generate(prompt=prompt, model=self.model_name, **kwargs)
            return self._parse_analysis_response(response)
        except Exception as e:
            logger.error(f"Ошибка при анализе производительности: {e}")
            return {}
    
    def _build_performance_analysis_prompt(self, result: str) -> str:
        """Создает промпт для анализа производительности."""
        return f"""Проанализируйте эффективность результата:

Результат: {result}

Проанализируйте:
1. Эффективность
2. Оптимальность
3. Скорость выполнения
4. Использование ресурсов
5. Масштабируемость

Формат ответа:
score: <число от 0 до 100>
strengths: [<список сильных сторон>]
achievements: [<список достижений>]
innovations: [<список инноваций>]
benefits: [<список преимуществ>]
"""
    
    def _build_praise(
        self,
        quality: Dict[str, Any],
        relevance: Dict[str, Any],
        performance: Dict[str, Any],
        **kwargs
    ) -> Optional[str]:
        """Создает похвальный анализ на основе результатов.
        
        Args:
            quality: Результаты анализа качества
            relevance: Результаты анализа релевантности
            performance: Результаты анализа производительности
            **kwargs: Дополнительные параметры для генерации
            
        Returns:
            Похвальный анализ или None в случае ошибки
        """
        try:
            prompt = f"""Создайте похвальный анализ на основе результатов:

Качество:
- Оценка: {quality.get('score', 0)}
- Сильные стороны: {', '.join(quality.get('strengths', []))}
- Достижения: {', '.join(quality.get('achievements', []))}
- Инновации: {', '.join(quality.get('innovations', []))}
- Преимущества: {', '.join(quality.get('benefits', []))}

Релевантность:
- Оценка: {relevance.get('score', 0)}
- Сильные стороны: {', '.join(relevance.get('strengths', []))}
- Достижения: {', '.join(relevance.get('achievements', []))}
- Инновации: {', '.join(relevance.get('innovations', []))}
- Преимущества: {', '.join(relevance.get('benefits', []))}

Производительность:
- Оценка: {performance.get('score', 0)}
- Сильные стороны: {', '.join(performance.get('strengths', []))}
- Достижения: {', '.join(performance.get('achievements', []))}
- Инновации: {', '.join(performance.get('innovations', []))}
- Преимущества: {', '.join(performance.get('benefits', []))}

Создайте объективный похвальный анализ, который:
1. Отмечает конкретные достижения
2. Выделяет инновационные решения
3. Подчеркивает преимущества
4. Сохраняет баланс и объективность
"""
            # Извлекаем параметры генерации из kwargs
            generation_params = {
                'temperature': kwargs.get('temperature', 0.7),
                'top_p': kwargs.get('top_p', 0.9),
                'presence_penalty': kwargs.get('presence_penalty', 0.1),
                'frequency_penalty': kwargs.get('frequency_penalty', 0.1),
                'max_tokens': kwargs.get('max_tokens', 1000)
            }

            response = self.client.generate(
                prompt=prompt,
                model=self.model_name,
                stream=False,
                **generation_params
            )
            return response.strip()
        except Exception as e:
            logger.error(f"Ошибка при создании похвалы: {e}")
            return None
    
    def _is_objective(self, praise: str, **kwargs) -> bool:
        """Проверяет объективность похвалы.
        
        Args:
            praise: Похвала для проверки
            
        Returns:
            True если похвала объективна
        """
        try:
            prompt = f"""Проверьте объективность похвалы:

Похвала: {praise}

Проверьте:
1. Наличие конкретных фактов
2. Баланс между похвалой и критикой
3. Профессиональный тон
4. Полезность для улучшения

Формат ответа:
is_objective: <true/false>
reason: <причина>
"""
            response = self.client.generate(prompt, self.model_name, **kwargs)
            return "is_objective: true" in response.lower()
        except Exception as e:
            logger.error(f"Ошибка при проверке объективности: {e}")
            return False
    
    def _make_objective(self, praise: str, **kwargs) -> str:
        """Улучшает объективность похвалы.
        
        Args:
            praise: Исходная похвала
            
        Returns:
            Улучшенная похвала
        """
        try:
            prompt = f"""Улучшите объективность похвалы:

Исходная похвала: {praise}

Улучшите:
1. Добавьте конкретные факты
2. Усильте баланс между похвалой и критикой
3. Сделайте тон более профессиональным
4. Усильте полезность для улучшения

Формат ответа:
improved_praise: <улучшенная похвала>
"""
            response = self.client.generate(prompt, self.model_name, **kwargs)
            
            # Извлекаем улучшенную похвалу
            match = re.search(r"improved_praise:\s*(.*)", response, re.DOTALL)
            if match:
                return match.group(1).strip()
            return praise
            
        except Exception as e:
            logger.error(f"Ошибка при улучшении объективности: {e}")
            return praise
    
    def _stream_result(self, result: str) -> Generator[str, None, None]:
        """Создает потоковый вывод результата.
        
        Args:
            result: Результат для потокового вывода
            
        Yields:
            Части результата
        """
        try:
            # Разбиваем результат на части
            parts = result.split("\n")
            
            # Выводим каждую часть
            for part in parts:
                if part.strip():
                    yield part + "\n"
                    time.sleep(0.1)  # Небольшая задержка для читаемости
                    
        except Exception as e:
            logger.error(f"Ошибка при потоковом выводе: {e}")
            yield f"Ошибка: {str(e)}"

class ArbiterAgent(BaseAgent):
    """Агент для синтеза обратной связи и генерации инструкций по улучшению.
    
    Этот агент анализирует критические замечания и похвалу от других агентов,
    синтезирует их в единую обратную связь и генерирует конкретные инструкции
    по улучшению результатов.
    """
    
    def __init__(self, name: str, system_prompt: str, model_name: str, client: OllamaClient):
        """Инициализирует ArbiterAgent.
        
        Args:
            name: Имя агента
            system_prompt: Системный промпт
            model_name: Имя модели LLM
            client: Клиент Ollama
        """
        super().__init__(name, system_prompt or ARBITER_PROMPT, model_name, client)

    def produce_rework_instruction(
        self,
        exec_text: str,
        cr_text: str,
        pr_text: str,
        stream: bool = False,
        **ollama_opts
    ) -> Union[str, Generator[str, None, None]]:
        """Выполняет синтез обратной связи и формирует инструкции по улучшению.
        
        Args:
            exec_text: Результат выполнения от ExecutorAgent
            cr_text: Критический анализ от CriticAgent
            pr_text: Похвальный анализ от PraiseAgent
            stream: Режим потоковой передачи
            **ollama_opts: Дополнительные параметры для Ollama
            
        Returns:
            Сгенерированные инструкции или генератор чанков
        """
        if not all([exec_text, cr_text, pr_text]):
            raise ValueError("Все входные тексты должны быть непустыми")
            
        self.update_state(AgentStatus.ARBITRATING, "Синтез обратной связи", total_steps=4)
        try:
            # Шаг 1: Анализ критики
            self.update_progress(0.25, AgentStatus.ARBITRATING)
            critique_analysis = self._analyze_critique(cr_text)
            if not critique_analysis:
                raise ValueError("Не удалось проанализировать критику")
            self.increment_steps()
            
            # Шаг 2: Анализ похвалы
            self.update_progress(0.5, AgentStatus.ARBITRATING)
            praise_analysis = self._analyze_praise(pr_text)
            if not praise_analysis:
                raise ValueError("Не удалось проанализировать похвалу")
            self.increment_steps()
            
            # Шаг 3: Синтез обратной связи
            self.update_progress(0.75, AgentStatus.ARBITRATING)
            feedback = self._synthesize_feedback(critique_analysis, praise_analysis)
            if not feedback:
                raise ValueError("Не удалось синтезировать обратную связь")
            self.increment_steps()
            
            # Шаг 4: Формирование инструкций
            self.update_progress(1.0, AgentStatus.ARBITRATING)
            instructions = self._generate_instructions(feedback)
            if not instructions:
                raise ValueError("Не удалось сгенерировать инструкции")
                
            if not self._validate_instructions(instructions):
                instructions = self._improve_instructions(instructions)
            self.increment_steps()
            
            self.complete_state()
            
            if not stream:
                return instructions
            else:
                def gen_instructions():
                    yield instructions
                return gen_instructions()
                
        except Exception as e:
            self.state.set_error(str(e))
            raise

    def _analyze_critique(self, critique: str) -> Dict[str, Any]:
        """Анализирует критический отзыв.
        
        Args:
            critique: Текст критического отзыва
            
        Returns:
            Результаты анализа в виде словаря
        """
        if not critique:
            raise ValueError("Критический отзыв не может быть пустым")
            
        analysis_prompt = f"""
        Проанализируй следующий критический отзыв:
        
        {critique}
        
        Проверь:
        1. Обоснованность критики
        2. Конструктивность предложений
        3. Приоритетность проблем
        4. Реалистичность улучшений
        5. Потенциальное влияние
        
        Ответь в формате:
        score: 0-100
        strengths: [список сильных сторон]
        weaknesses: [список слабых сторон]
        suggestions: [список предложений по улучшению]
        severity: [уровень серьезности проблем]
        """
        
        try:
            response = self.client.generate(
                prompt=analysis_prompt,
                model=self.model_name,
                stream=False
            )
            return self._parse_analysis_response(response)
        except Exception as e:
            logger.error(f"Ошибка при анализе критики: {e}")
            return {
                "score": 0,
                "strengths": [],
                "weaknesses": [],
                "suggestions": [],
                "severity": "low"
            }

    def _analyze_praise(self, praise: str) -> Dict[str, Any]:
        """Анализирует похвальный отзыв.
        
        Args:
            praise: Текст похвального отзыва
            
        Returns:
            Результаты анализа в виде словаря
        """
        if not praise:
            raise ValueError("Похвальный отзыв не может быть пустым")
            
        analysis_prompt = f"""
        Проанализируй следующий похвальный отзыв:
        
        {praise}
        
        Проверь:
        1. Обоснованность похвалы
        2. Значимость достижений
        3. Реалистичность оценок
        4. Потенциал развития
        5. Практическая ценность
        
        Ответь в формате:
        score: 0-100
        strengths: [список сильных сторон]
        weaknesses: [список слабых сторон]
        suggestions: [список предложений по улучшению]
        impact: [уровень влияния]
        """
        
        try:
            response = self.client.generate(
                prompt=analysis_prompt,
                model=self.model_name,
                stream=False
            )
            return self._parse_analysis_response(response)
        except Exception as e:
            logger.error(f"Ошибка при анализе похвалы: {e}")
            return {
                "score": 0,
                "strengths": [],
                "weaknesses": [],
                "suggestions": [],
                "impact": "low"
            }

    def _synthesize_feedback(self, critique: Dict, praise: Dict) -> Dict[str, Any]:
        """Синтезирует обратную связь.
        
        Args:
            critique: Результаты анализа критики
            praise: Результаты анализа похвалы
            
        Returns:
            Синтезированная обратная связь
        """
        synthesis_prompt = f"""
        Синтезируй обратную связь на основе следующих данных:
        
        Критика:
        - Оценка: {critique.get('score', 0)}
        - Сильные стороны: {', '.join(critique.get('strengths', []))}
        - Слабые стороны: {', '.join(critique.get('weaknesses', []))}
        - Предложения: {', '.join(critique.get('suggestions', []))}
        - Серьезность: {critique.get('severity', 'low')}
        
        Похвала:
        - Оценка: {praise.get('score', 0)}
        - Сильные стороны: {', '.join(praise.get('strengths', []))}
        - Слабые стороны: {', '.join(praise.get('weaknesses', []))}
        - Предложения: {', '.join(praise.get('suggestions', []))}
        - Влияние: {praise.get('impact', 'low')}
        
        Сформируй сбалансированный анализ с приоритетами улучшения.
        """
        
        try:
            response = self.client.generate(
                prompt=synthesis_prompt,
                model=self.model_name,
                stream=False
            )
            return self._parse_analysis_response(response)
        except Exception as e:
            logger.error(f"Ошибка при синтезе обратной связи: {e}")
            return {
                "score": 0,
                "strengths": [],
                "weaknesses": [],
                "suggestions": [],
                "priority": "low"
            }

    def _generate_instructions(self, feedback: Dict) -> str:
        """Генерирует инструкции по улучшению.
        
        Args:
            feedback: Синтезированная обратная связь
            
        Returns:
            Сгенерированные инструкции
        """
        instructions_prompt = f"""
        Сформируй инструкции по улучшению на основе следующего анализа:
        
        {feedback}
        
        Требования к инструкциям:
        1. Четкость и конкретность
        2. Приоритизация задач
        3. Измеримые результаты
        4. Реалистичные сроки
        5. Конкретные шаги
        
        Предоставь структурированные инструкции с четкими критериями успеха.
        """
        
        try:
            return self.client.generate(
                prompt=instructions_prompt,
                model=self.model_name,
                stream=False
            )
        except Exception as e:
            logger.error(f"Ошибка при генерации инструкций: {e}")
            return "Не удалось сгенерировать инструкции"

    def _validate_instructions(self, instructions: str) -> bool:
        """Проверяет качество инструкций."""
        if not instructions:
            return False
            
        try:
            # Проверяем базовые требования
            if len(instructions.strip()) < 10:
                logger.warning("Инструкции слишком короткие")
                return False
            
            # Проверяем наличие конкретных шагов
            if not any(word in instructions.lower() for word in ['шаг', 'действие', 'выполнить', 'улучшить']):
                logger.warning("Инструкции не содержат конкретных шагов")
                return False
            
            # Проверяем структуру
            if not re.search(r'\d+\.|\-|\*', instructions):
                logger.warning("Инструкции не имеют структурированного формата")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Ошибка при валидации инструкций: {e}")
            return False

    def _improve_instructions(self, instructions: str) -> str:
        """Улучшает качество инструкций."""
        try:
            improvement_prompt = f"""
            Улучшите следующие инструкции:
            
            {instructions}
            
            Требования к улучшению:
            1. Добавьте конкретные шаги
            2. Сделайте инструкции более четкими
            3. Добавьте примеры где это необходимо
            4. Используйте структурированный формат
            5. Добавьте критерии проверки
            
            Предоставьте улучшенную версию инструкций.
            """
            
            improved = self.client.generate(
                prompt=improvement_prompt,
                model=self.model_name,
                stream=False
            )
            
            return improved if improved else instructions
            
        except Exception as e:
            logger.error(f"Ошибка при улучшении инструкций: {e}")
            return instructions

def init_vector_store():
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = SimpleVectorStore()
        # Добавляем ограничение на размер хранилища
        st.session_state.vector_store.max_documents = 1000
        st.session_state.vector_store.max_document_size = 1000000  # 1MB
