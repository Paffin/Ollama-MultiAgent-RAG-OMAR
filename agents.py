from typing import List, Dict, Union, Generator, Optional, Tuple
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
import re
import time
from dataclasses import dataclass
from enum import Enum

class AgentStatus(Enum):
    IDLE = "idle"
    PLANNING = "planning"
    EXECUTING = "executing"
    CRITICIZING = "criticizing"
    PRAISING = "praising"
    ARBITRATING = "arbitrating"
    COMPLETED = "completed"
    ERROR = "error"

@dataclass
class AgentState:
    status: AgentStatus
    current_task: str
    start_time: float
    end_time: Optional[float] = None
    error: Optional[str] = None
    metadata: Dict = None

class BaseAgent:
    def __init__(self, name: str, system_prompt: str, model_name: str, client: OllamaClient):
        self.name = name
        self.system_prompt = system_prompt
        self.model_name = model_name
        self.client = client
        self.history: List[Dict[str, str]] = []
        self.state: Optional[AgentState] = None
        self.context: Dict = {}
    
    def add_message(self, role: str, content: str) -> None:
        """Добавляет сообщение в историю диалога."""
        self.history.append({"role": role, "content": content})
    
    def build_prompt(self) -> str:
        """Строит полный промпт для LLM из системного промпта, контекста и истории."""
        prompt_parts = [
            f"System({self.name}): {self.system_prompt}\n"
        ]
        
        if self.context:
            prompt_parts.append("Context:")
            prompt_parts.extend(f"- {k}: {v}" for k, v in self.context.items())
            
        prompt_parts.extend(
            f"{msg['role'].upper()}: {msg['content']}\n"
            for msg in self.history
        )
        
        return "\n".join(prompt_parts)

    def update_state(self, status: AgentStatus, task: str, metadata: Dict = None) -> None:
        """Обновляет состояние агента."""
        if self._should_create_new_state():
            self._create_new_state(status, task, metadata)
        else:
            self._update_existing_state(status, task, metadata)

    def _should_create_new_state(self) -> bool:
        """Проверяет, нужно ли создавать новое состояние."""
        return self.state is None or self.state.status == AgentStatus.COMPLETED

    def _create_new_state(self, status: AgentStatus, task: str, metadata: Dict) -> None:
        """Создает новое состояние агента."""
        self.state = AgentState(
            status=status,
            current_task=task,
            start_time=time.time(),
            metadata=metadata or {}
        )

    def _update_existing_state(self, status: AgentStatus, task: str, metadata: Dict) -> None:
        """Обновляет существующее состояние агента."""
        self.state.status = status
        self.state.current_task = task
        if metadata:
            self.state.metadata.update(metadata)

    def complete_state(self, error: str = None) -> None:
        """Завершает текущее состояние агента."""
        if self.state:
            self.state.status = AgentStatus.ERROR if error else AgentStatus.COMPLETED
            self.state.end_time = time.time()
            if error:
                self.state.error = error

    def get_state_summary(self) -> str:
        """Возвращает текстовое описание текущего состояния."""
        if not self.state:
            return "No active state"
            
        duration = self._calculate_duration()
        return (
            f"Status: {self.state.status.value}, "
            f"Task: {self.state.current_task}, "
            f"Duration: {duration:.2f}s"
        )

    def _calculate_duration(self) -> float:
        """Вычисляет длительность выполнения текущей задачи."""
        end_time = self.state.end_time or time.time()
        return end_time - self.state.start_time

    def clear_history(self) -> None:
        """Очищает историю диалога и контекст агента."""
        self.history = []
        self.context = {}
        self.state = None

class PlannerAgent(BaseAgent):
    """Агент для планирования и анализа запросов пользователя."""
    
    def __init__(self, name: str, system_prompt: str, model_name: str, client: OllamaClient):
        super().__init__(name, system_prompt or PLANNER_PROMPT, model_name, client)

    def create_detailed_plan(self, user_query: str) -> str:
        """Создает детальный план действий на основе запроса пользователя."""
        self.update_state(AgentStatus.PLANNING, "Creating detailed plan")
        try:
            plan = self._generate_plan(user_query)
            self.complete_state()
            return plan.strip()
        except Exception as e:
            self.complete_state(error=str(e))
            raise

    def _generate_plan(self, user_query: str) -> str:
        """Генерирует план действий с помощью LLM."""
        prompt = self._build_plan_prompt(user_query)
        plan = self.client.generate(prompt=prompt, model=self.model_name, stream=False)
        return ''.join(plan) if not isinstance(plan, str) else plan

    def _build_plan_prompt(self, user_query: str) -> str:
        """Создает промпт для генерации плана."""
        return (
            f"Создай детальный план действий для следующего запроса. "
            f"План должен включать:\n"
            f"1. Основную цель запроса\n"
            f"2. Необходимые шаги для достижения цели\n"
            f"3. Требуемые инструменты (поиск, браузер, локальные данные)\n"
            f"4. Потенциальные риски или ограничения\n"
            f"5. Критерии успешного выполнения\n\n"
            f"Запрос: {user_query}\n\n"
            f"План:"
        )

    def analyze_query(self, user_query: str) -> str:
        """Анализирует запрос пользователя и определяет необходимые действия."""
        self.update_state(AgentStatus.PLANNING, "Analyzing query")
        try:
            analysis = self._generate_analysis(user_query)
            self.complete_state()
            return analysis.strip() or user_query
        except Exception as e:
            self.complete_state(error=str(e))
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
        """Генерирует инструкцию для выполнения запроса."""
        self.update_state(AgentStatus.PLANNING, "Generating instruction")
        try:
            self.add_message("user", user_query)
            
            analysis = self.analyze_query(user_query)
            self.add_message("assistant", f"Анализ запроса:\n{analysis}")
            
            detailed_plan = self.create_detailed_plan(user_query)
            self.add_message("assistant", f"Детальный план:\n{detailed_plan}")
            
            plan = self._determine_execution_plan(user_query, vector_store)
            final_instruction = self._build_final_instruction(analysis, detailed_plan, plan)
            
            self.add_message("assistant", final_instruction)
            
            if not stream:
                self.complete_state()
                return final_instruction
                
            def gen_plan():
                yield final_instruction
            self.complete_state()
            return gen_plan()
            
        except Exception as e:
            self.complete_state(error=str(e))
            raise

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
    """Агент для выполнения инструкций и команд."""
    
    def __init__(self, name: str, system_prompt: str, model_name: str, client: OllamaClient):
        super().__init__(name, system_prompt or EXECUTOR_PROMPT, model_name, client)
        self.browser = None

    def execute_instruction(
        self,
        instruction: str,
        vector_store: SimpleVectorStore,
        stream=False,
        **ollama_opts
    ) -> Union[str, Generator[str, None, None]]:
        """Выполняет инструкцию и возвращает результат."""
        self.update_state(AgentStatus.EXECUTING, "Executing instruction")
        try:
            self.add_message("user", instruction)
            instruction = self._extract_instruction(instruction)
            
            result = self._execute_command(instruction, vector_store, stream, **ollama_opts)
            self.complete_state()
            return result
            
        except Exception as e:
            self.complete_state(error=str(e))
            raise

    def _extract_instruction(self, instruction: str) -> str:
        """Извлекает чистую инструкцию из полного текста."""
        parts = instruction.split("\n\n")
        if len(parts) > 1:
            self.context["plan"] = parts[0].replace("План действий:", "").strip()
            return parts[1].replace("Инструкция для выполнения:", "").strip()
        return instruction

    def _execute_command(
        self,
        instruction: str,
        vector_store: SimpleVectorStore,
        stream: bool,
        **ollama_opts
    ) -> Union[str, Generator[str, None, None]]:
        """Выполняет команду в зависимости от типа инструкции."""
        instr_lower = instruction.lower()
        
        if "ducksearch:" in instruction:
            return self._execute_ducksearch(instruction, **ollama_opts)
        elif instr_lower.startswith("search:"):
            return self._execute_local_search(instruction, vector_store)
        elif instr_lower.startswith("cmd:"):
            return self._execute_system_command(instruction)
        elif instr_lower.startswith("ls:"):
            return self._execute_list_directory(instruction)
        elif instr_lower.startswith("browser:"):
            return self._execute_browser_actions(instruction)
        elif instr_lower.startswith("visual:"):
            return self._execute_visual_analysis(instruction)
        else:
            return self._generate_llm_response(stream, **ollama_opts)

    def _execute_ducksearch(self, instruction: str, **ollama_opts) -> str:
        """Выполняет поиск через DuckDuckGo."""
        self.update_state(AgentStatus.EXECUTING, "Performing DuckDuckGo search")
        search_query = instruction.split("ducksearch:")[1].strip()
        
        try:
            results = self._perform_ducksearch(search_query)
            aggregated_text = self._aggregate_search_results(results)
            summary = self._generate_search_summary(search_query, aggregated_text, **ollama_opts)
            
            return self._format_search_output(results, summary)
        except Exception as e:
            return f"Ошибка при веб-поиске: {e}"

    def _perform_ducksearch(self, query: str) -> List[Dict]:
        """Выполняет поиск через DuckDuckGo API."""
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            return list(ddgs.text(query, max_results=15))

    def _aggregate_search_results(self, results: List[Dict]) -> str:
        """Агрегирует результаты поиска."""
        aggregated_text = ""
        for result in results:
            if result.get('href'):
                aggregated_text += self._process_search_result(result)
        return aggregated_text

    def _process_search_result(self, result: Dict) -> str:
        """Обрабатывает отдельный результат поиска."""
        try:
            br = PlaywrightBrowser(headless=True)
            start_time = time.time()
            br.launch()
            br.goto(result['href'])
            load_time = time.time() - start_time
            
            br.page.wait_for_selector("body", timeout=5000)
            page_text = br.page.inner_text("body")
            ssl_ok = "https:" in br.page.evaluate("() => window.location.protocol")
            
            return (
                f"URL: {result['href']}\n"
                f"Title: {br.get_page_title()}\n"
                f"Load Time: {load_time:.2f} сек\n"
                f"SSL: {'Да' if ssl_ok else 'Нет'}\n"
                f"Content (первые 1000 символов):\n{page_text[:1000]}\n\n"
            )
        except Exception as e:
            return f"Ошибка при извлечении информации с {result['href']}: {e}\n\n"
        finally:
            br.close()

    def _generate_search_summary(self, query: str, text: str, **ollama_opts) -> str:
        """Генерирует сводку результатов поиска."""
        if not text.strip():
            return ""
            
        summary_prompt = (
            f"Сделай сводку по запросу: '{query}'.\n"
            f"Исходный текст:\n\n{text}\n\nСводка:"
        )
        summary = self.client.generate(
            prompt=summary_prompt,
            model=self.model_name,
            stream=False,
            **ollama_opts
        )
        return ''.join(summary) if not isinstance(summary, str) else summary

    def _format_search_output(self, results: List[Dict], summary: str) -> str:
        """Форматирует вывод результатов поиска."""
        output = "Результаты поиска в DuckDuckGo:\n"
        
        if not results:
            output += "Результаты не найдены."
        else:
            for i, r in enumerate(results, start=1):
                output += (
                    f"{i}) {r.get('title', '')}\n"
                    f"   URL: {r.get('href', '')}\n"
                    f"   {r.get('body', '')}\n\n"
                )
        
        if summary:
            output += "\nСводка полученной информации:\n" + summary
        elif not results:
            output += "\nНе удалось извлечь информацию с найденных ссылок."
            
        self.add_message("assistant", output)
        return output

    def _execute_local_search(self, instruction: str, vector_store: SimpleVectorStore) -> str:
        """Выполняет локальный поиск через RAG."""
        self.update_state(AgentStatus.EXECUTING, "Performing local search")
        query = instruction.split("search:")[1].strip()
        found_docs = vector_store.search(query, k=3)
        
        output = f"[RAG] Найдено документов: {len(found_docs)}\n"
        for i, d in enumerate(found_docs, 1):
            output += f"{i}. {d[:200]}...\n"
            
        self.add_message("assistant", output)
        return output

    def _execute_system_command(self, instruction: str) -> str:
        """Выполняет системную команду."""
        self.update_state(AgentStatus.EXECUTING, "Executing system command")
        cmd_text = instruction.split("cmd:")[1].strip()
        output = run_system_command(cmd_text)
        self.add_message("assistant", output)
        return output

    def _execute_list_directory(self, instruction: str) -> str:
        """Выполняет команду просмотра директории."""
        self.update_state(AgentStatus.EXECUTING, "Listing directory")
        path = instruction.split("ls:")[1].strip()
        output = list_directory(path)
        self.add_message("assistant", output)
        return output

    def _execute_browser_actions(self, instruction: str) -> str:
        """Выполняет действия в браузере."""
        self.update_state(AgentStatus.EXECUTING, "Performing browser actions")
        actions = instruction.split("browser:")[1].split(";")
        br = PlaywrightBrowser(headless=True)
        
        try:
            br.launch()
            output = self._process_browser_actions(br, actions)
            self.add_message("assistant", output)
            return output
        finally:
            br.close()

    def _process_browser_actions(self, browser: PlaywrightBrowser, actions: List[str]) -> str:
        """Обрабатывает список действий в браузере."""
        output = []
        for act in actions:
            act = act.strip()
            if act.startswith("open url="):
                url = act.replace("open url=", "").strip()
                browser.goto(url)
                output.append(f"Открыли URL: {url}")
            elif act.startswith("screenshot path="):
                sc_path = act.replace("screenshot path=", "").strip()
                path_taken = browser.screenshot(path=sc_path)
                output.append(f"Скриншот сохранён в {path_taken}")
            elif act.startswith("click selector="):
                sel = act.replace("click selector=", "").strip()
                browser.click(sel)
                output.append(f"Кликнули по селектору {sel}")
            elif act.startswith("type selector="):
                sel, text = self._parse_type_action(act)
                browser.type_text(sel, text)
                output.append(f"В селектор {sel} введён текст {text}")
            elif act.startswith("parse selector="):
                sel = act.replace("parse selector=", "").strip()
                try:
                    parsed_text = browser.page.inner_text(sel)
                    output.append(f"Текст из {sel}: {parsed_text}")
                except Exception as e:
                    output.append(f"Ошибка при извлечении текста по селектору {sel}: {e}")
        return "\n".join(output)

    def _parse_type_action(self, action: str) -> Tuple[str, str]:
        """Парсит действие ввода текста."""
        parts = action.split(",")
        sel_part = parts[0].replace("type selector=", "").strip()
        text_part = parts[1].replace("text=", "").strip()
        return sel_part, text_part

    def _execute_visual_analysis(self, instruction: str) -> str:
        """Выполняет анализ визуального контента."""
        self.update_state(AgentStatus.EXECUTING, "Analyzing visual content")
        sub = instruction.split("visual:")[1].strip()
        
        if "||" in sub:
            img_path, prompt_text = sub.split("||", 1)
        else:
            img_path = sub
            prompt_text = "Describe the screenshot"
            
        output = llava_analyze_screenshot_via_ollama_llava(
            image_path=img_path.strip(),
            prompt=prompt_text.strip(),
            model="ollama:llava:13b"
        )
        self.add_message("assistant", output)
        return output

    def _generate_llm_response(
        self,
        stream: bool,
        **ollama_opts
    ) -> Union[str, Generator[str, None, None]]:
        """Генерирует ответ с помощью LLM."""
        self.update_state(AgentStatus.EXECUTING, "Generating LLM response")
        prompt = self.build_prompt()
        
        if not stream:
            resp = self.client.generate(
                prompt=prompt,
                model=self.model_name,
                stream=False,
                **ollama_opts
            )
            self.add_message("assistant", resp)
            return resp
            
        return self.client.generate(
            prompt=prompt,
            model=self.model_name,
            stream=True,
            **ollama_opts
        )

    def __del__(self):
        """Очистка ресурсов при удалении агента."""
        if self.browser:
            self.browser.close()

class CriticAgent(BaseAgent):
    def __init__(self, name: str, system_prompt: str, model_name: str, client: OllamaClient):
        super().__init__(name, system_prompt or CRITIC_PROMPT, model_name, client)

    def criticize(self, executor_result: str, stream=False, **ollama_opts) -> Union[str, Generator[str, None, None]]:
        self.update_state(AgentStatus.CRITICIZING, "Criticizing execution result")
        try:
            user_msg = f"Вот ответ Исполнителя. Определи ошибки, слабые стороны, неточности:\n\n{executor_result}"
            self.add_message("user", user_msg)
            prompt = self.build_prompt()
            if not stream:
                resp = self.client.generate(
                    prompt=prompt,
                    model=self.model_name,
                    stream=False,
                    **ollama_opts
                )
                self.add_message("assistant", resp)
                self.complete_state()
                return resp
            else:
                gen = self.client.generate(
                    prompt=prompt,
                    model=self.model_name,
                    stream=True,
                    **ollama_opts
                )
                self.complete_state()
                return gen
        except Exception as e:
            self.complete_state(error=str(e))
            raise

class PraiseAgent(BaseAgent):
    def __init__(self, name: str, system_prompt: str, model_name: str, client: OllamaClient):
        super().__init__(name, system_prompt or PRAISE_PROMPT, model_name, client)

    def praise(self, executor_result: str, stream=False, **ollama_opts) -> Union[str, Generator[str, None, None]]:
        self.update_state(AgentStatus.PRAISING, "Praising execution result")
        try:
            user_msg = f"Вот ответ Исполнителя. Покажи, что в нём хорошего, какие сильные стороны:\n\n{executor_result}"
            self.add_message("user", user_msg)
            prompt = self.build_prompt()
            if not stream:
                resp = self.client.generate(
                    prompt=prompt,
                    model=self.model_name,
                    stream=False,
                    **ollama_opts
                )
                self.add_message("assistant", resp)
                self.complete_state()
                return resp
            else:
                gen = self.client.generate(
                    prompt=prompt,
                    model=self.model_name,
                    stream=True,
                    **ollama_opts
                )
                self.complete_state()
                return gen
        except Exception as e:
            self.complete_state(error=str(e))
            raise

class ArbiterAgent(BaseAgent):
    def __init__(self, name: str, system_prompt: str, model_name: str, client: OllamaClient):
        super().__init__(name, system_prompt or ARBITER_PROMPT, model_name, client)

    def produce_rework_instruction(
        self,
        executor_result: str,
        critic_text: str,
        praise_text: str,
        stream=False,
        **ollama_opts
    ) -> Union[str, Generator[str, None, None]]:
        self.update_state(AgentStatus.ARBITRATING, "Producing rework instruction")
        try:
            user_msg = (
                "У нас есть:\n"
                f"1) Ответ Исполнителя:\n{executor_result}\n\n"
                f"2) Критика:\n{critic_text}\n\n"
                f"3) Похвала:\n{praise_text}\n\n"
                "Сформируй инструкцию, как улучшить ответ, не переписывая всё с нуля (если нет больших ошибок). "
                "Опиши чётко, что Исполнителю нужно доработать."
            )
            self.add_message("user", user_msg)
            prompt = self.build_prompt()
            if not stream:
                resp = self.client.generate(
                    prompt=prompt,
                    model=self.model_name,
                    stream=False,
                    **ollama_opts
                )
                self.add_message("assistant", resp)
                self.complete_state()
                return resp
            else:
                gen = self.client.generate(
                    prompt=prompt,
                    model=self.model_name,
                    stream=True,
                    **ollama_opts
                )
                self.complete_state()
                return gen
        except Exception as e:
            self.complete_state(error=str(e))
            raise

def init_vector_store():
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = SimpleVectorStore()
        # Добавляем ограничение на размер хранилища
        st.session_state.vector_store.max_documents = 1000
        st.session_state.vector_store.max_document_size = 1000000  # 1MB
