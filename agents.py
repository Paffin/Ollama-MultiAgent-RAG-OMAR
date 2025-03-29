from typing import List, Dict, Union, Generator, Optional
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
    
    def add_message(self, role: str, content: str):
        self.history.append({"role": role, "content": content})
    
    def build_prompt(self) -> str:
        parts = [f"System({self.name}): {self.system_prompt}\n"]
        if self.context:
            parts.append("Context:")
            for k, v in self.context.items():
                parts.append(f"- {k}: {v}")
        for msg in self.history:
            parts.append(f"{msg['role'].upper()}: {msg['content']}\n")
        return "\n".join(parts)

    def update_state(self, status: AgentStatus, task: str, metadata: Dict = None):
        if self.state is None or self.state.status == AgentStatus.COMPLETED:
            self.state = AgentState(
                status=status,
                current_task=task,
                start_time=time.time(),
                metadata=metadata or {}
            )
        else:
            self.state.status = status
            self.state.current_task = task
            if metadata:
                self.state.metadata.update(metadata)

    def complete_state(self, error: str = None):
        if self.state:
            self.state.status = AgentStatus.ERROR if error else AgentStatus.COMPLETED
            self.state.end_time = time.time()
            if error:
                self.state.error = error

    def get_state_summary(self) -> str:
        if not self.state:
            return "No active state"
        duration = self.state.end_time - self.state.start_time if self.state.end_time else time.time() - self.state.start_time
        return f"Status: {self.state.status.value}, Task: {self.state.current_task}, Duration: {duration:.2f}s"

    def clear_history(self):
        self.history = []
        self.context = {}
        self.state = None

class PlannerAgent(BaseAgent):
    """
    PlannerAgent теперь выполняет предварительный анализ запроса с использованием LLM.
    Он отправляет запрос к LLM для определения того, какой тип операции требуется:
      - Если локальные данные недостаточны, генерируется инструкция ducksearch:.
      - Если запрос требует операций в браузере (например, регистрация, заполнение формы), генерируется инструкция browser:.
      - Если запрос неоднозначен, может быть запрошено уточнение.
    """
    def __init__(self, name: str, system_prompt: str, model_name: str, client: OllamaClient):
        super().__init__(name, system_prompt or PLANNER_PROMPT, model_name, client)

    def create_detailed_plan(self, user_query: str) -> str:
        """
        Создаёт детальный план действий на основе запроса пользователя.
        """
        self.update_state(AgentStatus.PLANNING, "Creating detailed plan")
        try:
            prompt = (
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
            plan = self.client.generate(prompt=prompt, model=self.model_name, stream=False)
            if not isinstance(plan, str):
                plan = ''.join(plan)
            self.complete_state()
            return plan.strip()
        except Exception as e:
            self.complete_state(error=str(e))
            raise

    def analyze_query(self, user_query: str) -> str:
        self.update_state(AgentStatus.PLANNING, "Analyzing query")
        try:
            prompt = (
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
            analysis = self.client.generate(prompt=prompt, model=self.model_name, stream=False)
            
            if not isinstance(analysis, str):
                analysis = ''.join(analysis)

            if analysis.strip() == "":
                return user_query
            self.complete_state()
            return analysis.strip()
        except Exception as e:
            self.complete_state(error=str(e))
            raise

    def generate_instruction(
        self, 
        user_query: str,
        vector_store: SimpleVectorStore,
        stream=False,
        **ollama_opts
    ) -> Union[str, Generator[str, None, None]]:
        self.update_state(AgentStatus.PLANNING, "Generating instruction")
        try:
            self.add_message("user", user_query)
            
            # Анализируем запрос
            analysis = self.analyze_query(user_query)
            self.add_message("assistant", f"Анализ запроса:\n{analysis}")
            
            # Создаём детальный план
            detailed_plan = self.create_detailed_plan(user_query)
            self.add_message("assistant", f"Детальный план:\n{detailed_plan}")
            
            # Определяем необходимые инструменты на основе анализа
            if "ducksearch:" in user_query.lower():
                plan = user_query
            elif "browser:" in user_query.lower():
                plan = user_query
            elif "visual:" in user_query.lower():
                plan = user_query
            elif "cmd:" in user_query.lower():
                plan = user_query
            elif "ls:" in user_query.lower():
                plan = user_query
            else:
                # Проверяем локальные данные
                local_hits = vector_store.search(user_query, k=1)
                if len(local_hits) == 0:
                    # Если локальных данных нет, используем поиск
                    plan = f"ducksearch: {user_query}"
                else:
                    # Если есть локальные данные, используем их
                    plan = user_query
                    
            # Формируем финальную инструкцию
            final_instruction = (
                f"Анализ запроса:\n{analysis}\n\n"
                f"План действий:\n{detailed_plan}\n\n"
                f"Инструкция для выполнения:\n{plan}"
            )
            self.add_message("assistant", final_instruction)
            
            if not stream:
                self.complete_state()
                return final_instruction
            else:
                def gen_plan():
                    yield final_instruction
                self.complete_state()
                return gen_plan()
        except Exception as e:
            self.complete_state(error=str(e))
            raise

class ExecutorAgent(BaseAgent):
    """
    ExecutorAgent выполняет инструкции:
      - search:, cmd:, ls:, ducksearch:, browser:, visual:
    Если ни одна команда не распознана – генерирует LLM-ответ.
    """
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
        self.update_state(AgentStatus.EXECUTING, "Executing instruction")
        try:
            self.add_message("user", instruction)
            instr_lower = instruction.lower()

            # Извлекаем план действий и инструкцию
            plan_and_instruction = instruction.split("\n\n")
            if len(plan_and_instruction) > 1:
                plan = plan_and_instruction[0].replace("План действий:", "").strip()
                instruction = plan_and_instruction[1].replace("Инструкция для выполнения:", "").strip()
                self.context["plan"] = plan

            # Проверяем наличие команды ducksearch в инструкции
            if "ducksearch:" in instruction:
                self.update_state(AgentStatus.EXECUTING, "Performing DuckDuckGo search")
                search_query = instruction.split("ducksearch:")[1].strip()
                tool_out = "Результаты поиска в DuckDuckGo:\n"
                aggregated_text = ""
                try:
                    from duckduckgo_search import DDGS
                    with DDGS() as ddgs:
                        results = ddgs.text(search_query, max_results=15)
                    if not results:
                        tool_out += "Результаты не найдены."
                    else:
                        for i, r in enumerate(results, start=1):
                            title = r.get('title', '')
                            link = r.get('href', '')
                            snippet = r.get('body', '')
                            tool_out += f"{i}) {title}\n   URL: {link}\n   {snippet}\n\n"
                            if link:
                                try:
                                    br = PlaywrightBrowser(headless=True)
                                    start_time = time.time()
                                    br.launch()
                                    br.goto(link)
                                    load_time = time.time() - start_time
                                    # Ожидание загрузки содержимого
                                    br.page.wait_for_selector("body", timeout=5000)
                                    page_text = br.page.inner_text("body")
                                    ssl_ok = "https:" in br.page.evaluate("() => window.location.protocol")
                                    aggregated_text += (
                                        f"URL: {link}\n"
                                        f"Title: {br.get_page_title()}\n"
                                        f"Load Time: {load_time:.2f} сек\n"
                                        f"SSL: {'Да' if ssl_ok else 'Нет'}\n"
                                        f"Content (первые 1000 символов):\n{page_text[:1000]}\n\n"
                                    )
                                    br.close()
                                except Exception as e:
                                    aggregated_text += f"Ошибка при извлечении информации с {link}: {e}\n\n"
                        if aggregated_text.strip():
                            summary_prompt = (
                                f"Сделай сводку по запросу: '{search_query}'.\n"
                                f"Исходный текст:\n\n{aggregated_text}\n\nСводка:"
                            )
                            summary = self.client.generate(
                                prompt=summary_prompt,
                                model=self.model_name,
                                stream=False,
                                **ollama_opts
                            )
                            if not isinstance(summary, str):
                                summary = ''.join(summary)
                            tool_out += "\nСводка полученной информации:\n" + summary
                        else:
                            tool_out += "\nНе удалось извлечь информацию с найденных ссылок."
                except Exception as e:
                    tool_out = f"Ошибка при веб-поиске: {e}"
                self.add_message("assistant", tool_out)
                self.complete_state()
                return tool_out

            # 2) Локальный поиск через RAG
            elif instr_lower.startswith("search:"):
                self.update_state(AgentStatus.EXECUTING, "Performing local search")
                query = instruction.split("search:")[1].strip()
                found_docs = vector_store.search(query, k=3)
                tool_out = f"[RAG] Найдено документов: {len(found_docs)}\n"
                for i, d in enumerate(found_docs, 1):
                    tool_out += f"{i}. {d[:200]}...\n"
                self.add_message("assistant", tool_out)
                self.complete_state()
                return tool_out

            # 3) Системная команда
            elif instr_lower.startswith("cmd:"):
                self.update_state(AgentStatus.EXECUTING, "Executing system command")
                cmd_text = instruction.split("cmd:")[1].strip()
                tool_out = run_system_command(cmd_text)
                self.add_message("assistant", tool_out)
                self.complete_state()
                return tool_out

            # 4) Просмотр директории
            elif instr_lower.startswith("ls:"):
                self.update_state(AgentStatus.EXECUTING, "Listing directory")
                path = instruction.split("ls:")[1].strip()
                tool_out = list_directory(path)
                self.add_message("assistant", tool_out)
                self.complete_state()
                return tool_out

            # 5) Действия в браузере
            elif instr_lower.startswith("browser:"):
                self.update_state(AgentStatus.EXECUTING, "Performing browser actions")
                browser_out = ""
                actions = instruction.split("browser:")[1].split(";")
                br = PlaywrightBrowser(headless=True)
                br.launch()
                try:
                    for act in actions:
                        act = act.strip()
                        if act.startswith("open url="):
                            url = act.replace("open url=", "").strip()
                            br.goto(url)
                            browser_out += f"Открыли URL: {url}\n"
                        elif act.startswith("screenshot path="):
                            sc_path = act.replace("screenshot path=", "").strip()
                            path_taken = br.screenshot(path=sc_path)
                            browser_out += f"Скриншот сохранён в {path_taken}\n"
                        elif act.startswith("click selector="):
                            sel = act.replace("click selector=", "").strip()
                            br.click(sel)
                            browser_out += f"Кликнули по селектору {sel}\n"
                        elif act.startswith("type selector="):
                            parts = act.split(",")
                            if len(parts) == 2:
                                sel_part = parts[0].replace("type selector=", "").strip()
                                text_part = parts[1].replace("text=", "").strip()
                                br.type_text(sel_part, text_part)
                                browser_out += f"В селектор {sel_part} введён текст {text_part}\n"
                        elif act.startswith("parse selector="):
                            sel = act.replace("parse selector=", "").strip()
                            try:
                                parsed_text = br.page.inner_text(sel)
                                browser_out += f"Текст из {sel}: {parsed_text}\n"
                            except Exception as e:
                                browser_out += f"Ошибка при извлечении текста по селектору {sel}: {e}\n"
                finally:
                    br.close()
                self.add_message("assistant", browser_out)
                self.complete_state()
                return browser_out

            # 6) Анализ визуального содержимого через LLaVA
            elif instr_lower.startswith("visual:"):
                self.update_state(AgentStatus.EXECUTING, "Analyzing visual content")
                sub = instruction.split("visual:")[1].strip()
                if "||" in sub:
                    img_path, prompt_text = sub.split("||", 1)
                    img_path = img_path.strip()
                    prompt_text = prompt_text.strip()
                else:
                    img_path = sub
                    prompt_text = "Describe the screenshot"
                tool_out = llava_analyze_screenshot_via_ollama_llava(
                    image_path=img_path,
                    prompt=prompt_text,
                    model="ollama:llava:13b"
                )
                self.add_message("assistant", tool_out)
                self.complete_state()
                return tool_out
            
            # 7) Если ни одна команда не распознана – LLM-ответ
            else:
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

    def __del__(self):
        """Очистка ресурсов при удалении агента"""
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
