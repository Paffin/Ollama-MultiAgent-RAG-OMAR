from typing import List, Dict, Union, Generator
from ollama_client import OllamaClient
from rag_db import SimpleVectorStore
from windows_tools import (
    run_system_command, list_directory,
    PlaywrightBrowser,
    llava_analyze_screenshot_via_ollama_llava
)
import re
import time

class BaseAgent:
    def __init__(self, name: str, system_prompt: str, model_name: str, client: OllamaClient):
        self.name = name
        self.system_prompt = system_prompt
        self.model_name = model_name
        self.client = client
        self.history: List[Dict[str, str]] = []
    
    def add_message(self, role: str, content: str):
        self.history.append({"role": role, "content": content})
    
    def build_prompt(self) -> str:
        parts = [f"System({self.name}): {self.system_prompt}\n"]
        for msg in self.history:
            parts.append(f"{msg['role'].upper()}: {msg['content']}\n")
        return "\n".join(parts)

class PlannerAgent(BaseAgent):
    """
    PlannerAgent теперь выполняет предварительный анализ запроса с использованием LLM.
    Он отправляет запрос к LLM для определения того, какой тип операции требуется:
      - Если локальные данные недостаточны, генерируется инструкция ducksearch:.
      - Если запрос требует операций в браузере (например, регистрация, заполнение формы), генерируется инструкция browser:.
      - Если запрос неоднозначен, может быть запрошено уточнение.
    """
    def analyze_query(self, user_query: str) -> str:
        # Отправляем запрос к LLM для анализа намерений пользователя.
        prompt = (
        f"Проанализируй следующий запрос и ответь кратко в формате:\n"
        f"- Если нужно выполнить поиск в интернете, ответь: ducksearch: <запрос>.\n"
        f"- Если нужно выполнить действия в браузере (например, регистрация, заполнение формы), ответь: browser: open url=<URL>; ...\n"
        f"- Если данные достаточно локальны, просто повтори запрос.\n"
        f"Запрос: {user_query}"
    )
        analysis = self.client.generate(prompt=prompt, model=self.model_name, stream=False)
    
    # Если вернулся генератор, превращаем его в строку
        if not isinstance(analysis, str):
            analysis = ''.join(analysis)

    # Теперь можно спокойно вызвать strip()
        if analysis.strip() == "":
            return user_query
        return analysis.strip()

    def generate_instruction(
        self, 
        user_query: str,
        vector_store: SimpleVectorStore,
        stream=False,
        **ollama_opts
    ) -> Union[str, Generator[str, None, None]]:
        self.add_message("user", user_query)
        # Предварительный анализ запроса с помощью LLM.
        analysis = self.analyze_query(user_query)
        # Если анализ вернул явную команду, используем её; иначе проверяем локальные данные.
        if analysis.lower().startswith("ducksearch:") or analysis.lower().startswith("browser:"):
            plan = analysis
        else:
            local_hits = vector_store.search(user_query, k=1)
            if len(local_hits) == 0:
                plan = f"ducksearch: {user_query}"
            else:
                plan = user_query
        self.add_message("assistant", plan)
        if not stream:
            return plan
        else:
            def gen_plan():
                yield plan
            return gen_plan()

class ExecutorAgent(BaseAgent):
    """
    ExecutorAgent выполняет инструкции:
      - search:, cmd:, ls:, ducksearch:, browser:, visual:
    Если ни одна команда не распознана – генерирует LLM-ответ.
    """
    def execute_instruction(
        self,
        instruction: str,
        vector_store: SimpleVectorStore,
        stream=False,
        **ollama_opts
    ) -> Union[str, Generator[str, None, None]]:
        self.add_message("user", instruction)
        instr_lower = instruction.lower()

        # 1) DuckDuckGo поиск с агрегацией (приоритетная ветка)
        if instr_lower.startswith("ducksearch:"):
            search_query = instruction.split("ducksearch:")[1].strip()
            tool_out = "Результаты поиска в DuckDuckGo:\n"
            aggregated_text = ""
            try:
                from duckduckgo_search import DDGS
                with DDGS() as ddgs:
                    results = ddgs.text(search_query, max_results=5)
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
            return tool_out

        # 2) Локальный поиск через RAG
        elif instr_lower.startswith("search:"):
            query = instruction.split("search:")[1].strip()
            found_docs = vector_store.search(query, k=3)
            tool_out = f"[RAG] Найдено документов: {len(found_docs)}\n"
            for i, d in enumerate(found_docs, 1):
                tool_out += f"{i}. {d[:200]}...\n"
            self.add_message("assistant", tool_out)
            return tool_out

        # 3) Системная команда
        elif instr_lower.startswith("cmd:"):
            cmd_text = instruction.split("cmd:")[1].strip()
            tool_out = run_system_command(cmd_text)
            self.add_message("assistant", tool_out)
            return tool_out

        # 4) Просмотр директории
        elif instr_lower.startswith("ls:"):
            path = instruction.split("ls:")[1].strip()
            tool_out = list_directory(path)
            self.add_message("assistant", tool_out)
            return tool_out

        # 5) Действия в браузере
        elif instr_lower.startswith("browser:"):
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
            return browser_out

        # 6) Анализ визуального содержимого через LLaVA
        elif instr_lower.startswith("visual:"):
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
            return tool_out
        
        # 7) Если ни одна команда не распознана – LLM-ответ
        else:
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
            else:
                gen = self.client.generate(
                    prompt=prompt,
                    model=self.model_name,
                    stream=True,
                    **ollama_opts
                )
                return gen

class CriticAgent(BaseAgent):
    def criticize(self, executor_result: str, stream=False, **ollama_opts) -> Union[str, Generator[str, None, None]]:
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
            return resp
        else:
            gen = self.client.generate(
                prompt=prompt,
                model=self.model_name,
                stream=True,
                **ollama_opts
            )
            return gen

class PraiseAgent(BaseAgent):
    def praise(self, executor_result: str, stream=False, **ollama_opts) -> Union[str, Generator[str, None, None]]:
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
            return resp
        else:
            gen = self.client.generate(
                prompt=prompt,
                model=self.model_name,
                stream=True,
                **ollama_opts
            )
            return gen

class ArbiterAgent(BaseAgent):
    def produce_rework_instruction(
        self,
        executor_result: str,
        critic_text: str,
        praise_text: str,
        stream=False,
        **ollama_opts
    ) -> Union[str, Generator[str, None, None]]:
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
            return resp
        else:
            gen = self.client.generate(
                prompt=prompt,
                model=self.model_name,
                stream=True,
                **ollama_opts
            )
            return gen
