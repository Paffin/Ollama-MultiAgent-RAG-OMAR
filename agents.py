from typing import List, Dict, Union, Generator
from ollama_client import OllamaClient
from rag_db import SimpleVectorStore
from windows_tools import (
    run_system_command, list_directory,
    PlaywrightBrowser,
    llava_analyze_screenshot_via_ollama_llava
)
from prompts import (
    PLANNER_PROMPT,
    EXECUTOR_PROMPT,
    CRITIC_PROMPT,
    PRAISE_PROMPT,
    ARBITER_PROMPT,
    PLANNER_ANALYSIS_PROMPT
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

    def evaluate_response_quality(self, response: str) -> float:
        """
        Оценивает качество ответа по шкале от 0 до 1.
        """
        evaluation_prompt = f"""
        Оцени качество следующего ответа по шкале от 0 до 1, где:
        0 - полностью неудовлетворительный ответ
        1 - идеальный ответ
        
        Критерии оценки:
        1. Полнота ответа (0.2):
           - Все ли аспекты запроса учтены
           - Достаточно ли деталей
           - Нет ли пропущенных важных моментов
        
        2. Точность информации (0.2):
           - Корректность фактов
           - Актуальность данных
           - Отсутствие ошибок
        
        3. Структурированность (0.15):
           - Логическая организация
           - Четкость изложения
           - Удобство восприятия
        
        4. Полезность (0.15):
           - Практическая применимость
           - Решение проблемы
           - Полезность для пользователя
        
        5. Техническое качество (0.15):
           - Правильность использования инструментов
           - Обработка ошибок
           - Оптимальность решения
        
        6. Соответствие запросу (0.15):
           - Точное соответствие требованиям
           - Учет контекста
           - Релевантность ответа
        
        Ответ для оценки:
        {response}
        
        Формат ответа:
        [Оценка]: число от 0 до 1
        [Детали]: краткое объяснение оценки
        [Критические моменты]: что нужно улучшить
        """
        
        try:
            evaluation = self.client.generate(
                prompt=evaluation_prompt,
                model=self.model_name,
                stream=False
            )
            if not isinstance(evaluation, str):
                evaluation = ''.join(evaluation)
            
            # Извлекаем оценку из ответа
            score_match = re.search(r'\[Оценка\]:\s*(\d*\.?\d+)', evaluation)
            if score_match:
                score = float(score_match.group(1))
                return min(max(score, 0.0), 1.0)  # Ограничиваем диапазон
            
            # Если не удалось найти оценку, пробуем найти число в тексте
            number_match = re.search(r'0\.\d+|1\.0|1', evaluation)
            if number_match:
                score = float(number_match.group())
                return min(max(score, 0.0), 1.0)
            
            print(f"Не удалось извлечь оценку из ответа: {evaluation}")
            return 0.5
            
        except (ValueError, TypeError) as e:
            print(f"Ошибка при извлечении оценки: {e}")
            return 0.5
        except Exception as e:
            print(f"Неожиданная ошибка при оценке качества: {e}")
            return 0.5

class PlannerAgent(BaseAgent):
    """
    PlannerAgent теперь выполняет предварительный анализ запроса с использованием LLM.
    Он отправляет запрос к LLM для определения того, какой тип операции требуется:
      - Если локальные данные недостаточны, генерируется инструкция ducksearch:.
      - Если запрос требует операций в браузере (например, регистрация, заполнение формы), генерируется инструкция browser:.
      - Если запрос неоднозначен, может быть запрошено уточнение.
      - Если запрос требует комплексного решения, генерируется инструкция complex:.
    """
    def __init__(self, name: str, model_name: str, client: OllamaClient):
        super().__init__(name, PLANNER_PROMPT, model_name, client)

    def analyze_query(self, user_query: str) -> str:
        """
        Анализирует запрос пользователя и определяет тип инструкции.
        Всегда возвращает строку с префиксом типа инструкции.
        """
        try:
            # Отправляем запрос к LLM для анализа намерений пользователя
            prompt = PLANNER_ANALYSIS_PROMPT.format(user_query=user_query)
            analysis = self.client.generate(prompt=prompt, model=self.model_name, stream=False)
        
            # Если вернулся генератор, превращаем его в строку
            if not isinstance(analysis, str):
                analysis = ''.join(analysis)

            # Извлекаем все важные компоненты анализа
            analysis_match = re.search(r'\[Анализ\]:\s*(.*?)(?=\[|$)', analysis, re.DOTALL)
            recommendation_match = re.search(r'\[Рекомендация\]:\s*(\w+)', analysis)
            justification_match = re.search(r'\[Обоснование\]:\s*(.*?)(?=\[|$)', analysis, re.DOTALL)
            additional_match = re.search(r'\[Дополнительно\]:\s*(.*?)(?=\[|$)', analysis, re.DOTALL)

            # Определяем тип запроса и необходимые действия
            query_type = None
            required_actions = []
            data_sources = []

            if analysis_match:
                analysis_text = analysis_match.group(1).strip()
                # Анализируем тип запроса
                if any(word in analysis_text.lower() for word in ['курс', 'погода', 'новости', 'актуальный', 'сейчас']):
                    query_type = 'ducksearch'
                elif any(word in analysis_text.lower() for word in ['браузер', 'сайт', 'страница', 'форма', 'регистрация']):
                    query_type = 'browser'
                elif any(word in analysis_text.lower() for word in ['файл', 'директория', 'папка', 'система']):
                    query_type = 'cmd'
                elif any(word in analysis_text.lower() for word in ['изображение', 'скриншот', 'фото', 'картинка']):
                    query_type = 'visual'
                elif any(word in analysis_text.lower() for word in ['и', 'затем', 'после', 'потом', 'сначала', 'далее']):
                    query_type = 'complex'
                elif any(word in analysis_text.lower() for word in ['поиск', 'найти', 'найти в']):
                    query_type = 'search'

            # Если тип не определен по анализу, используем рекомендацию
            if not query_type and recommendation_match:
                recommendation = recommendation_match.group(1).lower()
                if recommendation in ['ducksearch', 'browser', 'cmd', 'visual', 'complex', 'search']:
                    query_type = recommendation

            # Формируем инструкцию на основе определенного типа
            if query_type:
                if query_type == 'ducksearch':
                    return f"ducksearch: {user_query}"
                elif query_type == 'browser':
                    return f"browser: {user_query}"
                elif query_type == 'cmd':
                    return f"cmd: {user_query}"
                elif query_type == 'visual':
                    return f"visual: {user_query}"
                elif query_type == 'search':
                    return f"search: {user_query}"
                elif query_type == 'complex':
                    # Разбиваем запрос на шаги
                    steps = []
                    # Разбиваем по союзам и предлогам
                    parts = re.split(r'\s+(?:и|затем|после|потом|сначала|далее|затем|в конце)\s+', user_query.lower())
                    for part in parts:
                        part = part.strip()
                        if part and len(part) > 2:  # Игнорируем слишком короткие части
                            steps.append(part)
                    
                    if len(steps) > 1:
                        # Формируем комплексное решение с полными шагами
                        full_steps = []
                        current_pos = 0
                        for step in steps:
                            # Находим позицию шага в оригинальном запросе
                            pos = user_query.lower().find(step, current_pos)
                            if pos != -1:
                                # Берем оригинальный текст шага с сохранением регистра
                                full_step = user_query[pos:pos + len(step)]
                                full_steps.append(full_step)
                                current_pos = pos + len(step)
                        
                        if full_steps:
                            return "complex: " + "; ".join(full_steps)

            # Если не удалось определить тип запроса, используем LLM
            return f"llm: {user_query}"
            
        except Exception as e:
            print(f"Ошибка при анализе запроса: {e}")
            return f"llm: {user_query}"  # В случае ошибки используем LLM

    def generate_instruction(
        self, 
        user_query: str,
        vector_store: SimpleVectorStore,
        stream=False,
        **ollama_opts
    ) -> Union[str, Generator[str, None, None]]:
        self.add_message("user", user_query)
        
        # Предварительный анализ запроса с помощью LLM
        analysis = self.analyze_query(user_query)
        
        # Если анализ вернул явную команду, используем её
        if analysis.lower().startswith(("ducksearch:", "browser:", "llm:", "search:", "cmd:", "ls:", "visual:", "complex:")):
            plan = analysis
        else:
            # Проверяем локальные данные
            local_hits = vector_store.search(user_query, k=1)
            if len(local_hits) == 0:
                # Если нет локальных данных, используем ducksearch для актуальных данных
                if any(word in user_query.lower() for word in ['курс', 'погода', 'новости', 'актуальный', 'сейчас']):
                    plan = f"ducksearch: {user_query}"
                # Используем browser для веб-действий
                elif any(word in user_query.lower() for word in ['браузер', 'сайт', 'страница', 'форма', 'регистрация']):
                    plan = f"browser: {user_query}"
                # Используем cmd для системных операций
                elif any(word in user_query.lower() for word in ['файл', 'директория', 'папка', 'система']):
                    plan = f"cmd: {user_query}"
                # Используем visual для работы с изображениями
                elif any(word in user_query.lower() for word in ['изображение', 'скриншот', 'фото', 'картинка']):
                    plan = f"visual: {user_query}"
                # Используем complex для многошаговых задач
                elif any(word in user_query.lower() for word in ['и', 'затем', 'после', 'потом', 'сначала', 'далее']):
                    steps = re.split(r'\s+(?:и|затем|после|потом|сначала|далее|затем|в конце)\s+', user_query.lower())
                    if len(steps) > 1:
                        plan = "complex: " + "; ".join(steps)
                    else:
                        plan = f"llm: {user_query}"
                # Используем search для поисковых запросов
                elif any(word in user_query.lower() for word in ['поиск', 'найти', 'найти в']):
                    plan = f"search: {user_query}"
                # В остальных случаях используем LLM
                else:
                    plan = f"llm: {user_query}"
            else:
                plan = f"search: {user_query}"
                
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
      - llm: локальный LLM-запрос
      - ducksearch: поиск в интернете
      - browser: действия в браузере
      - search: поиск в локальных данных
      - cmd: системные команды
      - ls: просмотр директорий
      - visual: анализ визуального контента
      - complex: комплексные решения
    """
    def __init__(self, name: str, model_name: str, client: OllamaClient):
        super().__init__(name, EXECUTOR_PROMPT, model_name, client)

    def execute_instruction(
        self,
        instruction: str,
        vector_store: SimpleVectorStore,
        stream=False,
        **ollama_opts
    ) -> Union[str, Generator[str, None, None]]:
        self.add_message("user", instruction)
        instr_lower = instruction.lower()

        # Инициализация переменных
        steps = []
        goal = None
        required_data = []
        results = []

        # 1) Комплексное решение (приоритетная ветка)
        if instr_lower.startswith("complex:"):
            steps = instruction.split("complex:")[1].split(";")
            for step in steps:
                step_result = self.execute_instruction(
                    step,
                    vector_store,
                    stream=False,
                    **ollama_opts
                )
                if isinstance(step_result, str):
                    results.append(f"Шаг: {step}\nРезультат: {step_result}\n")
                else:
                    results.append(f"Шаг: {step}\nРезультат: [Генератор]\n")
            
            final_result = "Результаты выполнения комплексного решения:\n\n" + "\n".join(results)
            self.add_message("assistant", final_result)
            return final_result

        # 2) Локальный LLM-запрос
        elif instr_lower.startswith("llm:"):
            query = instruction.split("llm:")[1].strip()
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

        # 3) Умное взаимодействие с браузером
        elif instr_lower.startswith("browser:"):
            browser_out = ""
            br = PlaywrightBrowser(headless=True)
            br.launch()
            try:
                # Анализируем запрос для определения действий
                analysis_prompt = f"""
                Проанализируй запрос пользователя и определи необходимые действия в браузере:
                {instruction}
                
                Формат ответа:
                [Цель]: краткое описание цели
                [Шаги]: список шагов для достижения цели
                [Требуемые данные]: какие данные нужно собрать или ввести
                """
                
                analysis = self.client.generate(
                    prompt=analysis_prompt,
                    model=self.model_name,
                    stream=False,
                    **ollama_opts
                )
                
                if not isinstance(analysis, str):
                    analysis = ''.join(analysis)
                
                browser_out += f"Анализ запроса:\n{analysis}\n\n"
                
                # Извлекаем цель и шаги
                goal_match = re.search(r'\[Цель\]:\s*(.*?)(?=\[|$)', analysis)
                steps_match = re.search(r'\[Шаги\]:\s*(.*?)(?=\[|$)', analysis, re.DOTALL)
                data_match = re.search(r'\[Требуемые данные\]:\s*(.*?)(?=\[|$)', analysis, re.DOTALL)
                
                if goal_match:
                    goal = goal_match.group(1).strip()
                    browser_out += f"Цель: {goal}\n\n"
                
                if steps_match:
                    steps = [s.strip() for s in steps_match.group(1).split('\n') if s.strip()]
                    browser_out += "Шаги:\n"
                    for i, step in enumerate(steps, 1):
                        browser_out += f"{i}. {step}\n"
                    browser_out += "\n"
                
                if data_match:
                    required_data = [d.strip() for d in data_match.group(1).split('\n') if d.strip()]
                    browser_out += "Требуемые данные:\n"
                    for data in required_data:
                        browser_out += f"- {data}\n"
                    browser_out += "\n"
                
                # Выполняем шаги
                for i, step in enumerate(steps, 1):
                    browser_out += f"\nВыполнение шага {i}: {step}\n"
                    
                    # Анализируем текущую страницу
                    page_analysis = br.analyze_page()
                    browser_out += f"Анализ страницы:\n{page_analysis}\n"
                    
                    # Определяем действие для текущего шага
                    action_prompt = f"""
                    На основе текущего шага и анализа страницы определи следующее действие:
                    
                    Шаг: {step}
                    Анализ страницы: {page_analysis}
                    
                    Возможные действия:
                    1. Переход по URL
                    2. Поиск и клик по элементу
                    3. Ввод текста в поле
                    4. Ожидание появления элемента
                    5. Прокрутка страницы
                    6. Сбор данных
                    
                    Формат ответа:
                    [Действие]: тип действия
                    [Селектор]: CSS-селектор элемента (если применимо)
                    [Значение]: значение для ввода (если применимо)
                    [Обоснование]: почему выбрано это действие
                    """
                    
                    action_analysis = self.client.generate(
                        prompt=action_prompt,
                        model=self.model_name,
                        stream=False,
                        **ollama_opts
                    )
                    
                    if not isinstance(action_analysis, str):
                        action_analysis = ''.join(action_analysis)
                    
                    browser_out += f"Анализ действия:\n{action_analysis}\n"
                    
                    # Извлекаем параметры действия
                    action_match = re.search(r'\[Действие\]:\s*(\w+)', action_analysis)
                    selector_match = re.search(r'\[Селектор\]:\s*(.*?)(?=\[|$)', action_analysis)
                    value_match = re.search(r'\[Значение\]:\s*(.*?)(?=\[|$)', action_analysis)
                    
                    if action_match:
                        action = action_match.group(1).lower()
                        
                        if action == 'url':
                            if value_match:
                                url = value_match.group(1).strip()
                                br.goto(url)
                                browser_out += f"Переход по URL: {url}\n"
                        
                        elif action == 'click':
                            if selector_match:
                                selector = selector_match.group(1).strip()
                                try:
                                    br.click(selector)
                                    browser_out += f"Клик по элементу: {selector}\n"
                                except Exception as e:
                                    browser_out += f"Ошибка при клике: {e}\n"
                        
                        elif action == 'type':
                            if selector_match and value_match:
                                selector = selector_match.group(1).strip()
                                value = value_match.group(1).strip()
                                try:
                                    br.type_text(selector, value)
                                    browser_out += f"Ввод текста в {selector}: {value}\n"
                                except Exception as e:
                                    browser_out += f"Ошибка при вводе текста: {e}\n"
                        
                        elif action == 'wait':
                            if selector_match:
                                selector = selector_match.group(1).strip()
                                try:
                                    br.wait_for_selector(selector)
                                    browser_out += f"Ожидание элемента: {selector}\n"
                                except Exception as e:
                                    browser_out += f"Ошибка при ожидании: {e}\n"
                        
                        elif action == 'scroll':
                            if selector_match:
                                selector = selector_match.group(1).strip()
                                try:
                                    br.scroll_to(selector)
                                    browser_out += f"Прокрутка к элементу: {selector}\n"
                                except Exception as e:
                                    browser_out += f"Ошибка при прокрутке: {e}\n"
                        
                        elif action == 'collect':
                            if selector_match:
                                selector = selector_match.group(1).strip()
                                try:
                                    value = br.get_text(selector)
                                    browser_out += f"Собраны данные: {value}\n"
                                except Exception as e:
                                    browser_out += f"Ошибка при сборе данных: {e}\n"
                    
                    # Ожидаем загрузки страницы после действия
                    br.wait_for_navigation()
                
                # Проверяем успешность выполнения
                success_prompt = f"""
                Проверь, достигнута ли цель:
                Цель: {goal}
                Текущий URL: {br.get_current_url()}
                Заголовок страницы: {br.get_page_title()}
                
                Формат ответа:
                [Успех]: да/нет
                [Причина]: объяснение
                [Следующие шаги]: что нужно сделать дальше
                """
                
                success_analysis = self.client.generate(
                    prompt=success_prompt,
                    model=self.model_name,
                    stream=False,
                    **ollama_opts
                )
                
                if not isinstance(success_analysis, str):
                    success_analysis = ''.join(success_analysis)
                
                browser_out += f"\nАнализ результата:\n{success_analysis}\n"
                
            finally:
                br.close()
            
            self.add_message("assistant", browser_out)
            return browser_out

        # 4) DuckDuckGo поиск с агрегацией
        elif instr_lower.startswith("ducksearch:"):
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

        # 5) Локальный поиск через RAG
        elif instr_lower.startswith("search:"):
            query = instruction.split("search:")[1].strip()
            found_docs = vector_store.search(query, k=3)
            tool_out = f"[RAG] Найдено документов: {len(found_docs)}\n"
            for i, d in enumerate(found_docs, 1):
                tool_out += f"{i}. {d[:200]}...\n"
            self.add_message("assistant", tool_out)
            return tool_out

        # 6) Системная команда
        elif instr_lower.startswith("cmd:"):
            cmd_text = instruction.split("cmd:")[1].strip()
            tool_out = run_system_command(cmd_text)
            self.add_message("assistant", tool_out)
            return tool_out

        # 7) Просмотр директории
        elif instr_lower.startswith("ls:"):
            path = instruction.split("ls:")[1].strip()
            tool_out = list_directory(path)
            self.add_message("assistant", tool_out)
            return tool_out

        # 8) Анализ визуального содержимого через LLaVA
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
        
        # 9) Если ни одна команда не распознана – LLM-ответ
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
    def __init__(self, name: str, model_name: str, client: OllamaClient):
        super().__init__(name, CRITIC_PROMPT, model_name, client)

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
    def __init__(self, name: str, model_name: str, client: OllamaClient):
        super().__init__(name, PRAISE_PROMPT, model_name, client)

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
    def __init__(self, name: str, model_name: str, client: OllamaClient):
        super().__init__(name, ARBITER_PROMPT, model_name, client)

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
