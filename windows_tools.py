import subprocess
import os
from PIL import Image
import base64
import time
from playwright.sync_api import sync_playwright
from typing import List, Dict

def run_system_command(cmd: str) -> str:
    """
    Запуск shell-команды (cmd.exe / bash).
    Возвращаем stdout + stderr.
    """
    try:
        proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return proc.stdout + "\n" + proc.stderr
    except Exception as e:
        return f"Ошибка при выполнении команды: {e}"

def list_directory(path: str = ".") -> str:
    """
    Список файлов в директории.
    """
    try:
        files = os.listdir(path)
        return "\n".join(files)
    except Exception as e:
        return f"Ошибка: {e}"

def duckduckgo_search(query: str, max_results: int = 5) -> str:
    """
    Выполняет поиск в DuckDuckGo с помощью duckduckgo_search.
    Возвращает строку-список результатов.
    """
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=max_results)
        if not results:
            return "Результаты не найдены."
        output = "Результаты поиска в DuckDuckGo:\n"
        for i, r in enumerate(results, start=1):
            title = r.get('title', '')
            link = r.get('href', '')
            snippet = r.get('body', '')
            output += f"{i}) {title}\n   URL: {link}\n   {snippet}\n\n"
        return output
    except Exception as e:
        return f"Ошибка при поиске DuckDuckGo: {e}"

class PlaywrightBrowser:
    """
    Упрощённая обёртка для браузера (через Playwright).
    """
    def __init__(self, headless: bool = True):
        self.headless = headless
        self.browser = None
        self.context = None
        self.page = None
        self.current_url = None
        self.last_screenshot = None
        self.last_analysis = None

    def launch(self):
        """Запускает браузер и создает новый контекст"""
        from playwright.sync_api import sync_playwright
        playwright = sync_playwright().start()
        self.browser = playwright.chromium.launch(headless=self.headless)
        self.context = self.browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        )
        self.page = self.context.new_page()

    def close(self):
        """Закрывает браузер и освобождает ресурсы"""
        if self.page:
            self.page.close()
        if self.context:
            self.context.close()
        if self.browser:
            self.browser.close()

    def goto(self, url: str):
        """Переходит по указанному URL"""
        self.page.goto(url)
        self.current_url = url
        self.page.wait_for_load_state('networkidle')

    def get_page_title(self) -> str:
        """Возвращает заголовок страницы"""
        return self.page.title()

    def get_current_url(self) -> str:
        """Возвращает текущий URL"""
        return self.page.url

    def screenshot(self, path: str = None) -> str:
        """Делает скриншот страницы"""
        if path is None:
            path = f"screenshot_{int(time.time())}.png"
        self.page.screenshot(path=path, full_page=True)
        self.last_screenshot = path
        return path

    def click(self, selector: str, timeout: int = 5000):
        """Кликает по элементу"""
        self.page.wait_for_selector(selector, timeout=timeout)
        self.page.click(selector)

    def type_text(self, selector: str, text: str, timeout: int = 5000):
        """Вводит текст в поле"""
        self.page.wait_for_selector(selector, timeout=timeout)
        self.page.fill(selector, text)

    def get_text(self, selector: str, timeout: int = 5000) -> str:
        """Получает текст элемента"""
        self.page.wait_for_selector(selector, timeout=timeout)
        return self.page.inner_text(selector)

    def get_attribute(self, selector: str, attribute: str, timeout: int = 5000) -> str:
        """Получает значение атрибута элемента"""
        self.page.wait_for_selector(selector, timeout=timeout)
        return self.page.get_attribute(selector, attribute)

    def wait_for_selector(self, selector: str, timeout: int = 5000):
        """Ожидает появления элемента"""
        self.page.wait_for_selector(selector, timeout=timeout)

    def wait_for_navigation(self, timeout: int = 5000):
        """Ожидает завершения навигации"""
        self.page.wait_for_load_state('networkidle', timeout=timeout)

    def scroll_to(self, selector: str, timeout: int = 5000):
        """Прокручивает к элементу"""
        self.page.wait_for_selector(selector, timeout=timeout)
        self.page.evaluate(f"document.querySelector('{selector}').scrollIntoView()")

    def analyze_page(self, prompt: str = None) -> str:
        """Анализирует текущую страницу с помощью LLaVA"""
        if not self.last_screenshot:
            self.screenshot()
        
        if prompt is None:
            prompt = "Опиши текущую страницу, найди все важные элементы (кнопки, поля ввода, ссылки) и их расположение"
        
        analysis = llava_analyze_screenshot_via_ollama_llava(
            image_path=self.last_screenshot,
            prompt=prompt,
            model="ollama:llava:13b"
        )
        self.last_analysis = analysis
        return analysis

    def find_element_by_text(self, text: str, timeout: int = 5000) -> str:
        """Находит селектор элемента по тексту"""
        self.page.wait_for_load_state('networkidle')
        elements = self.page.query_selector_all('*')
        for element in elements:
            if text.lower() in element.inner_text().lower():
                return element.evaluate('el => el.tagName.toLowerCase() + (el.id ? "#" + el.id : "") + (el.className ? "." + el.className.split(" ").join(".") : "")')
        return None

    def find_element_by_attribute(self, attribute: str, value: str, timeout: int = 5000) -> str:
        """Находит селектор элемента по атрибуту"""
        self.page.wait_for_load_state('networkidle')
        elements = self.page.query_selector_all('*')
        for element in elements:
            if element.get_attribute(attribute) == value:
                return element.evaluate('el => el.tagName.toLowerCase() + (el.id ? "#" + el.id : "") + (el.className ? "." + el.className.split(" ").join(".") : "")')
        return None

    def get_form_fields(self) -> List[Dict[str, str]]:
        """Получает список всех полей формы на странице"""
        fields = []
        inputs = self.page.query_selector_all('input, select, textarea')
        for input_el in inputs:
            field_type = input_el.get_attribute('type') or input_el.evaluate('el => el.tagName.toLowerCase()')
            field_name = input_el.get_attribute('name') or input_el.get_attribute('id') or ''
            field_label = ''
            
            # Ищем label для поля
            label = input_el.evaluate('el => el.labels ? el.labels[0]?.textContent : ""')
            if label:
                field_label = label
            
            fields.append({
                'type': field_type,
                'name': field_name,
                'label': field_label,
                'selector': input_el.evaluate('el => el.tagName.toLowerCase() + (el.id ? "#" + el.id : "") + (el.className ? "." + el.className.split(" ").join(".") : "")')
            })
        return fields

    def get_clickable_elements(self) -> List[Dict[str, str]]:
        """Получает список всех кликабельных элементов на странице"""
        elements = []
        clickables = self.page.query_selector_all('button, a, [role="button"], input[type="submit"], input[type="button"]')
        for el in clickables:
            text = el.inner_text()
            if text:
                elements.append({
                    'text': text,
                    'selector': el.evaluate('el => el.tagName.toLowerCase() + (el.id ? "#" + el.id : "") + (el.className ? "." + el.className.split(" ").join(".") : "")')
                })
        return elements

    def wait_for_element_state(self, selector: str, state: str, timeout: int = 5000):
        """Ожидает определенного состояния элемента"""
        if state == 'visible':
            self.page.wait_for_selector(selector, state='visible', timeout=timeout)
        elif state == 'hidden':
            self.page.wait_for_selector(selector, state='hidden', timeout=timeout)
        elif state == 'enabled':
            self.page.wait_for_selector(f'{selector}:not([disabled])', timeout=timeout)
        elif state == 'disabled':
            self.page.wait_for_selector(f'{selector}[disabled]', timeout=timeout)

    def check_element_exists(self, selector: str) -> bool:
        """Проверяет существование элемента"""
        try:
            self.page.wait_for_selector(selector, timeout=1000)
            return True
        except:
            return False

    def get_page_content(self) -> str:
        """Получает содержимое страницы"""
        return self.page.content()

    def get_element_bounds(self, selector: str) -> Dict[str, int]:
        """Получает координаты и размеры элемента"""
        return self.page.evaluate(f'document.querySelector("{selector}").getBoundingClientRect()')

def llava_analyze_screenshot_via_ollama_llava(image_path: str, prompt: str, model="ollama:llava:13b") -> str:
    """
    Отправляем изображение в Ollama (модель LLaVA).
    Предполагается, что Ollama принимает поле "image" (base64).
    """
    import requests
    try:
        with open(image_path, "rb") as f:
            img_data = f.read()
        img_b64 = base64.b64encode(img_data).decode("utf-8")
        url = "http://localhost:11434/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "image": img_b64,
            "stream": False,
            "options": {}
        }
        resp = requests.post(url, json=payload, timeout=300)
        resp.raise_for_status()
        data = resp.json()
        return data.get("response", "")
    except Exception as e:
        return f"[Ошибка LLaVA] {e}"
