import subprocess
import os
import base64
import time
from typing import Optional
from playwright.sync_api import sync_playwright, Page, Browser

def run_system_command(cmd: str) -> str:
    """
    Выполняет системную команду.
    
    Args:
        cmd: Команда для выполнения
        
    Returns:
        Результат выполнения команды (stdout + stderr)
    """
    try:
        process = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return process.stdout + "\n" + process.stderr
    except Exception as e:
        return f"Ошибка при выполнении команды: {e}"

def list_directory(path: str = ".") -> str:
    """
    Возвращает список файлов в директории.
    
    Args:
        path: Путь к директории
        
    Returns:
        Список файлов или сообщение об ошибке
    """
    try:
        files = os.listdir(path)
        return "\n".join(files)
    except Exception as e:
        return f"Ошибка: {e}"

def duckduckgo_search(query: str, max_results: int = 5) -> str:
    """
    Выполняет поиск в DuckDuckGo.
    
    Args:
        query: Поисковый запрос
        max_results: Максимальное количество результатов
        
    Returns:
        Отформатированные результаты поиска
    """
    try:
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            return "Ошибка: пакет duckduckgo-search не установлен. Установите его через pip install duckduckgo-search"
            
        with DDGS() as ddgs:
            try:
                results = list(ddgs.text(query, max_results=max_results, timeout=30))
            except Exception as e:
                return f"Ошибка при выполнении поиска: {e}"
            
        if not results:
            return "Результаты не найдены."
            
        return _format_search_results(results)
    except Exception as e:
        return f"Ошибка при поиске DuckDuckGo: {e}"

def _format_search_results(results: list) -> str:
    """Форматирует результаты поиска."""
    output = "Результаты поиска в DuckDuckGo:\n"
    for i, result in enumerate(results, start=1):
        output += (
            f"{i}) {result.get('title', '')}\n"
            f"   URL: {result.get('href', '')}\n"
            f"   {result.get('body', '')}\n\n"
        )
    return output

class PlaywrightBrowser:
    """Управление браузером через Playwright."""
    
    def __init__(self, headless: bool = True, timeout: int = 30000):
        """
        Инициализирует браузер.
        
        Args:
            headless: Режим без GUI
            timeout: Таймаут в миллисекундах
        """
        self.headless = headless
        self.timeout = timeout
        self.browser: Optional[Browser] = None
        self.page: Optional[Page] = None
        self._playwright = None

    def launch(self) -> None:
        """Запускает браузер."""
        try:
            self._playwright = sync_playwright().start()
            self.browser = self._playwright.chromium.launch(headless=self.headless)
            context = self.browser.new_context(
                viewport={'width': 1280, 'height': 800},
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            )
            self.page = context.new_page()
            self.page.set_default_timeout(self.timeout)
        except Exception as e:
            self.close()
            raise RuntimeError(f"Ошибка при запуске браузера: {e}")

    def goto(self, url: str) -> None:
        """
        Переходит по URL.
        
        Args:
            url: Адрес страницы
            
        Raises:
            RuntimeError: Если браузер не запущен или произошла ошибка
        """
        self._check_browser()
        try:
            self.page.goto(url, wait_until='networkidle', timeout=self.timeout)
        except Exception as e:
            raise RuntimeError(f"Ошибка при переходе по URL {url}: {e}")

    def screenshot(self, path: str = "screenshot.png") -> str:
        """
        Делает скриншот страницы.
        
        Args:
            path: Путь для сохранения
            
        Returns:
            Путь к сохраненному скриншоту
            
        Raises:
            RuntimeError: Если браузер не запущен или произошла ошибка
        """
        self._check_browser()
        try:
            self.page.screenshot(path=path, timeout=self.timeout)
            return path
        except Exception as e:
            raise RuntimeError(f"Ошибка при создании скриншота: {e}")

    def click(self, selector: str) -> None:
        """
        Кликает по элементу.
        
        Args:
            selector: CSS-селектор
            
        Raises:
            RuntimeError: Если браузер не запущен или произошла ошибка
        """
        self._check_browser()
        try:
            self.page.click(selector, timeout=self.timeout)
            self.page.wait_for_load_state('networkidle', timeout=self.timeout)
        except Exception as e:
            raise RuntimeError(f"Ошибка при клике по элементу {selector}: {e}")

    def type_text(self, selector: str, text: str) -> None:
        """
        Вводит текст в поле.
        
        Args:
            selector: CSS-селектор
            text: Текст для ввода
            
        Raises:
            RuntimeError: Если браузер не запущен или произошла ошибка
        """
        self._check_browser()
        try:
            self.page.fill(selector, text, timeout=self.timeout)
        except Exception as e:
            raise RuntimeError(f"Ошибка при вводе текста в элемент {selector}: {e}")

    def get_page_title(self) -> str:
        """
        Возвращает заголовок страницы.
        
        Returns:
            Заголовок страницы
            
        Raises:
            RuntimeError: Если браузер не запущен или произошла ошибка
        """
        self._check_browser()
        try:
            return self.page.title()
        except Exception as e:
            raise RuntimeError(f"Ошибка при получении заголовка страницы: {e}")

    def check_ssl(self) -> bool:
        """
        Проверяет SSL-сертификат.
        
        Returns:
            True если используется HTTPS
            
        Raises:
            RuntimeError: Если браузер не запущен или произошла ошибка
        """
        self._check_browser()
        try:
            return self.page.evaluate("() => window.location.protocol") == "https:"
        except Exception as e:
            raise RuntimeError(f"Ошибка при проверке SSL: {e}")

    def close(self) -> None:
        """Закрывает браузер."""
        try:
            if self.browser:
                self.browser.close()
            if self._playwright:
                self._playwright.stop()
        except Exception as e:
            logger.error(f"Ошибка при закрытии браузера: {e}")
        finally:
            self.browser = None
            self.page = None
            self._playwright = None

    def _check_browser(self) -> None:
        """
        Проверяет, запущен ли браузер.
        
        Raises:
            RuntimeError: Если браузер не запущен
        """
        if not self.page or not self.browser:
            raise RuntimeError("Браузер не запущен. Вызовите launch()")

def llava_analyze_screenshot_via_ollama_llava(
    image_path: str,
    prompt: str,
    model: str = "ollama:llava:13b"
) -> str:
    """
    Анализирует изображение через LLaVA.
    
    Args:
        image_path: Путь к изображению
        prompt: Текст запроса
        model: Имя модели
        
    Returns:
        Результат анализа или сообщение об ошибке
    """
    try:
        image_data = _read_image(image_path)
        image_b64 = _encode_image(image_data)
        response = _send_to_ollama(image_b64, prompt, model)
        return response.get("response", "")
    except Exception as e:
        return f"[Ошибка LLaVA] {e}"

def _read_image(path: str) -> bytes:
    """Читает изображение из файла."""
    with open(path, "rb") as f:
        return f.read()

def _encode_image(data: bytes) -> str:
    """Кодирует изображение в base64."""
    return base64.b64encode(data).decode("utf-8")

def _send_to_ollama(image_b64: str, prompt: str, model: str) -> dict:
    """Отправляет запрос в Ollama."""
    import requests
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "image": image_b64,
        "stream": False,
        "options": {}
    }
    response = requests.post(url, json=payload, timeout=300)
    response.raise_for_status()
    return response.json()
