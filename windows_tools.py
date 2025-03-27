import subprocess
import os
from PIL import Image
import base64
import time
from playwright.sync_api import sync_playwright, Browser, Page
from typing import List, Dict, Any, Optional, Union
import re
import logging
from urllib.parse import urlparse
import ssl
import hashlib
from functools import lru_cache

logger = logging.getLogger(__name__)

def sanitize_command(command: str) -> str:
    """
    Очистка команды от потенциально опасных символов.
    
    Args:
        command: Исходная команда
        
    Returns:
        Очищенная команда
    """
    # Удаляем опасные символы
    command = re.sub(r'[;&|`$]', '', command)
    # Удаляем множественные пробелы
    command = re.sub(r'\s+', ' ', command)
    return command.strip()

def run_system_command(command: str, timeout: int = 30) -> str:
    """
    Выполнение системной команды.
    
    Args:
        command: Команда для выполнения
        timeout: Таймаут выполнения в секундах
        
    Returns:
        Результат выполнения команды
        
    Raises:
        subprocess.TimeoutExpired: При превышении таймаута
        subprocess.SubprocessError: При ошибке выполнения
    """
    try:
        # Очищаем команду
        command = sanitize_command(command)
        
        # Проверяем на опасные команды
        dangerous_commands = ['rm', 'del', 'format', 'mkfs', 'dd']
        if any(cmd in command.lower() for cmd in dangerous_commands):
            raise ValueError("Команда содержит потенциально опасные операции")
            
        # Выполняем команду
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        # Проверяем результат
        if result.returncode != 0:
            logger.warning(f"Команда завершилась с ошибкой: {result.stderr}")
            return f"Ошибка: {result.stderr}"
            
        return result.stdout
        
    except subprocess.TimeoutExpired:
        logger.error(f"Таймаут выполнения команды: {command}")
        raise
    except subprocess.SubprocessError as e:
        logger.error(f"Ошибка выполнения команды: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Неожиданная ошибка: {str(e)}")
        raise

def list_directory(path: str = '.') -> List[str]:
    """
    Получение списка файлов в директории.
    
    Args:
        path: Путь к директории
        
    Returns:
        Список файлов
        
    Raises:
        FileNotFoundError: Если директория не найдена
        PermissionError: Если нет доступа к директории
    """
    try:
        # Очищаем путь
        path = os.path.normpath(path)
        
        # Проверяем существование
        if not os.path.exists(path):
            raise FileNotFoundError(f"Директория не найдена: {path}")
            
        # Проверяем, что это директория
        if not os.path.isdir(path):
            raise NotADirectoryError(f"Путь не является директорией: {path}")
            
        # Получаем список файлов
        files = os.listdir(path)
        
        # Фильтруем скрытые файлы
        files = [f for f in files if not f.startswith('.')]
        
        return files
        
    except Exception as e:
        logger.error(f"Ошибка при получении списка файлов: {str(e)}")
        raise

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
    def __init__(self, timeout: int = 30):
        """
        Инициализация браузера.
        
        Args:
            timeout: Таймаут операций в секундах
        """
        self.timeout = timeout
        self.browser: Optional[Browser] = None
        self.page: Optional[Page] = None
        self.playwright = None
        self._cache = {}
        self.current_url = None
        self.last_screenshot = None
        self.last_analysis = None
        self._page_cache = {}  # Кэш для результатов анализа страницы
        self._timeout = 30000  # Таймаут по умолчанию (30 секунд)

    def launch(self, headless: bool = True) -> None:
        """
        Запуск браузера.
        
        Args:
            headless: Флаг запуска в фоновом режиме
        """
        try:
            self.playwright = sync_playwright().start()
            self.browser = self.playwright.chromium.launch(
                headless=headless,
                args=['--no-sandbox', '--disable-setuid-sandbox']
            )
            self.page = self.browser.new_page()
            
            # Настраиваем SSL
            self.page.route("**/*", self._handle_ssl)
            
        except Exception as e:
            logger.error(f"Ошибка при запуске браузера: {str(e)}")
            raise
            
    def _handle_ssl(self, route) -> None:
        """
        Обработка SSL-сертификатов.
        
        Args:
            route: Маршрут запроса
        """
        try:
            # Проверяем сертификат
            if not route.request.is_navigation_request():
                route.continue_()
                return
                
            url = route.request.url
            parsed = urlparse(url)
            
            # Проверяем домен
            if parsed.netloc in ['localhost', '127.0.0.1']:
                route.continue_()
                return
                
            # Проверяем сертификат
            try:
                ssl.get_server_certificate((parsed.netloc, 443))
                route.continue_()
            except ssl.SSLError:
                logger.warning(f"SSL-сертификат недействителен для {url}")
                route.abort()
                
        except Exception as e:
            logger.error(f"Ошибка при обработке SSL: {str(e)}")
            route.abort()
            
    def goto(self, url: str) -> None:
        """
        Переход по URL.
        
        Args:
            url: URL для перехода
        """
        try:
            # Проверяем URL
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
                
            # Проверяем кэш
            cache_key = hashlib.md5(url.encode()).hexdigest()
            if cache_key in self._cache:
                logger.info(f"Используем кэшированную страницу: {url}")
                self._load_from_cache(cache_key)
                return
                
            # Переходим по URL
            self.page.goto(url, timeout=self.timeout * 1000)
            
            # Сохраняем в кэш
            self._save_to_cache(cache_key)
            
        except Exception as e:
            logger.error(f"Ошибка при переходе по URL: {str(e)}")
            raise
            
    def _save_to_cache(self, key: str) -> None:
        """
        Сохранение страницы в кэш.
        
        Args:
            key: Ключ кэша
        """
        try:
            content = self.page.content()
            self._cache[key] = {
                'content': content,
                'timestamp': time.time()
            }
        except Exception as e:
            logger.error(f"Ошибка при сохранении в кэш: {str(e)}")
            
    def _load_from_cache(self, key: str) -> None:
        """
        Загрузка страницы из кэша.
        
        Args:
            key: Ключ кэша
        """
        try:
            cached = self._cache[key]
            # Проверяем актуальность кэша (1 час)
            if time.time() - cached['timestamp'] > 3600:
                del self._cache[key]
                return
                
            self.page.set_content(cached['content'])
            
        except Exception as e:
            logger.error(f"Ошибка при загрузке из кэша: {str(e)}")
            
    def screenshot(self, path: str) -> None:
        """
        Создание скриншота страницы.
        
        Args:
            path: Путь для сохранения
        """
        try:
            self.page.screenshot(path=path)
        except Exception as e:
            logger.error(f"Ошибка при создании скриншота: {str(e)}")
            raise
            
    def get_text(self) -> str:
        """
        Получение текста страницы.
        
        Returns:
            Текст страницы
        """
        try:
            return self.page.inner_text('body')
        except Exception as e:
            logger.error(f"Ошибка при получении текста: {str(e)}")
            raise
            
    def close(self) -> None:
        """Закрытие браузера."""
        try:
            if self.page:
                self.page.close()
            if self.browser:
                self.browser.close()
            if self.playwright:
                self.playwright.stop()
        except Exception as e:
            logger.error(f"Ошибка при закрытии браузера: {str(e)}")
            
    def __enter__(self):
        """Контекстный менеджер."""
        self.launch()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Закрытие при выходе из контекста."""
        self.close()

    def get_page_title(self) -> str:
        """Возвращает заголовок страницы"""
        return self.page.title()

    def get_current_url(self) -> str:
        """Возвращает текущий URL"""
        return self.page.url

    def click(self, selector: str, timeout: int = None):
        """Кликает по элементу"""
        timeout = timeout or self._timeout
        try:
            self.page.wait_for_selector(selector, timeout=timeout)
            self.page.click(selector)
        except Exception as e:
            print(f"Ошибка при клике по элементу {selector}: {e}")
            raise

    def type_text(self, selector: str, text: str, timeout: int = None):
        """Вводит текст в поле"""
        timeout = timeout or self._timeout
        try:
            self.page.wait_for_selector(selector, timeout=timeout)
            self.page.fill(selector, text)
        except Exception as e:
            print(f"Ошибка при вводе текста в {selector}: {e}")
            raise

    def get_attribute(self, selector: str, attribute: str, timeout: int = None) -> str:
        """Получает значение атрибута элемента"""
        timeout = timeout or self._timeout
        try:
            self.page.wait_for_selector(selector, timeout=timeout)
            return self.page.get_attribute(selector, attribute)
        except Exception as e:
            print(f"Ошибка при получении атрибута {attribute} элемента {selector}: {e}")
            raise

    def wait_for_selector(self, selector: str, timeout: int = None):
        """Ожидает появления элемента"""
        timeout = timeout or self._timeout
        try:
            self.page.wait_for_selector(selector, timeout=timeout)
        except Exception as e:
            print(f"Ошибка при ожидании элемента {selector}: {e}")
            raise

    def wait_for_navigation(self, timeout: int = None):
        """Ожидает завершения навигации"""
        timeout = timeout or self._timeout
        try:
            self.page.wait_for_load_state('networkidle', timeout=timeout)
        except Exception as e:
            print(f"Ошибка при ожидании навигации: {e}")
            raise

    def scroll_to(self, selector: str, timeout: int = None):
        """Прокручивает к элементу"""
        timeout = timeout or self._timeout
        try:
            self.page.wait_for_selector(selector, timeout=timeout)
            self.page.evaluate(f"document.querySelector('{selector}').scrollIntoView()")
        except Exception as e:
            print(f"Ошибка при прокрутке к элементу {selector}: {e}")
            raise

    def analyze_page(self, prompt: str = None) -> str:
        """Анализирует текущую страницу с помощью LLaVA"""
        # Проверяем кэш
        cache_key = f"{self.current_url}_{prompt}"
        if cache_key in self._page_cache:
            return self._page_cache[cache_key]

        if not self.last_screenshot:
            self.screenshot()
        
        if prompt is None:
            prompt = "Опиши текущую страницу, найди все важные элементы (кнопки, поля ввода, ссылки) и их расположение"
        
        try:
            analysis = llava_analyze_screenshot_via_ollama_llava(
                image_path=self.last_screenshot,
                prompt=prompt,
                model="ollama:llava:13b"
            )
            self.last_analysis = analysis
            # Сохраняем в кэш
            self._page_cache[cache_key] = analysis
            return analysis
        except Exception as e:
            print(f"Ошибка при анализе страницы: {e}")
            raise

    def find_element_by_text(self, text: str, timeout: int = None) -> str:
        """Находит селектор элемента по тексту"""
        timeout = timeout or self._timeout
        try:
            self.page.wait_for_load_state('networkidle', timeout=timeout)
            elements = self.page.query_selector_all('*')
            for element in elements:
                if text.lower() in element.inner_text().lower():
                    return element.evaluate('el => el.tagName.toLowerCase() + (el.id ? "#" + el.id : "") + (el.className ? "." + el.className.split(" ").join(".") : "")')
            return None
        except Exception as e:
            print(f"Ошибка при поиске элемента по тексту {text}: {e}")
            raise

    def find_element_by_attribute(self, attribute: str, value: str, timeout: int = None) -> str:
        """Находит селектор элемента по атрибуту"""
        timeout = timeout or self._timeout
        try:
            self.page.wait_for_load_state('networkidle', timeout=timeout)
            elements = self.page.query_selector_all('*')
            for element in elements:
                if element.get_attribute(attribute) == value:
                    return element.evaluate('el => el.tagName.toLowerCase() + (el.id ? "#" + el.id : "") + (el.className ? "." + el.className.split(" ").join(".") : "")')
            return None
        except Exception as e:
            print(f"Ошибка при поиске элемента по атрибуту {attribute}={value}: {e}")
            raise

    def get_form_fields(self) -> List[Dict[str, str]]:
        """Получает список всех полей формы на странице"""
        try:
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
        except Exception as e:
            print(f"Ошибка при получении полей формы: {e}")
            raise

    def get_clickable_elements(self) -> List[Dict[str, str]]:
        """Получает список всех кликабельных элементов на странице"""
        try:
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
        except Exception as e:
            print(f"Ошибка при получении кликабельных элементов: {e}")
            raise

    def wait_for_element_state(self, selector: str, state: str, timeout: int = None):
        """Ожидает определенного состояния элемента"""
        timeout = timeout or self._timeout
        try:
            if state == 'visible':
                self.page.wait_for_selector(selector, state='visible', timeout=timeout)
            elif state == 'hidden':
                self.page.wait_for_selector(selector, state='hidden', timeout=timeout)
            elif state == 'enabled':
                self.page.wait_for_selector(f'{selector}:not([disabled])', timeout=timeout)
            elif state == 'disabled':
                self.page.wait_for_selector(f'{selector}[disabled]', timeout=timeout)
            else:
                raise ValueError(f"Неизвестное состояние элемента: {state}")
        except Exception as e:
            print(f"Ошибка при ожидании состояния {state} элемента {selector}: {e}")
            raise

    def check_element_exists(self, selector: str) -> bool:
        """Проверяет существование элемента"""
        try:
            self.page.wait_for_selector(selector, timeout=1000)
            return True
        except:
            return False

    def get_page_content(self) -> str:
        """Получает содержимое страницы"""
        try:
            return self.page.content()
        except Exception as e:
            print(f"Ошибка при получении содержимого страницы: {e}")
            raise

    def get_element_bounds(self, selector: str) -> Dict[str, int]:
        """Получает координаты и размеры элемента"""
        try:
            return self.page.evaluate(f'document.querySelector("{selector}").getBoundingClientRect()')
        except Exception as e:
            print(f"Ошибка при получении размеров элемента {selector}: {e}")
            raise

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
