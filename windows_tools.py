import subprocess
import os
import base64
import time
import logging
from typing import Optional, List
from playwright.sync_api import sync_playwright, Page, Browser

# Настройка логирования
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Создаем форматтер для логов
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Добавляем обработчик для вывода в файл
file_handler = logging.FileHandler('browser.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Добавляем обработчик для вывода в консоль
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

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
        self._context = None

    def launch(self) -> None:
        """Запускает браузер."""
        try:
            self._playwright = sync_playwright().start()
            self.browser = self._playwright.chromium.launch(
                headless=self.headless,
                args=['--disable-dev-shm-usage', '--no-sandbox']
            )
            self._context = self.browser.new_context(
                viewport={'width': 1280, 'height': 800},
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                ignore_https_errors=True
            )
            self.page = self._context.new_page()
            self.page.set_default_timeout(self.timeout)
            
            # Настраиваем обработчики
            self._setup_error_handlers()
            self._setup_request_handlers()
            
        except Exception as e:
            self.close()
            raise RuntimeError(f"Ошибка при запуске браузера: {e}")

    def _setup_error_handlers(self) -> None:
        """Настраивает обработчики ошибок."""
        def handle_error(error):
            logger.error(f"Ошибка страницы: {error}")
            self._handle_browser_error("page_error", error)
        
        def handle_crash(error):
            logger.error(f"Сбой страницы: {error}")
            self._handle_browser_error("crash", error)
            self.restart()
        
        def handle_console(msg):
            logger.debug(f"Консоль браузера: {msg.text}")
        
        self.page.on("pageerror", handle_error)
        self.page.on("crash", handle_crash)
        self.page.on("console", handle_console)

    def _setup_request_handlers(self) -> None:
        """Настраивает обработчики запросов."""
        def handle_request(request):
            try:
                logger.debug(f"Запрос: {request.method} {request.url}")
            except Exception as e:
                logger.error(f"Ошибка в обработчике запроса: {e}")
        
        def handle_response(response):
            try:
                logger.debug(f"Ответ: {response.status} {response.url}")
            except Exception as e:
                logger.error(f"Ошибка в обработчике ответа: {e}")
        
        self.page.on("request", handle_request)
        self.page.on("response", handle_response)

    def _handle_browser_error(self, error_type: str, error: Exception) -> None:
        """Обрабатывает ошибки браузера."""
        try:
            error_msg = str(error)
            logger.error(f"Ошибка браузера [{error_type}]: {error_msg}")
            
            # Сохраняем скриншот при ошибке
            if self.page:
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                screenshot_path = f"error_screenshots/browser_error_{timestamp}.png"
                os.makedirs("error_screenshots", exist_ok=True)
                try:
                    self.page.screenshot(path=screenshot_path)
                    logger.info(f"Сохранен скриншот ошибки: {screenshot_path}")
                except Exception as e:
                    logger.error(f"Не удалось сохранить скриншот ошибки: {e}")
            
            # Собираем дополнительную информацию
            debug_info = {
                "url": self.page.url if self.page else "unknown",
                "title": self.page.title() if self.page else "unknown",
                "viewport": self.page.viewport_size if self.page else "unknown",
                "timestamp": time.time()
            }
            logger.debug(f"Отладочная информация: {debug_info}")
            
        except Exception as e:
            logger.error(f"Ошибка при обработке ошибки браузера: {e}")

    def restart(self) -> None:
        """Перезапускает браузер."""
        logger.info("Перезапуск браузера...")
        try:
            self.close()
            time.sleep(1)  # Даем время на освобождение ресурсов
            self.launch()
            logger.info("Браузер успешно перезапущен")
        except Exception as e:
            logger.error(f"Ошибка при перезапуске браузера: {e}")
            raise RuntimeError(f"Не удалось перезапустить браузер: {e}")

    def goto(self, url: str) -> None:
        """
        Переходит по URL.
        
        Args:
            url: Адрес страницы
        """
        self._check_browser()
        try:
            # Проверяем URL
            if not url.startswith(('http://', 'https://')):
                raise ValueError("Некорректный URL")
            
            logger.info(f"Переход по URL: {url}")
            
            # Увеличиваем количество попыток и добавляем задержки
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    # Добавляем случайный User-Agent
                    user_agents = [
                        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
                        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15'
                    ]
                    import random
                    self._context.add_init_script(f"Object.defineProperty(navigator, 'userAgent', {{get: () => '{random.choice(user_agents)}'}});")
                    
                    # Добавляем заголовки для имитации реального браузера
                    extra_headers = {
                        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                        'Accept-Language': 'ru-RU,ru;q=0.8,en-US;q=0.5,en;q=0.3',
                        'DNT': '1',
                        'Connection': 'keep-alive',
                        'Upgrade-Insecure-Requests': '1'
                    }
                    self.page.set_extra_http_headers(extra_headers)
                    
                    response = self.page.goto(
                        url,
                        wait_until='networkidle',
                        timeout=self.timeout
                    )
                    
                    # Проверяем статус ответа
                    if response and response.status == 418:
                        logger.warning(f"Получен статус 418 от DuckDuckGo (попытка {attempt + 1}/{max_retries})")
                        # Увеличиваем задержку с каждой попыткой
                        time.sleep(5 * (attempt + 1))
                        continue
                        
                    if response and response.ok:
                        # Ждем загрузку JavaScript
                        self.page.wait_for_load_state('domcontentloaded')
                        logger.info(f"Успешно загружена страница: {url}")
                        return
                        
                    raise RuntimeError(f"Ошибка загрузки: {response.status if response else 'нет ответа'}")
                    
                except Exception as e:
                    logger.warning(f"Попытка {attempt + 1}/{max_retries} перехода по URL не удалась: {e}")
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(5 * (attempt + 1))
                    
        except Exception as e:
            error_msg = f"Ошибка при переходе по URL {url}: {e}"
            logger.error(error_msg)
            self._handle_browser_error("navigation", e)
            raise RuntimeError(error_msg)

    def screenshot(self, path: str = "screenshot.png") -> str:
        """
        Делает скриншот страницы.
        
        Args:
            path: Путь для сохранения
            
        Returns:
            Путь к сохраненному скриншоту
        """
        self._check_browser()
        try:
            # Проверяем путь
            if not path.endswith(('.png', '.jpg', '.jpeg')):
                path += '.png'
            
            # Ждем стабилизации страницы
            self.page.wait_for_load_state('networkidle')
            
            # Делаем скриншот всей страницы
            self.page.screenshot(
                path=path,
                full_page=True,
                timeout=self.timeout
            )
            return path
        except Exception as e:
            raise RuntimeError(f"Ошибка при создании скриншота: {e}")

    def click(self, selector: str) -> None:
        """
        Кликает по элементу.
        
        Args:
            selector: CSS-селектор
        """
        self._check_browser()
        try:
            # Ждем появления элемента
            element = self.page.wait_for_selector(
                selector,
                state='visible',
                timeout=self.timeout
            )
            if not element:
                raise RuntimeError(f"Элемент не найден: {selector}")
            
            # Скроллим к элементу
            element.scroll_into_view_if_needed()
            
            # Кликаем с повторными попытками
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    element.click(timeout=self.timeout)
                    # Ждем стабилизации после клика
                    self.page.wait_for_load_state('networkidle')
                    return
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(2 ** attempt)
                    
        except Exception as e:
            raise RuntimeError(f"Ошибка при клике по элементу {selector}: {e}")

    def type_text(self, selector: str, text: str) -> None:
        """
        Вводит текст в поле.
        
        Args:
            selector: CSS-селектор
            text: Текст для ввода
        """
        self._check_browser()
        try:
            # Ждем появления элемента
            element = self.page.wait_for_selector(
                selector,
                state='visible',
                timeout=self.timeout
            )
            if not element:
                raise RuntimeError(f"Элемент не найден: {selector}")
            
            # Очищаем поле
            element.evaluate('el => el.value = ""')
            
            # Вводим текст
            element.type(text, delay=100)  # Небольшая задержка между символами
            
            # Имитируем нажатие Enter
            element.press('Enter')
            
            # Ждем стабилизации после ввода
            self.page.wait_for_load_state('networkidle')
            
        except Exception as e:
            raise RuntimeError(f"Ошибка при вводе текста в элемент {selector}: {e}")

    def get_page_title(self) -> str:
        """
        Возвращает заголовок страницы.
        
        Returns:
            Заголовок страницы
        """
        self._check_browser()
        try:
            return self.page.title() or "Без заголовка"
        except Exception as e:
            raise RuntimeError(f"Ошибка при получении заголовка страницы: {e}")

    def check_ssl(self) -> bool:
        """
        Проверяет SSL-сертификат.
        
        Returns:
            True если используется HTTPS
        """
        self._check_browser()
        try:
            return self.page.evaluate("() => window.location.protocol") == "https:"
        except Exception as e:
            raise RuntimeError(f"Ошибка при проверке SSL: {e}")

    def close(self) -> None:
        """Закрывает браузер."""
        try:
            if self.page:
                self.page.close()
            if self._context:
                self._context.close()
            if self.browser:
                self.browser.close()
            if self._playwright:
                self._playwright.stop()
        except Exception as e:
            logger.error(f"Ошибка при закрытии браузера: {e}")
        finally:
            self.page = None
            self._context = None
            self.browser = None
            self._playwright = None

    def _check_browser(self) -> None:
        """
        Проверяет, запущен ли браузер.
        
        Raises:
            RuntimeError: Если браузер не запущен
        """
        if not all([self.page, self._context, self.browser, self._playwright]):
            raise RuntimeError("Браузер не запущен. Вызовите launch()")

    def _wait_for_search_results(self) -> None:
        """Ожидает загрузки результатов поиска."""
        try:
            # Увеличиваем таймаут и добавляем все возможные селекторы
            selectors = [
                '.results', 
                '.result__body',
                '#links',
                '.serp__results',
                '.result',
                '.result__title'
            ]
            
            # Пробуем разные селекторы с увеличенным таймаутом
            timeout = 45000  # 45 секунд
            found = False
            
            for selector in selectors:
                try:
                    logger.debug(f"Ожидание селектора: {selector}")
                    element = self.page.wait_for_selector(
                        selector,
                        state='visible',
                        timeout=timeout
                    )
                    if element:
                        found = True
                        logger.info(f"Найден селектор: {selector}")
                        break
                except Exception as e:
                    logger.debug(f"Селектор {selector} не найден: {e}")
                    continue
            
            if not found:
                raise Exception("Не найдены селекторы результатов поиска")
                
            # Ждем исчезновения индикатора загрузки
            loading_selectors = [
                '.loading',
                '.js-loading',
                '.search__loading',
                '#loading'
            ]
            for loading_selector in loading_selectors:
                try:
                    if self.page.query_selector(loading_selector):
                        logger.debug(f"Ожидание исчезновения: {loading_selector}")
                        self.page.wait_for_selector(
                            loading_selector,
                            state='hidden',
                            timeout=15000
                        )
                except Exception as e:
                    logger.debug(f"Индикатор загрузки не найден: {e}")
                    continue
            
            # Ждем стабилизации сети и даем время для рендеринга
            logger.debug("Ожидание стабилизации сети")
            self.page.wait_for_load_state('networkidle', timeout=timeout)
            time.sleep(3)
            
            # Проверяем наличие результатов
            if not self._has_search_results():
                self._save_debug_screenshot("no_results")
                raise Exception("Результаты поиска не найдены после загрузки")
                
        except Exception as e:
            error_msg = f"Таймаут ожидания результатов поиска: {e}"
            logger.error(error_msg)
            self._save_debug_screenshot("search_timeout")
            raise Exception(error_msg)

    def _has_search_results(self) -> bool:
        """Проверяет наличие результатов поиска."""
        try:
            # Проверяем разные селекторы результатов
            result_selectors = [
                '.result__body',
                '.result__title',
                '#links .result',
                '.serp__results .result'
            ]
            
            for selector in result_selectors:
                elements = self.page.query_selector_all(selector)
                if elements and len(elements) > 0:
                    logger.debug(f"Найдены результаты по селектору: {selector}")
                    return True
                    
            logger.warning("Результаты поиска не найдены")
            return False
            
        except Exception as e:
            logger.error(f"Ошибка при проверке результатов: {e}")
            return False

    def _save_debug_screenshot(self, reason: str) -> None:
        """Сохраняет скриншот для отладки."""
        try:
            if not self.page:
                return
                
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            debug_dir = "debug_screenshots"
            os.makedirs(debug_dir, exist_ok=True)
            
            screenshot_path = f"{debug_dir}/{reason}_{timestamp}.png"
            self.page.screenshot(path=screenshot_path, full_page=True)
            logger.info(f"Сохранен отладочный скриншот: {screenshot_path}")
            
            # Сохраняем HTML для отладки
            html_path = f"{debug_dir}/{reason}_{timestamp}.html"
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(self.page.content())
            logger.info(f"Сохранен HTML страницы: {html_path}")
            
        except Exception as e:
            logger.error(f"Ошибка при сохранении отладочных данных: {e}")

    def _extract_search_results(self) -> List[str]:
        """Извлекает результаты поиска со страницы."""
        results = []
        try:
            # Актуальные селекторы DuckDuckGo
            selectors = {
                'result': ['.result__body', '.result', '.serp__result'],
                'title': ['.result__title', '.result__a', '.serp__title'],
                'link': ['.result__url', '.result__a', '.serp__url'],
                'snippet': ['.result__snippet', '.result__snippet-text', '.serp__snippet']
            }
            
            # Пробуем разные комбинации селекторов
            for result_selector in selectors['result']:
                result_elements = self.page.query_selector_all(result_selector)
                if not result_elements:
                    continue
                
                logger.debug(f"Найдено {len(result_elements)} результатов по селектору {result_selector}")
                
                # Обрабатываем первые 5 результатов
                for i, result in enumerate(result_elements[:5], 1):
                    try:
                        # Пробуем получить компоненты результата
                        title = None
                        link = None
                        snippet = None
                        
                        # Ищем заголовок
                        for title_selector in selectors['title']:
                            title_element = result.query_selector(title_selector)
                            if title_element:
                                title = title_element.inner_text().strip()
                                break
                                
                        # Ищем ссылку
                        for link_selector in selectors['link']:
                            link_element = result.query_selector(link_selector)
                            if link_element:
                                link = link_element.get_attribute('href') or link_element.inner_text().strip()
                                break
                                
                        # Ищем сниппет
                        for snippet_selector in selectors['snippet']:
                            snippet_element = result.query_selector(snippet_selector)
                            if snippet_element:
                                snippet = snippet_element.inner_text().strip()
                                break
                        
                        # Проверяем наличие всех компонентов
                        if all([title, link, snippet]):
                            formatted_result = (
                                f"{i}. {title}\n"
                                f"   URL: {link}\n"
                                f"   {snippet}\n"
                            )
                            results.append(formatted_result)
                            logger.debug(f"Извлечен результат {i}: {title}")
                        else:
                            logger.warning(f"Неполный результат {i}: title={bool(title)}, link={bool(link)}, snippet={bool(snippet)}")
                            
                    except Exception as e:
                        logger.warning(f"Не удалось извлечь результат {i}: {e}")
                        continue
                
                # Если нашли хотя бы один результат, прекращаем поиск
                if results:
                    break
                    
        except Exception as e:
            logger.error(f"Ошибка при извлечении результатов: {e}")
            self._save_debug_screenshot("extraction_error")
            
        return results

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
