from typing import List, Union, Generator
import requests
import json
import time
import logging
from requests.exceptions import RequestException

# Настройка логирования
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Создаем форматтер для логов
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Добавляем обработчик для вывода в файл
file_handler = logging.FileHandler('ollama.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Добавляем обработчик для вывода в консоль
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

class OllamaClient:
    """Клиент для взаимодействия с Ollama API."""
    
    def __init__(self, host: str = "http://localhost:11434", max_retries: int = 3, retry_delay: float = 1.0):
        """
        Инициализирует клиент Ollama.
        
        Args:
            host: URL сервера Ollama
            max_retries: Максимальное количество повторных попыток
            retry_delay: Задержка между попытками в секундах
        """
        self.host = host
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._api_tags_url = f"{host}/api/tags"
        self._api_generate_url = f"{host}/api/generate"
        logger.info(f"Инициализирован Ollama клиент с хостом {host}")

    def list_models(self) -> List[str]:
        """Получает список доступных моделей через API."""
        try:
            for attempt in range(self.max_retries):
                try:
                    response = self._make_request("GET", self._api_tags_url)
                    models = response.get("models", [])
                    model_names = [model["name"] for model in models if "name" in model]
                    logger.info(f"Получен список моделей: {model_names}")
                    return model_names
                except Exception as e:
                    if attempt == self.max_retries - 1:
                        raise
                    logger.warning(f"Попытка {attempt + 1}/{self.max_retries} получения списка моделей не удалась: {e}")
                    time.sleep(self.retry_delay * (2 ** attempt))
        except Exception as e:
            logger.error(f"Ошибка при получении списка моделей: {e}")
            return []
    
    def generate(
        self,
        prompt: str,
        model: str,
        stream: bool = False,
        **options
    ) -> Union[str, Generator[str, None, None]]:
        """Генерирует текст с помощью указанной модели."""
        try:
            payload = self._build_generate_payload(prompt, model, stream, options)
            logger.debug(f"Запрос генерации для модели {model}")
            
            if not stream:
                return self._generate_with_retry(payload)
            return self._generate_stream_with_retry(payload)
            
        except Exception as e:
            error_msg = f"Ошибка при генерации текста: {e}"
            logger.error(error_msg)
            if stream:
                def error_gen():
                    yield error_msg
                return error_gen()
            return error_msg

    def _make_request(self, method: str, url: str, **kwargs) -> dict:
        """Выполняет HTTP запрос к API."""
        try:
            response = requests.request(method, url, timeout=600, **kwargs)
            
            # Проверяем статус ответа
            if response.status_code >= 500:
                raise RequestException(f"Ошибка сервера Ollama: {response.status_code}")
            elif response.status_code >= 400:
                raise RequestException(f"Ошибка запроса: {response.status_code}")
                
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.Timeout:
            raise RequestException("Таймаут запроса к Ollama")
        except requests.exceptions.ConnectionError:
            raise RequestException("Ошибка подключения к Ollama")
        except json.JSONDecodeError:
            raise RequestException("Некорректный JSON в ответе")
        except Exception as e:
            raise RequestException(f"Ошибка запроса: {str(e)}")

    def _build_generate_payload(self, prompt: str, model: str, stream: bool, options: dict) -> dict:
        """Создает payload для запроса генерации."""
        return {
            "model": model,
            "prompt": prompt,
            "stream": bool(stream),
            "options": options
        }

    def _generate_with_retry(self, payload: dict) -> str:
        """Генерирует текст с повторными попытками."""
        for attempt in range(self.max_retries):
            try:
                response = self._make_request("POST", self._api_generate_url, json=payload)
                return response.get("response", "")
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise
                logger.warning(f"Попытка {attempt + 1}/{self.max_retries} генерации не удалась: {e}")
                time.sleep(self.retry_delay * (2 ** attempt))

    def _generate_stream_with_retry(self, payload: dict) -> Generator[str, None, None]:
        """Генерирует текст в потоковом режиме с повторными попытками."""
        for attempt in range(self.max_retries):
            try:
                with requests.post(self._api_generate_url, json=payload, stream=True) as response:
                    response.raise_for_status()
                    for line in response.iter_lines(decode_unicode=True):
                        if line:
                            try:
                                chunk = self._parse_stream_chunk(line)
                                if chunk is not None:
                                    yield chunk
                            except json.JSONDecodeError:
                                logger.warning(f"Некорректный JSON в потоке: {line}")
                                continue
                return
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise
                logger.warning(f"Попытка {attempt + 1}/{self.max_retries} потоковой генерации не удалась: {e}")
                time.sleep(self.retry_delay * (2 ** attempt))

    def _parse_stream_chunk(self, line: str) -> str:
        """Парсит чанк из потока."""
        try:
            data = json.loads(line)
            if data.get("done", False):
                return None
            return data.get("response", "")
        except json.JSONDecodeError as e:
            logger.error(f"Ошибка парсинга JSON: {e}, строка: {line}")
            return ""
        except Exception as e:
            logger.error(f"Ошибка при парсинге чанка: {e}")
            return ""

    def _check_server_health(self) -> bool:
        """Проверяет доступность сервера Ollama."""
        try:
            response = requests.get(f"{self.host}/api/health", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Сервер Ollama недоступен: {e}")
            return False
