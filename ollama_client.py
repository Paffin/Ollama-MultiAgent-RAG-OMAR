import requests
import json
from typing import Dict, Any, List, Optional, Generator, Union
import time
from requests.exceptions import RequestException, Timeout, ConnectionError
import logging

class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434", timeout: int = 30):
        """
        Инициализация клиента Ollama.
        
        Args:
            base_url: Базовый URL сервера Ollama
            timeout: Таймаут запросов в секундах
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()
        self.logger = logging.getLogger(__name__)
        
    def _make_request(
        self, 
        method: str, 
        endpoint: str, 
        data: Optional[Dict[str, Any]] = None,
        stream: bool = False
    ) -> Union[Dict[str, Any], Generator[str, None, None]]:
        """
        Выполнение HTTP запроса к API Ollama.
        
        Args:
            method: HTTP метод
            endpoint: Конечная точка API
            data: Данные запроса
            stream: Флаг потокового режима
            
        Returns:
            Ответ от API
            
        Raises:
            OllamaError: При ошибке запроса
            OllamaTimeoutError: При превышении таймаута
            OllamaConnectionError: При ошибке подключения
        """
        url = f"{self.base_url}{endpoint}"
        
        try:
            response = self.session.request(
                method=method,
                url=url,
                json=data,
                stream=stream,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            if stream:
                return self._stream_response(response)
            else:
                return response.json()
                
        except Timeout:
            self.logger.error(f"Таймаут запроса к {url}")
            raise OllamaTimeoutError(f"Таймаут запроса к {url}")
        except ConnectionError:
            self.logger.error(f"Ошибка подключения к {url}")
            raise OllamaConnectionError(f"Ошибка подключения к {url}")
        except RequestException as e:
            self.logger.error(f"Ошибка запроса к {url}: {str(e)}")
            raise OllamaError(f"Ошибка запроса к {url}: {str(e)}")
            
    def _stream_response(self, response: requests.Response) -> Generator[str, None, None]:
        """
        Обработка потокового ответа.
        
        Args:
            response: Объект ответа
            
        Yields:
            Части ответа
        """
        for line in response.iter_lines():
            if line:
                try:
                    data = json.loads(line)
                    if 'response' in data:
                        yield data['response']
                except json.JSONDecodeError:
                    self.logger.warning(f"Ошибка декодирования JSON: {line}")
                    continue
                    
    def list_models(self) -> List[str]:
        """
        Получение списка доступных моделей.
        
        Returns:
            Список имен моделей
        """
        try:
            response = self._make_request('GET', '/api/tags')
            return [model['name'] for model in response.get('models', [])]
        except Exception as e:
            self.logger.error(f"Ошибка при получении списка моделей: {str(e)}")
            return []
            
    def generate(
        self,
        model: str,
        prompt: str,
        stream: bool = True,
        options: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Union[str, Generator[str, None, None]]:
        """
        Генерация текста с помощью модели.
        
        Args:
            model: Имя модели
            prompt: Текст запроса
            stream: Флаг потокового режима
            options: Дополнительные параметры
            **kwargs: Дополнительные параметры
            
        Returns:
            Сгенерированный текст или генератор частей текста
        """
        data = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
            **(options or {}),
            **kwargs
        }
        
        try:
            response = self._make_request('POST', '/api/generate', data=data, stream=stream)
            
            if stream:
                return response
            else:
                return response.get('response', '')
                
        except Exception as e:
            self.logger.error(f"Ошибка при генерации текста: {str(e)}")
            raise
            
    def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        stream: bool = True,
        options: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Union[str, Generator[str, None, None]]:
        """
        Чат с моделью.
        
        Args:
            model: Имя модели
            messages: Список сообщений
            stream: Флаг потокового режима
            options: Дополнительные параметры
            **kwargs: Дополнительные параметры
            
        Returns:
            Ответ модели или генератор частей ответа
        """
        data = {
            "model": model,
            "messages": messages,
            "stream": stream,
            **(options or {}),
            **kwargs
        }
        
        try:
            response = self._make_request('POST', '/api/chat', data=data, stream=stream)
            
            if stream:
                return response
            else:
                return response.get('message', {}).get('content', '')
                
        except Exception as e:
            self.logger.error(f"Ошибка при чате с моделью: {str(e)}")
            raise
            
    def embeddings(
        self,
        model: str,
        prompt: str,
        options: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[float]:
        """
        Получение эмбеддингов для текста.
        
        Args:
            model: Имя модели
            prompt: Текст запроса
            options: Дополнительные параметры
            **kwargs: Дополнительные параметры
            
        Returns:
            Список эмбеддингов
        """
        data = {
            "model": model,
            "prompt": prompt,
            **(options or {}),
            **kwargs
        }
        
        try:
            response = self._make_request('POST', '/api/embeddings', data=data)
            return response.get('embedding', [])
            
        except Exception as e:
            self.logger.error(f"Ошибка при получении эмбеддингов: {str(e)}")
            raise
            
    def pull(self, model: str) -> Dict[str, Any]:
        """
        Загрузка модели.
        
        Args:
            model: Имя модели
            
        Returns:
            Информация о загрузке
        """
        try:
            return self._make_request('POST', '/api/pull', data={"name": model})
        except Exception as e:
            self.logger.error(f"Ошибка при загрузке модели: {str(e)}")
            raise
            
    def push(self, model: str) -> Dict[str, Any]:
        """
        Отправка модели.
        
        Args:
            model: Имя модели
            
        Returns:
            Информация об отправке
        """
        try:
            return self._make_request('POST', '/api/push', data={"name": model})
        except Exception as e:
            self.logger.error(f"Ошибка при отправке модели: {str(e)}")
            raise
            
    def delete(self, model: str) -> Dict[str, Any]:
        """
        Удаление модели.
        
        Args:
            model: Имя модели
            
        Returns:
            Информация об удалении
        """
        try:
            return self._make_request('DELETE', f'/api/delete/{model}')
        except Exception as e:
            self.logger.error(f"Ошибка при удалении модели: {str(e)}")
            raise

class OllamaError(Exception):
    """Базовый класс для ошибок Ollama."""
    pass

class OllamaTimeoutError(OllamaError):
    """Ошибка таймаута запроса."""
    pass

class OllamaConnectionError(OllamaError):
    """Ошибка подключения к серверу."""
    pass
