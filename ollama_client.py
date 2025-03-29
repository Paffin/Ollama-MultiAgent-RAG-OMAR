from typing import List, Union, Generator
import requests
import json

class OllamaClient:
    """Клиент для взаимодействия с Ollama API."""
    
    def __init__(self, host: str = "http://localhost:11434"):
        """
        Инициализирует клиент Ollama.
        
        Args:
            host: URL сервера Ollama (по умолчанию http://localhost:11434)
        """
        self.host = host
        self._api_tags_url = f"{host}/api/tags"
        self._api_generate_url = f"{host}/api/generate"

    def list_models(self) -> List[str]:
        """
        Получает список доступных моделей через API.
        
        Returns:
            Список имен моделей
        """
        try:
            response = self._make_request("GET", self._api_tags_url)
            models = response.get("models", [])
            return [model["name"] for model in models if "name" in model]
        except Exception as e:
            print(f"Ошибка при получении списка моделей Ollama: {e}")
            return []
    
    def generate(
        self,
        prompt: str,
        model: str,
        stream: bool = False,
        **options
    ) -> Union[str, Generator[str, None, None]]:
        """
        Генерирует текст с помощью указанной модели.
        
        Args:
            prompt: Текст запроса
            model: Имя модели
            stream: Режим потоковой передачи
            **options: Дополнительные параметры для генерации
            
        Returns:
            Сгенерированный текст или генератор чанков
        """
        payload = self._build_generate_payload(prompt, model, stream, options)
        
        try:
            if not stream:
                return self._generate_single(payload)
            return self._generate_stream(payload)
        except Exception as e:
            print(f"Ошибка при генерации текста: {e}")
            return "" if not stream else None

    def _make_request(self, method: str, url: str, **kwargs) -> dict:
        """Выполняет HTTP запрос к API."""
        response = requests.request(method, url, timeout=600, **kwargs)
        response.raise_for_status()
        return response.json()

    def _build_generate_payload(self, prompt: str, model: str, stream: bool, options: dict) -> dict:
        """Создает payload для запроса генерации."""
        return {
            "model": model,
            "prompt": prompt,
            "stream": bool(stream),
            "options": options
        }

    def _generate_single(self, payload: dict) -> str:
        """Генерирует текст в режиме одиночного запроса."""
        response = self._make_request("POST", self._api_generate_url, json=payload)
        return response.get("response", "")

    def _generate_stream(self, payload: dict) -> Generator[str, None, None]:
        """Генерирует текст в режиме потоковой передачи."""
        with requests.post(self._api_generate_url, json=payload, stream=True) as response:
            response.raise_for_status()
            for line in response.iter_lines(decode_unicode=True):
                if line:
                    try:
                        chunk = self._parse_stream_chunk(line)
                        if chunk:
                            yield chunk
                    except json.JSONDecodeError:
                        continue

    def _parse_stream_chunk(self, line: str) -> str:
        """Парсит чанк из потока."""
        data = json.loads(line)
        if data.get("done", False):
            return None
        return data.get("response", "")
