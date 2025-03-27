import requests
from typing import List, Union, Generator

class OllamaClient:
    def __init__(self, host: str = "http://localhost:11434"):
        """
        Запускайте Ollama так:
          ollama serve --port 11434
        Этот клиент работает с /api/tags и /api/generate.
        """
        self.host = host

    def list_models(self) -> List[str]:
        """
        Запрашивает GET /api/tags и возвращает список моделей.
        """
        url = f"{self.host}/api/tags"
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            models = data.get("models", [])
            return [m["name"] for m in models if "name" in m]
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
        POST-запрос к /api/generate.
        """
        url = f"{self.host}/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": bool(stream),
            "options": {}
        }
        for k, v in options.items():
            payload["options"][k] = v

        try:
            if not stream:
                resp = requests.post(url, json=payload, timeout=600)
                resp.raise_for_status()
                data = resp.json()
                return data.get("response", "")
            else:
                with requests.post(url, json=payload, stream=True, timeout=600) as r:
                    r.raise_for_status()
                    for line in r.iter_lines(decode_unicode=True):
                        if line:
                            import json
                            try:
                                obj = json.loads(line)
                                chunk = obj.get("response", "")
                                yield chunk
                                if obj.get("done", False):
                                    return
                            except:
                                pass
        except Exception as e:
            print(f"Ошибка при генерации текста: {e}")
            if stream:
                return
            else:
                return ""
