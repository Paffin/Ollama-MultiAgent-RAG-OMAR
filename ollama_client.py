# ollama_client.py

import requests
from typing import List, Union, Generator

class OllamaClient:
    def __init__(self, host: str = "http://localhost:11434"):
        """
        Запускайте Ollama так:
          ollama serve --port 11434
        Тогда этот клиент будет ходить на /api/tags и /api/generate
        """
        self.host = host

    def list_models(self) -> List[str]:
        """
        Запрашиваем GET /api/tags, получаем JSON вида:
          {"models": [ {"name": "llama3.2", ...}, ... ]}
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
        POST /api/generate
        - prompt: текст
        - model: имя модели (из /api/tags)
        - stream=True => вернём генератор
        - options => dict с параметрами (temperature, top_p, frequency_penalty, presence_penalty, num_ctx, num_predict, ...)
        """
        url = f"{self.host}/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": bool(stream),
            "options": {}
        }
        # Помещаем доп. настройки в payload["options"]
        for k, v in options.items():
            payload["options"][k] = v

        try:
            if not stream:
                # Обычный запрос
                resp = requests.post(url, json=payload, timeout=600)
                resp.raise_for_status()
                data = resp.json()
                return data.get("response", "")
            else:
                # Поток chunk'ов
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
                                # пропускаем строки, не парсящиеся как JSON
                                pass
        except Exception as e:
            print(f"Ошибка при генерации текста: {e}")
            if stream:
                return
            else:
                return ""
