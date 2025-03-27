import subprocess
import os
from PIL import Image
import base64
import time
from playwright.sync_api import sync_playwright

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
        self.page = None

    def launch(self):
        from playwright.sync_api import sync_playwright
        p = sync_playwright().start()
        self.browser = p.chromium.launch(headless=self.headless)
        context = self.browser.new_context()
        self.page = context.new_page()

    def goto(self, url: str):
        if not self.page:
            raise RuntimeError("Browser not launched. Call launch() first.")
        self.page.goto(url)
        time.sleep(2)  # ожидание загрузки

    def screenshot(self, path: str = "screenshot.png"):
        if not self.page:
            raise RuntimeError("Browser not launched. Call launch() first.")
        self.page.screenshot(path=path)
        return path

    def click(self, selector: str):
        if not self.page:
            raise RuntimeError("Browser not launched. Call launch() first.")
        self.page.click(selector)
        time.sleep(1)

    def type_text(self, selector: str, text: str):
        if not self.page:
            raise RuntimeError("Browser not launched. Call launch() first.")
        self.page.fill(selector, text)
        time.sleep(1)

    def get_page_title(self) -> str:
        if not self.page:
            return ""
        return self.page.title()

    def check_ssl(self) -> bool:
        if not self.page:
            return False
        # Проверка, что протокол равен "https:"
        return self.page.evaluate("() => window.location.protocol") == "https:"

    def close(self):
        if self.browser:
            self.browser.close()
            self.browser = None
            self.page = None

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
