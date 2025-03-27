# windows_tools.py

import subprocess
import os

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
    Список файлов. 
    """
    try:
        files = os.listdir(path)
        return "\n".join(files)
    except Exception as e:
        return f"Ошибка: {e}"
