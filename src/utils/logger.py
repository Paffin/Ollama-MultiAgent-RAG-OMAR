import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime
from logging.handlers import RotatingFileHandler
import os

class Logger:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Logger, cls).__new__(cls)
            cls._instance._initialize_logger()
        return cls._instance
    
    def _initialize_logger(self):
        """Инициализация логгера"""
        try:
            self.logger = logging.getLogger('omar')
            self.logger.setLevel(logging.DEBUG)
            
            # Создаем директорию для логов с обработкой ошибок
            try:
                log_dir = Path('logs')
                log_dir.mkdir(exist_ok=True, parents=True)
            except PermissionError:
                print("Ошибка: Нет прав на создание директории логов")
                sys.exit(1)
            except Exception as e:
                print(f"Ошибка при создании директории логов: {e}")
                sys.exit(1)
            
            # Форматтер для логов
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            
            # Хендлер для файла с ротацией
            log_file = log_dir / f'omar_{datetime.now().strftime("%Y%m%d")}.log'
            try:
                file_handler = RotatingFileHandler(
                    log_file,
                    maxBytes=10*1024*1024,  # 10MB
                    backupCount=5,
                    encoding='utf-8'
                )
                file_handler.setLevel(logging.DEBUG)
                file_handler.setFormatter(formatter)
            except PermissionError:
                print(f"Ошибка: Нет прав на запись в файл логов: {log_file}")
                sys.exit(1)
            except Exception as e:
                print(f"Ошибка при создании файла логов: {e}")
                sys.exit(1)
            
            # Хендлер для консоли
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(formatter)
            
            # Добавляем хендлеры
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)
            
        except Exception as e:
            print(f"Критическая ошибка при инициализации логгера: {e}")
            sys.exit(1)
        
    def debug(self, message: str, **kwargs):
        """Логирование отладочного сообщения"""
        self.logger.debug(message, extra=kwargs)
        
    def info(self, message: str, **kwargs):
        """Логирование информационного сообщения"""
        self.logger.info(message, extra=kwargs)
        
    def warning(self, message: str, **kwargs):
        """Логирование предупреждения"""
        self.logger.warning(message, extra=kwargs)
        
    def error(self, message: str, **kwargs):
        """Логирование ошибки"""
        self.logger.error(message, extra=kwargs)
        
    def critical(self, message: str, **kwargs):
        """Логирование критической ошибки"""
        self.logger.critical(message, extra=kwargs)
        
    def exception(self, message: str, **kwargs):
        """Логирование исключения"""
        self.logger.exception(message, extra=kwargs)
        
    def set_level(self, level: int):
        """Установка уровня логирования"""
        self.logger.setLevel(level)
        
    def add_file_handler(self, file_path: str, level: Optional[int] = None):
        """Добавление дополнительного файлового хендлера"""
        try:
            handler = logging.FileHandler(file_path, encoding='utf-8')
            if level:
                handler.setLevel(level)
            handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
            self.logger.addHandler(handler)
            return True
        except Exception as e:
            self.error(f"Ошибка добавления файлового хендлера: {e}")
            return False 