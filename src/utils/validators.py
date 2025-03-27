from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
import re
from datetime import datetime
import pandas as pd
import numpy as np
from utils.logger import Logger

@dataclass
class ValidationRule:
    """Правило валидации"""
    type: str
    params: Dict[str, Any]
    message: str

@dataclass
class ValidationResult:
    """Результат валидации"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    stats: Dict[str, Any]

class DataValidator:
    """Класс для валидации данных"""
    
    def __init__(self):
        self.rules: Dict[str, List[ValidationRule]] = {}
        
    def add_validation_rule(
        self,
        column: str,
        rule_type: str,
        params: Dict[str, Any],
        message: str
    ) -> None:
        """
        Добавление правила валидации
        
        Args:
            column: Колонка для валидации
            rule_type: Тип правила
            params: Параметры правила
            message: Сообщение об ошибке
        """
        if column not in self.rules:
            self.rules[column] = []
            
        self.rules[column].append(
            ValidationRule(
                type=rule_type,
                params=params,
                message=message
            )
        )
        
    def validate_data(self, data: pd.DataFrame) -> ValidationResult:
        """
        Валидация данных
        
        Args:
            data: DataFrame для валидации
            
        Returns:
            Результат валидации
        """
        errors = []
        warnings = []
        stats = {}
        
        for column, rules in self.rules.items():
            if column not in data.columns:
                errors.append(f"Колонка {column} не найдена в данных")
                continue
                
            column_stats = {
                'total': len(data[column]),
                'null_count': data[column].isna().sum(),
                'unique_count': data[column].nunique()
            }
            
            for rule in rules:
                if not self._validate_rule(data[column], rule):
                    errors.append(rule.message)
                    
            stats[column] = column_stats
            
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            stats=stats
        )
        
    def _validate_rule(self, series: pd.Series, rule: ValidationRule) -> bool:
        """
        Проверка правила валидации
        
        Args:
            series: Серия данных
            rule: Правило валидации
            
        Returns:
            True если правило выполнено, False в противном случае
        """
        if rule.type == 'type':
            return self._validate_type(series, rule.params)
        elif rule.type == 'range':
            return self._validate_range(series, rule.params)
        elif rule.type == 'unique':
            return self._validate_unique(series, rule.params)
        elif rule.type == 'format':
            return self._validate_format(series, rule.params)
        elif rule.type == 'custom':
            return self._validate_custom(series, rule.params)
        return False
        
    def _validate_type(self, series: pd.Series, params: Dict[str, Any]) -> bool:
        """Проверка типа данных"""
        expected_type = params.get('type')
        if expected_type == str:
            return series.apply(lambda x: isinstance(x, str)).all()
        elif expected_type == int:
            return series.apply(lambda x: isinstance(x, (int, np.integer))).all()
        elif expected_type == float:
            return series.apply(lambda x: isinstance(x, (float, np.floating))).all()
        elif expected_type == bool:
            return series.apply(lambda x: isinstance(x, bool)).all()
        elif expected_type == datetime:
            return series.apply(lambda x: isinstance(x, datetime)).all()
        return False
        
    def _validate_range(self, series: pd.Series, params: Dict[str, Any]) -> bool:
        """Проверка диапазона значений"""
        min_val = params.get('min')
        max_val = params.get('max')
        
        if min_val is not None:
            if not (series >= min_val).all():
                return False
                
        if max_val is not None:
            if not (series <= max_val).all():
                return False
                
        return True
        
    def _validate_unique(self, series: pd.Series, params: Dict[str, Any]) -> bool:
        """Проверка уникальности значений"""
        return series.nunique() == len(series)
        
    def _validate_format(self, series: pd.Series, params: Dict[str, Any]) -> bool:
        """Проверка формата значений"""
        pattern = params.get('pattern')
        if pattern:
            return series.apply(lambda x: bool(re.match(pattern, str(x)))).all()
        return False
        
    def _validate_custom(self, series: pd.Series, params: Dict[str, Any]) -> bool:
        """Проверка пользовательской функции"""
        func = params.get('func')
        if func and callable(func):
            return series.apply(func).all()
        return False

def validate_config(config: Dict[str, Any]) -> bool:
    """
    Валидация конфигурации
    
    Args:
        config: Конфигурация для валидации
        
    Returns:
        True если конфигурация валидна
    """
    logger = Logger()
    
    try:
        # Проверяем обязательные секции
        required_sections = [
            'ollama',
            'agents',
            'data',
            'notifications',
            'analytics',
            'logging',
            'cache'
        ]
        
        for section in required_sections:
            if section not in config:
                logger.error(f"Отсутствует обязательная секция конфигурации: {section}")
                return False
                
        # Проверяем параметры Ollama
        ollama_config = config['ollama']
        required_ollama_params = ['base_url', 'timeout', 'models']
        for param in required_ollama_params:
            if param not in ollama_config:
                logger.error(f"Отсутствует обязательный параметр Ollama: {param}")
                return False
                
        # Проверяем параметры агентов
        agents_config = config['agents']
        required_agents_params = ['temperature', 'top_p', 'max_tokens', 'context_window']
        for param in required_agents_params:
            if param not in agents_config:
                logger.error(f"Отсутствует обязательный параметр агентов: {param}")
                return False
                
        # Проверяем параметры данных
        data_config = config['data']
        required_data_params = ['chunk_size', 'min_text_length', 'max_text_length', 'supported_formats']
        for param in required_data_params:
            if param not in data_config:
                logger.error(f"Отсутствует обязательный параметр данных: {param}")
                return False
                
        # Проверяем параметры уведомлений
        notifications_config = config['notifications']
        required_notification_params = ['max_history', 'priority_levels', 'categories']
        for param in required_notification_params:
            if param not in notifications_config:
                logger.error(f"Отсутствует обязательный параметр уведомлений: {param}")
                return False
                
        # Проверяем параметры аналитики
        analytics_config = config['analytics']
        required_analytics_params = ['metrics_history_size', 'prediction_window', 'confidence_threshold']
        for param in required_analytics_params:
            if param not in analytics_config:
                logger.error(f"Отсутствует обязательный параметр аналитики: {param}")
                return False
                
        # Проверяем параметры логирования
        logging_config = config['logging']
        required_logging_params = ['level', 'file_path', 'max_file_size_mb', 'backup_count']
        for param in required_logging_params:
            if param not in logging_config:
                logger.error(f"Отсутствует обязательный параметр логирования: {param}")
                return False
                
        # Проверяем параметры кэширования
        cache_config = config['cache']
        required_cache_params = ['enabled', 'ttl_seconds', 'max_size_mb']
        for param in required_cache_params:
            if param not in cache_config:
                logger.error(f"Отсутствует обязательный параметр кэширования: {param}")
                return False
                
        return True
        
    except Exception as e:
        logger.error(f"Ошибка при валидации конфигурации: {str(e)}")
        return False 