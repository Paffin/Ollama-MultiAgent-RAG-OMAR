from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass
class OllamaModel:
    """Класс для работы с моделями Ollama"""
    
    name: str
    size: Optional[int] = None
    digest: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OllamaModel':
        """
        Создание экземпляра из словаря
        
        Args:
            data: Данные модели
            
        Returns:
            Экземпляр OllamaModel
        """
        return cls(
            name=data.get('name', ''),
            size=data.get('size'),
            digest=data.get('digest'),
            details=data.get('details')
        )
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Преобразование в словарь
        
        Returns:
            Словарь с данными модели
        """
        return {
            'name': self.name,
            'size': self.size,
            'digest': self.digest,
            'details': self.details
        } 