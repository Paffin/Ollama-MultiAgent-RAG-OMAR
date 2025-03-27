from typing import Dict, Any, List, Generator
from utils.logger import Logger

class StreamingProcessor:
    """Процессор потоковой обработки данных"""
    
    def __init__(self):
        self.logger = Logger()
        
    def process_stream(self, data_stream: Generator[Dict[str, Any], None, None]) -> Generator[Dict[str, Any], None, None]:
        """
        Обработка потока данных
        
        Args:
            data_stream: Поток входных данных
            
        Returns:
            Поток обработанных данных
        """
        try:
            for data in data_stream:
                processed_data = self._process_item(data)
                yield processed_data
                self.logger.info("Элемент потока обработан")
        except Exception as e:
            self.logger.error(f"Ошибка при обработке потока данных: {str(e)}")
            raise
            
    def _process_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Обработка отдельного элемента
        
        Args:
            item: Элемент для обработки
            
        Returns:
            Обработанный элемент
        """
        try:
            # Здесь должна быть логика обработки элемента
            processed_item = item.copy()
            return processed_item
        except Exception as e:
            self.logger.error(f"Ошибка при обработке элемента: {str(e)}")
            raise
            
    def batch_process(self, items: List[Dict[str, Any]], batch_size: int = 100) -> List[Dict[str, Any]]:
        """
        Пакетная обработка данных
        
        Args:
            items: Список элементов
            batch_size: Размер пакета
            
        Returns:
            Список обработанных элементов
        """
        try:
            processed_items = []
            for i in range(0, len(items), batch_size):
                batch = items[i:i + batch_size]
                processed_batch = [self._process_item(item) for item in batch]
                processed_items.extend(processed_batch)
                self.logger.info(f"Обработан пакет из {len(batch)} элементов")
            return processed_items
        except Exception as e:
            self.logger.error(f"Ошибка при пакетной обработке: {str(e)}")
            raise 