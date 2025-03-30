# Мультиагентная система на Python с использованием Ollama, FAISS и Streamlit

## Описание проекта

Данный проект реализует мультиагентную архитектуру, где несколько специализированных агентов (Planner, Executor, Critic, Praise, Arbiter) совместно решают поставленные задачи, комбинируя возможности языковых моделей и локального поиска. Основные особенности проекта:

- **Интеграция с Ollama** – взаимодействие через REST API для генерации ответов с поддержкой потокового вывода.
- **Векторная база с FAISS и SentenceTransformers** – поиск и агрегирование локальной информации (RAG).
- **Встроенные инструменты для работы с ОС** – выполнение системных команд, просмотр директорий, браузерные действия.
- **Поддержка визуального анализа** – анализ изображений через LLaVA, OCR, описание контента.
- **Многократные итерации до улучшения результата** – циклы типа Executor → Critic → Praise → Arbiter для доработки ответа.
- **Интерактивный веб-интерфейс на базе Streamlit** – удобное управление настройками агентов, параметрами генерации и загрузкой документов.

---

## Функциональные возможности

1. **Анализ запроса (PlannerAgent):**
   - Использует LLM для предварительного анализа пользовательского запроса
   - Определяет тип необходимых действий:
     - `ducksearch:` для интернет-поиска
     - `browser:` для браузерных действий
     - `visual:` для анализа изображений
     - Локальный поиск по векторной базе

2. **Выполнение инструкции (ExecutorAgent):**
   - **Браузерные действия:**
     - `url=<url>` - открытие страницы
     - `click=<selector>` - клик по элементу
     - `type=<selector>:<text>` - ввод текста
     - `screenshot=<path>` - создание скриншота
   - **Визуальный анализ:**
     - `analyze=<path>` - анализ изображения
     - `describe=<path>` - подробное описание
     - `ocr=<path>` - распознавание текста
   - **Системные команды:**
     - Безопасное выполнение с проверкой опасных паттернов
     - Работа с файлами и директориями
   - **Поиск информации:**
     - DuckDuckGo с таймаутами и обработкой ошибок
     - Векторный поиск по локальной базе

3. **Анализ качества ответа:**
   - **CriticAgent:**
     - Выявление ошибок и недостатков
     - Проверка релевантности
     - Анализ производительности
   - **PraiseAgent:**
     - Выделение сильных сторон
     - Оценка инноваций
     - Анализ эффективности

4. **Улучшение ответа (ArbiterAgent):**
   - Синтез критики и похвалы
   - Формирование инструкций по улучшению
   - Контроль итераций

5. **Интерактивный интерфейс:**
   - Настройка моделей для каждого агента
   - Управление параметрами генерации
   - Загрузка документов для RAG
   - Мониторинг статусов агентов
   - История выполнения

---

## Структура проекта

```
multiagent_system/
├── agents.py           # Реализация агентов с поддержкой всех типов действий
├── ollama_client.py    # Клиент Ollama с поддержкой потокового вывода
├── rag_db.py          # Векторная база с контролем размера и валидацией
├── streamlit_app.py    # Веб-интерфейс с расширенным мониторингом
├── windows_tools.py    # Инструменты для браузера и системных команд
├── requirements.txt    # Зависимости проекта
└── README.md          # Документация
```

---

## Установка и настройка

### 1. Установка Ollama

```bash
# Установка Ollama
curl https://ollama.ai/install.sh | sh

# Запуск сервера
ollama serve --port 11434

# Проверка работы
curl http://localhost:11434/api/tags
```

### 2. Установка зависимостей

```bash
# Создание виртуального окружения
python -m venv venv
source venv/bin/activate  # Linux/macOS
.\venv\Scripts\activate   # Windows

# Установка пакетов
pip install -r requirements.txt

# Установка Playwright
playwright install chromium
```

### 3. Настройка окружения

```bash
# Создание директорий для данных
mkdir -p data/images data/documents

# Установка переменных окружения (опционально)
export OLLAMA_HOST=http://localhost:11434
export PLAYWRIGHT_BROWSERS_PATH=0
```

---

## Использование

### Форматы команд

1. **Браузерные действия:**
```python
"browser:url=https://example.com;click=#button;type=#input:text;screenshot=shot.png"
```

2. **Визуальный анализ:**
```python
"visual:analyze=image.png;describe=photo.jpg;ocr=text.png"
```

3. **Поиск информации:**
```python
"ducksearch:ваш поисковый запрос"
```

### Примеры использования

1. **Анализ веб-страницы:**
```python
query = """
browser:url=https://example.com;
screenshot=page.png;
visual:analyze=page.png
"""
```

2. **Комплексный поиск:**
```python
query = """
ducksearch:Python multiagent systems;
browser:url={результат поиска};
screenshot=research.png;
visual:ocr=research.png
"""
```

3. **Работа с документами:**
```python
# Загрузка документов через интерфейс
st.file_uploader("Выберите файлы", accept_multiple_files=True)

# Поиск в загруженных документах
query = "найти информацию о мультиагентных системах"
```

---

## Безопасность

1. **Проверка команд:**
   - Блокировка опасных системных команд
   - Валидация путей файлов
   - Контроль перенаправлений

2. **Браузерная безопасность:**
   - Использование headless режима
   - Таймауты для всех операций
   - Проверка SSL-сертификатов

3. **Контроль данных:**
   - Ограничение размера документов
   - Валидация типов файлов
   - Безопасное хранение API ключей

---

## Расширение функционала

### 1. Добавление новых агентов

```python
class NewAgent(BaseAgent):
    def __init__(self, name: str, system_prompt: str, model_name: str, client: OllamaClient):
        super().__init__(name, system_prompt, model_name, client)
        
    def process(self, input_data: str, **kwargs) -> str:
        self.update_state(AgentStatus.PROCESSING, "Обработка данных")
        # Ваша логика
        return result
```

### 2. Новые браузерные действия

```python
def _perform_browser_actions(self, actions: str) -> str:
    # Добавьте новые действия в парсер
    if action.startswith('new_action='):
        # Ваша логика
        pass
```

### 3. Расширение визуального анализа

```python
def _perform_visual_actions(self, actions: str) -> str:
    # Добавьте новые типы анализа
    if action.startswith('new_analysis='):
        # Ваша логика
        pass
```

---

## Известные ограничения

1. **Ollama API:**
   - Ограничения на размер контекста
   - Возможные задержки при первой загрузке модели

2. **Браузерная автоматизация:**
   - Некоторые сайты могут блокировать автоматизацию
   - Ограничения на скорость действий

3. **Визуальный анализ:**
   - Зависимость от качества изображений
   - Ограничения моделей LLaVA

---

## Решение проблем

1. **Ошибки браузера:**
   ```python
   # Увеличьте таймауты
   browser = PlaywrightBrowser(timeout=60000)
   ```

2. **Проблемы с памятью:**
   ```python
   # Ограничьте размер векторного хранилища
   vector_store.max_documents = 500
   vector_store.max_document_size = 500000
   ```

3. **Медленная работа:**
   ```python
   # Используйте более легкие модели
   model_name = "gemma:2b"
   ```

---

## Дорожная карта

1. **Ближайшие улучшения:**
   - Поддержка параллельного выполнения агентов
   - Кэширование результатов поиска
   - Улучшенная обработка ошибок

2. **Среднесрочные планы:**
   - Интеграция новых моделей
   - Расширение визуальных возможностей
   - Улучшение UI/UX

3. **Долгосрочные цели:**
   - Распределенное выполнение
   - Поддержка новых типов данных
   - API для внешней интеграции

---

## Лицензия

MIT License

## Авторы

- **Максим Киктев** - *Основной разработчик*

## Благодарности

- Команде Ollama за отличный проект
- Сообществу Streamlit за инструменты
- Всем контрибьюторам за помощь