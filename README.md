# Мультиагентная система на Python со Streamlit и Ollama

## Описание

Данный проект реализует **мультиагентную архитектуру**, в которой несколько агентов (Planner, Executor, Critic, Praise, Arbiter) совместно решают задачу, используя:

1. **Оффлайн-модели Ollama** (через эндпоинты `/api/tags` и `/api/generate`).  
2. **Векторную базу** (FAISS + SentenceTransformers) для локального поиска документов (RAG).  
3. **Инструменты для ОС**: `cmd:`, `ls:`, а также `search:` (векторный поиск).  
4. **Многократные итерации** (по желанию) с цепочкой Critic/Praise/Arbiter → Executor.  
5. **Вывод цепочек (chain-of-thought / chain-of-messages)** в стиле [camel-ai/owl](https://github.com/camel-ai/owl).  
6. **Streamlit Web UI** для настройки и работы с системой (выбор моделей, системных промптов, параметров Ollama).  

Результат: при запуске приложения вы получите локальный веб-интерфейс, где можно задать запрос, наблюдать пошаговую работу агентов и видеть итоговый результат.

---

## 1. Установка и требования

1. **Установите Ollama**  
   - Скачать / установить [Ollama](https://github.com/jmorganca/ollama).  
   - Запустите Ollama сервер:
     ```bash
     ollama serve --port 11434
     ```
   - Проверьте: 
     ```bash
     curl http://localhost:11434/api/tags
     ```
     должно возвращать JSON со списком локальных моделей.

2. **Клонируйте проект** (либо скопируйте файлы) в локальную папку:
   ```bash
   git clone https://github.com/youruser/your-repo.git
   cd your-repo
   ```

3. **Создайте и активируйте виртуальное окружение** (рекомендуется):
   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/mac
   # или
   .\venv\Scripts\activate    # Windows
   ```

4. **Установите зависимости**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

   Если возникают ошибки при установке `faiss-cpu` на Windows, можно найти сборку `faiss-cpu` в Conda или PyPi, совместимую с вашей версией Python.

5. **Проверьте, что Ollama слушает на порту 11434**:
   ```bash
   # в другом терминале/окне
   ollama serve --port 11434
   ```

---

## 2. Структура проекта

```
my_multiagent_app/
├── requirements.txt      # список зависимостей (Streamlit, requests, faiss-cpu, sentence-transformers)
├── ollama_client.py      # клиент Ollama (list_models + generate)
├── windows_tools.py      # инструменты для работы с ОС (cmd, ls)
├── rag_db.py             # векторная БД (FAISS + SentenceTransformers)
├── agents.py             # 5 агентов: Planner, Executor, Critic, Praise, Arbiter
└── streamlit_app.py      # главный файл Web UI (Streamlit), логика итераций, chain-of-thought
```

**Основные файлы**:

- **`streamlit_app.py`**: запускается через `streamlit run streamlit_app.py`, содержит интерфейс.  
- **`agents.py`**: реализация классов-агентов (PlannerAgent, ExecutorAgent и т.д.).  
- **`ollama_client.py`**: взаимодействие с Ollama по REST API.  
- **`windows_tools.py`**: инструменты для запуска команд ОС.  
- **`rag_db.py`**: простая векторная база данных (FAISS) + SentenceTransformers.

---

## 3. Запуск

После клонирования и установки зависимостей:

1. Убедитесь, что Ollama запущен:
   ```bash
   ollama serve --port=11434
   ```
2. Запустите Streamlit-приложение:
   ```bash
   streamlit run streamlit_app.py
   ```
3. Откройте в браузере URL, который покажет Streamlit (обычно `http://localhost:8501`).

---

## 4. Использование

На **главной странице** Streamlit вы увидите несколько вкладок/блоков:

1. **Настройки агентов**:
   - **Выбор моделей** Ollama для каждого агента: Planner, Executor, Critic, Praise, Arbiter.  
   - **Системные промпты**: можно настроить под свою задачу (например, «Ты - Planner, твоя роль...», «Ты - Critic, ищи слабые стороны...» и т.д.).
   - Кнопка «Инициализировать агентов» перезапускает их конфигурацию.

2. **Загрузка документов (RAG)**:
   - Можно загрузить текстовые файлы. Они будут добавлены в FAISS-индекс, и ExecutorAgent сможет делать `search: <запрос>` для поиска.

3. **Параметры Ollama** (sidebar):
   - `temperature`, `top_p`, `presence_penalty`, `frequency_penalty`, `num_ctx`, `num_predict` (max tokens).  
   - Количество итераций `max_iterations` (1...5), указывает, сколько раз повторять цикл Executor→Critic→Praise→Arbiter→Executor.

4. **Поле ввода**: «Введите ваш запрос / задачу».  
   - При нажатии «Запустить» начинается цепочка:
     1. **PlannerAgent** генерирует инструкцию → ExecutorAgent выполняет.  
     2. **CriticAgent** и **PraiseAgent** дают свои комментарии → ArbiterAgent формирует «Rework Instruction».  
     3. **ExecutorAgent** снова пытается улучшить ответ.  
     4. Повторяется заданное `max_iterations` число раз.  
     5. Итоговый ответ (последний ответ ExecutorAgent) отображается в финальном окошке.

5. **Chain-of-thought trace**:
   - Внизу страницы показывается пошаговый лог (step by step) всех PROMPT/RESPONSE для агентов, в стиле [camel-ai/owl](https://github.com/camel-ai/owl).

### Дополнительная доработка (Arbiter→Executor)

После завершения `max_iterations` вы увидите «Финальный ответ» и кнопку (по желанию) вызвать ещё один дополнительный шаг (Arbiter->Executor). Это позволяет вручную улучшить ответ, если нужно.

---

## 5. Пример сценария использования

1. **Старт**: вы запускаете `streamlit run streamlit_app.py` → переходите в браузер (localhost:8501).  
2. **Настройки**:
   - Выбираете модели Ollama (например, `qwen2.5:32b` для Planner, `llama3.1:8b` для Executor и т.д.).  
   - Редактируете системные промпты (если нужно). Жмёте «Инициализировать агентов».  
3. **Загружаете** TXT-файлы, где может быть какая-то информация (для `search:`).  
4. **Вводите** запрос/задачу (например: «Напиши план проекта...»). Жмёте «Запустить».  
   - **PlannerAgent** выводит инструкцию (стримом).  
   - **ExecutorAgent** формирует ответ (стримом).  
   - **CriticAgent** даёт критику, **PraiseAgent** — похвалу, **ArbiterAgent** создаёт новую инструкцию,  
   - **ExecutorAgent** снова уточняет/улучшает ответ... — пока не исчерпаны итерации.  
5. **Финал**: в «Итоговом ответе» вы видите последний вариант. Если нужно, нажимаете кнопку «Сделать дополнительный Rework» для очередной доработки.

---

## 6. Возможные расширения

- **Дополнительные инструменты**: добавьте `windows_tools.py` функции для работы с PowerShell, браузером, локальными файлами PDF (parse PDF), и т.д.  
- **Итерации до условий**: вместо жёсткого `max_iterations`, можно останавливать, когда CriticAgent не находит критических замечаний, и/или когда пользователь подтверждает, что ответ устроил.  
- **Сложные сценарии**: больше агентов (агент-программист, агент-тестировщик, и т.д.).  
- **Сохранение логов**: писать chain-of-thought в базу (SQLite, Mongo) для дальнейшего анализа.  
- **Без Streamlit**: интегрируйте в любую другую web-фреймворк (Flask, FastAPI) или CLI.

---

## 7. Частые проблемы

1. **Ошибка 404 Not Found** на `/api/generate`:  
   - Убедитесь, что Ollama действительно отдаёт `/api/generate` (новые версии Ollama), а не `/v1/generate`.  
2. **`streamlit.errors.StreamlitAPIException: Invalid height 60px ... must be >= 68`**:  
   - Повышайте `height` в `st.text_area`, например, `height=80`.  
3. **Установка `faiss-cpu`** под Windows может быть сложной:  
   - Используйте PyPi [faiss-cpu](https://pypi.org/project/faiss-cpu/) подходящей версии. Или Anaconda.  
4. **Performance**: при больших моделях Ollama (50+ ГБ) могут быть задержки. Уменьшите `num_predict`, `temp`, или используйте более компактные модели.

---

## 8. Лицензия и авторы

- **Код** в данном репо (мультиагентная логика, Streamlit-пример) распространяется свободно  
- **Ollama**: см. лицензию в [репозитории Ollama](https://github.com/jmorganca/ollama).  
- **FAISS**: см. лицензию Facebook Research.  
- **SentenceTransformers**: Apache 2.0 (https://github.com/UKPLab/sentence-transformers).  

Авторы / contributors: Maxim Kiktev.  

**Удачи в использовании мультиагентной системы и локальных моделей Ollama!**