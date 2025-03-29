# streamlit_app.py

import sys
import asyncio

# 1) Устанавливаем политику цикла событий ProactorEventLoopPolicy
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# 2) Подключаем nest_asyncio
import nest_asyncio
nest_asyncio.apply()

import streamlit as st
import time
import re

from ollama_client import OllamaClient
from agents import (
    PlannerAgent,
    ExecutorAgent,
    CriticAgent,
    PraiseAgent,
    ArbiterAgent
)
from rag_db import SimpleVectorStore
from system_prompts import (
    PLANNER_PROMPT,
    EXECUTOR_PROMPT,
    CRITIC_PROMPT,
    PRAISE_PROMPT,
    ARBITER_PROMPT
)
from settings import Settings

# Настройка страницы
st.set_page_config(
    page_title="MultiAgent System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Стили CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        margin-top: 1rem;
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 4px;
        cursor: pointer;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .agent-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .status-badge {
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    .status-active {
        background-color: #e3f2fd;
        color: #1976d2;
    }
    .status-completed {
        background-color: #e8f5e9;
        color: #2e7d32;
    }
    .status-error {
        background-color: #ffebee;
        color: #c62828;
    }
    .chain-trace {
        background-color: #fff;
        border-radius: 8px;
        padding: 1rem;
        margin-top: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

def init_session_state():
    if "ollama_client" not in st.session_state:
        st.session_state.ollama_client = OllamaClient("http://localhost:11434")
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = SimpleVectorStore()
    if "planner_agent" not in st.session_state:
        st.session_state.planner_agent = None
    if "executor_agent" not in st.session_state:
        st.session_state.executor_agent = None
    if "critic_agent" not in st.session_state:
        st.session_state.critic_agent = None
    if "praise_agent" not in st.session_state:
        st.session_state.praise_agent = None
    if "arbiter_agent" not in st.session_state:
        st.session_state.arbiter_agent = None
    if "chain_trace" not in st.session_state:
        st.session_state.chain_trace = []
    if "final_answer" not in st.session_state:
        st.session_state.final_answer = ""
    if "system_prompts" not in st.session_state:
        st.session_state.system_prompts = {
            "planner": PLANNER_PROMPT,
            "executor": EXECUTOR_PROMPT,
            "critic": CRITIC_PROMPT,
            "praise": PRAISE_PROMPT,
            "arbiter": ARBITER_PROMPT
        }
    if "settings" not in st.session_state:
        st.session_state.settings = Settings()

def log_chain(agent_name: str, step_type: str, content: str):
    st.session_state.chain_trace.append({
        "agent": agent_name,
        "type": step_type,
        "content": content,
        "timestamp": time.strftime("%H:%M:%S")
    })

def display_chain_trace():
    st.markdown("### 📋 История выполнения")
    for i, record in enumerate(st.session_state.chain_trace):
        with st.expander(f"Шаг {i+1} – {record['agent']} [{record['type']}] ({record['timestamp']})"):
            st.markdown(record["content"])

def display_agent_status(agent):
    if agent and agent.state:
        status_class = {
            "idle": "status-active",
            "planning": "status-active",
            "executing": "status-active",
            "criticizing": "status-active",
            "praising": "status-active",
            "arbitrating": "status-active",
            "completed": "status-completed",
            "error": "status-error"
        }.get(agent.state.status.value, "status-active")
        
        st.markdown(f"""
            <div class="agent-card">
                <h4>{agent.name}</h4>
                <span class="status-badge {status_class}">{agent.state.status.value}</span>
                <p>{agent.state.current_task}</p>
                {f'<p style="color: #c62828;">Ошибка: {agent.state.error}</p>' if agent.state.error else ''}
            </div>
        """, unsafe_allow_html=True)

def detect_language(user_query: str) -> str:
    cyr_count = len(re.findall(r'[а-яА-ЯёЁ]', user_query))
    return "ru" if cyr_count >= 3 else "en"

def main():
    st.title("🤖 MultiAgent System")
    st.markdown("""
        <div style="background-color: #e3f2fd; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
            <h4 style="margin: 0;">Система мультиагентов с поддержкой браузера и DuckDuckGo</h4>
            <p style="margin: 0.5rem 0 0;">Используйте эту систему для выполнения сложных задач с помощью различных специализированных агентов.</p>
        </div>
    """, unsafe_allow_html=True)

    init_session_state()
    client = st.session_state.ollama_client
    settings = st.session_state.settings

    # --- Настройка агентов ---
    with st.expander("⚙️ Настройка агентов", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Модели")
            models = client.list_models()
            if not models:
                st.warning("⚠️ Не удалось получить список моделей Ollama.")
                models = ["unknown"]

            model_planner = st.selectbox(
                "Planner Model",
                models,
                index=models.index(settings.get_model("planner")) if settings.get_model("planner") in models else 0
            )
            model_executor = st.selectbox(
                "Executor Model",
                models,
                index=models.index(settings.get_model("executor")) if settings.get_model("executor") in models else 0
            )
            model_critic = st.selectbox(
                "Critic Model",
                models,
                index=models.index(settings.get_model("critic")) if settings.get_model("critic") in models else 0
            )
            model_praise = st.selectbox(
                "Praise Model",
                models,
                index=models.index(settings.get_model("praise")) if settings.get_model("praise") in models else 0
            )
            model_arbiter = st.selectbox(
                "Arbiter Model",
                models,
                index=models.index(settings.get_model("arbiter")) if settings.get_model("arbiter") in models else 0
            )

        with col2:
            st.markdown("### Системные промпты")
            sp_planner = st.text_area("PlannerAgent Prompt", st.session_state.system_prompts["planner"], height=120)
            sp_executor = st.text_area("ExecutorAgent Prompt", st.session_state.system_prompts["executor"], height=120)
            sp_critic = st.text_area("CriticAgent Prompt", st.session_state.system_prompts["critic"], height=120)
            sp_praise = st.text_area("PraiseAgent Prompt", st.session_state.system_prompts["praise"], height=120)
            sp_arbiter = st.text_area("ArbiterAgent Prompt", st.session_state.system_prompts["arbiter"], height=120)

        if st.button("🔄 Инициализировать агентов"):
            with st.spinner("Инициализация агентов..."):
                # Сохраняем выбранные модели
                settings.set_model("planner", model_planner)
                settings.set_model("executor", model_executor)
                settings.set_model("critic", model_critic)
                settings.set_model("praise", model_praise)
                settings.set_model("arbiter", model_arbiter)
                
                # Обновляем промпты в session_state
                st.session_state.system_prompts.update({
                    "planner": sp_planner,
                    "executor": sp_executor,
                    "critic": sp_critic,
                    "praise": sp_praise,
                    "arbiter": sp_arbiter
                })
                
                st.session_state.planner_agent = PlannerAgent(
                    name="Planner",
                    system_prompt=sp_planner,
                    model_name=model_planner,
                    client=client
                )
                st.session_state.executor_agent = ExecutorAgent(
                    name="Executor",
                    system_prompt=sp_executor,
                    model_name=model_executor,
                    client=client
                )
                st.session_state.critic_agent = CriticAgent(
                    name="Critic",
                    system_prompt=sp_critic,
                    model_name=model_critic,
                    client=client
                )
                st.session_state.praise_agent = PraiseAgent(
                    name="Praise",
                    system_prompt=sp_praise,
                    model_name=model_praise,
                    client=client
                )
                st.session_state.arbiter_agent = ArbiterAgent(
                    name="Arbiter",
                    system_prompt=sp_arbiter,
                    model_name=model_arbiter,
                    client=client
                )
                st.session_state.chain_trace = []
                st.session_state.final_answer = ""
                st.success("✅ Агенты успешно инициализированы!")

    # --- Загрузка документов (RAG) ---
    with st.expander("📚 Загрузка документов (RAG)", expanded=False):
        up_files = st.file_uploader("Выберите TXT-файлы", accept_multiple_files=True)
        if up_files:
            with st.spinner("Загрузка документов..."):
                for f in up_files:
                    text = f.read().decode("utf-8", errors="ignore")
                    st.session_state.vector_store.add_documents([text])
                st.success(f"✅ Загружено {len(up_files)} документов")

    # --- Параметры Ollama ---
    with st.sidebar:
        st.title("⚙️ Параметры Ollama")
        ollama_settings = settings.get_ollama_settings()
        
        st.markdown("### Основные параметры")
        temperature = st.slider(
            "Temperature",
            0.0, 2.0,
            ollama_settings["temperature"],
            0.1
        )
        top_p = st.slider(
            "Top P",
            0.0, 1.0,
            ollama_settings["top_p"],
            0.05
        )
        
        st.markdown("### Параметры штрафов")
        presence_penalty = st.slider(
            "Presence Penalty",
            0.0, 2.0,
            ollama_settings["presence_penalty"],
            0.1
        )
        frequency_penalty = st.slider(
            "Frequency Penalty",
            0.0, 2.0,
            ollama_settings["frequency_penalty"],
            0.1
        )
        
        st.markdown("### Параметры контекста")
        num_ctx = st.number_input(
            "Context Size",
            512, 8192,
            ollama_settings["num_ctx"],
            256
        )
        num_predict = st.number_input(
            "Max Tokens",
            64, 4096,
            ollama_settings["num_predict"],
            64
        )

        st.markdown("### Итерации")
        max_iterations = st.number_input(
            "Количество итераций",
            1, 5,
            settings.get_max_iterations(),
            1
        )

        # Сохраняем обновленные настройки
        settings.update_ollama_settings({
            "temperature": temperature,
            "top_p": top_p,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
            "num_ctx": num_ctx,
            "num_predict": num_predict
        })
        settings.set_max_iterations(max_iterations)

    # --- Основной ввод ---
    st.markdown("### 💬 Введите задачу/запрос")
    user_query = st.text_area("", height=100, placeholder="Опишите вашу задачу здесь...")

    if st.button("🚀 Запустить"):
        if not user_query:
            st.warning("⚠️ Пожалуйста, введите запрос")
            return

        # Получаем агентов
        planner = st.session_state.planner_agent
        executor = st.session_state.executor_agent
        critic = st.session_state.critic_agent
        praise = st.session_state.praise_agent
        arbiter = st.session_state.arbiter_agent
        
        if not all([planner, executor, critic, praise, arbiter]):
            st.error("❌ Агенты не инициализированы. Нажмите 'Инициализировать агентов'.")
            return

        st.session_state.chain_trace = []
        st.session_state.final_answer = ""

        # Параметры Ollama
        ollama_opts = settings.get_ollama_settings()

        # Отображение статуса агентов
        st.markdown("### 📊 Статус агентов")
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            display_agent_status(planner)
        with col2:
            display_agent_status(executor)
        with col3:
            display_agent_status(critic)
        with col4:
            display_agent_status(praise)
        with col5:
            display_agent_status(arbiter)

        # Шаг 1: PlannerAgent
        with st.expander("📋 Шаг 1: PlannerAgent", expanded=True):
            plan_text = ""
            log_chain("PlannerAgent", "PROMPT", user_query)
            plan_gen = planner.generate_instruction(
                user_query, 
                st.session_state.vector_store, 
                stream=True, 
                **ollama_opts
            )
            placeholder_plan = st.empty()
            for chunk in plan_gen:
                plan_text += chunk
                placeholder_plan.markdown(plan_text)
                time.sleep(0.02)
            log_chain("PlannerAgent", "RESPONSE", plan_text)

        current_instruction = plan_text

        # Итерации
        for iteration in range(1, max_iterations + 1):
            st.markdown(f"## 🔄 Итерация {iteration}")

            # ExecutorAgent
            with st.expander("⚡ ExecutorAgent", expanded=True):
                log_chain("ExecutorAgent", "PROMPT", current_instruction)
                exec_text = ""
                placeholder_exec = st.empty()
                ex_gen = executor.execute_instruction(
                    instruction=current_instruction,
                    vector_store=st.session_state.vector_store,
                    stream=True,
                    **ollama_opts
                )
                if isinstance(ex_gen, str):
                    exec_text = ex_gen
                    placeholder_exec.markdown(exec_text)
                else:
                    for ck in ex_gen:
                        exec_text += ck
                        placeholder_exec.markdown(exec_text)
                        time.sleep(0.02)
                log_chain("ExecutorAgent", "RESPONSE", exec_text)

            # CriticAgent
            with st.expander("🔍 CriticAgent", expanded=False):
                cr_text = ""
                log_chain("CriticAgent", "PROMPT", exec_text)
                placeholder_cr = st.empty()
                cr_gen = critic.criticize(exec_text, stream=True, **ollama_opts)
                if isinstance(cr_gen, str):
                    cr_text = cr_gen
                    placeholder_cr.markdown(cr_text)
                else:
                    for ck in cr_gen:
                        cr_text += ck
                        placeholder_cr.markdown(cr_text)
                        time.sleep(0.02)
                log_chain("CriticAgent", "RESPONSE", cr_text)

            # PraiseAgent
            with st.expander("🌟 PraiseAgent", expanded=False):
                pr_text = ""
                log_chain("PraiseAgent", "PROMPT", exec_text)
                placeholder_pr = st.empty()
                pr_gen = praise.praise(exec_text, stream=True, **ollama_opts)
                if isinstance(pr_gen, str):
                    pr_text = pr_gen
                    placeholder_pr.markdown(pr_text)
                else:
                    for ck in pr_gen:
                        pr_text += ck
                        placeholder_pr.markdown(pr_text)
                        time.sleep(0.02)
                log_chain("PraiseAgent", "RESPONSE", pr_text)

            # ArbiterAgent
            if iteration < max_iterations:
                with st.expander("⚖️ ArbiterAgent", expanded=False):
                    arb_text = ""
                    arb_prompt = f"Executor: {exec_text}\nCritic: {cr_text}\nPraise: {pr_text}"
                    log_chain("ArbiterAgent", "PROMPT", arb_prompt)
                    placeholder_arb = st.empty()
                    arb_gen = arbiter.produce_rework_instruction(exec_text, cr_text, pr_text, stream=True, **ollama_opts)
                    if isinstance(arb_gen, str):
                        arb_text = arb_gen
                        placeholder_arb.markdown(arb_text)
                    else:
                        for ck in arb_gen:
                            arb_text += ck
                            placeholder_arb.markdown(arb_text)
                            time.sleep(0.02)
                    log_chain("ArbiterAgent", "RESPONSE", arb_text)
                    current_instruction = arb_text
            else:
                st.session_state.final_answer = exec_text

        st.success("✅ Итерации завершены")

        # Итоговый ответ
        if st.session_state.final_answer:
            st.markdown("## 🎯 Итоговый ответ")
            st.markdown(
                f"""
                <div style="background-color: #e3f2fd; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
                    {st.session_state.final_answer}
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.warning("⚠️ Итоговый ответ отсутствует")

    # История выполнения
    if st.session_state.chain_trace:
        display_chain_trace()

if __name__ == "__main__":
    main()
