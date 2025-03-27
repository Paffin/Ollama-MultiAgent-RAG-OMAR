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
from prompts import (
    PLANNER_PROMPT,
    EXECUTOR_PROMPT,
    CRITIC_PROMPT,
    PRAISE_PROMPT,
    ARBITER_PROMPT
)

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

def log_chain(agent_name: str, step_type: str, content: str):
    st.session_state.chain_trace.append({
        "agent": agent_name,
        "type": step_type,
        "content": content
    })

def display_chain_trace():
    for i, record in enumerate(st.session_state.chain_trace):
        st.markdown(f"**Step {i+1} – {record['agent']} [{record['type']}]**")
        st.code(record["content"], language="markdown")

def detect_language(user_query: str) -> str:
    cyr_count = len(re.findall(r'[а-яА-ЯёЁ]', user_query))
    return "ru" if cyr_count >= 3 else "en"

def main():
    st.title("MultiAgent System with Browser & DuckDuckGo")

    init_session_state()
    client = st.session_state.ollama_client

    # --- Настройка агентов ---
    with st.expander("Настройка агентов"):
        models = client.list_models()
        if not models:
            st.warning("Не удалось получить список моделей Ollama.")
            models = ["unknown"]

        model_planner = st.selectbox("Planner Model", models, index=0)
        model_executor = st.selectbox("Executor Model", models, index=0)
        model_critic = st.selectbox("Critic Model", models, index=0)
        model_praise = st.selectbox("Praise Model", models, index=0)
        model_arbiter = st.selectbox("Arbiter Model", models, index=0)

        sp_planner = st.text_area("PlannerAgent Prompt", PLANNER_PROMPT, height=120)
        sp_executor = st.text_area("ExecutorAgent Prompt", EXECUTOR_PROMPT, height=120)
        sp_critic = st.text_area("CriticAgent Prompt", CRITIC_PROMPT, height=120)
        sp_praise = st.text_area("PraiseAgent Prompt", PRAISE_PROMPT, height=120)
        sp_arbiter = st.text_area("ArbiterAgent Prompt", ARBITER_PROMPT, height=120)

        if st.button("Инициализировать агентов"):
            from agents import (
                PlannerAgent,
                ExecutorAgent,
                CriticAgent,
                PraiseAgent,
                ArbiterAgent
            )
            st.session_state.planner_agent = PlannerAgent(
                name="Planner",
                model_name=model_planner,
                client=client
            )
            st.session_state.executor_agent = ExecutorAgent(
                name="Executor",
                model_name=model_executor,
                client=client
            )
            st.session_state.critic_agent = CriticAgent(
                name="Critic",
                model_name=model_critic,
                client=client
            )
            st.session_state.praise_agent = PraiseAgent(
                name="Praise",
                model_name=model_praise,
                client=client
            )
            st.session_state.arbiter_agent = ArbiterAgent(
                name="Arbiter",
                model_name=model_arbiter,
                client=client
            )
            st.session_state.chain_trace = []
            st.session_state.final_answer = ""
            st.success("Готово: агенты переинициализированы.")

    # --- Загрузка документов (RAG) ---
    with st.expander("Загрузка документов (RAG)"):
        up_files = st.file_uploader("Выберите TXT-файлы", accept_multiple_files=True)
        if up_files:
            for f in up_files:
                text = f.read().decode("utf-8", errors="ignore")
                st.session_state.vector_store.add_documents([text])
            st.success("Документы добавлены в FAISS.")

    # --- Параметры Ollama ---
    st.sidebar.title("Параметры Ollama")
    temperature = st.sidebar.slider("temperature", 0.0, 2.0, 0.8, 0.1)
    top_p = st.sidebar.slider("top_p", 0.0, 1.0, 0.9, 0.05)
    presence_penalty = st.sidebar.slider("presence_penalty", 0.0, 2.0, 0.0, 0.1)
    frequency_penalty = st.sidebar.slider("frequency_penalty", 0.0, 2.0, 0.0, 0.1)
    num_ctx = st.sidebar.number_input("num_ctx", 512, 10000, 10000, 256)
    num_predict = st.sidebar.number_input("num_predict", 64, 4096, 512, 64)

    st.sidebar.markdown("### Кол-во итераций")
    max_iterations = st.sidebar.number_input("iterations", 1, 5, 2, 1)

    # --- Основной ввод ---
    user_query = st.text_input("Введите задачу/запрос:")

    if st.button("Запустить"):
        user_lang = detect_language(user_query)
        st.markdown(f"**Определён язык:** `{user_lang}`")

        # Получаем агентов
        planner = st.session_state.planner_agent
        executor = st.session_state.executor_agent
        critic = st.session_state.critic_agent
        praise = st.session_state.praise_agent
        arbiter = st.session_state.arbiter_agent
        if not all([planner, executor, critic, praise, arbiter]):
            st.error("Агенты не инициализированы. Нажмите 'Инициализировать агентов'.")
            return

        st.session_state.chain_trace = []
        st.session_state.final_answer = ""

        # Параметры Ollama
        ollama_opts = dict(
            temperature=temperature,
            top_p=top_p,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            num_ctx=num_ctx,
            num_predict=num_predict
        )

        # Шаг 1: PlannerAgent
        st.markdown("### Шаг 1: PlannerAgent")
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
            placeholder_plan.code(plan_text, language="markdown")
            time.sleep(0.02)
        log_chain("PlannerAgent", "RESPONSE", plan_text)
        st.markdown("**Инструкция от Planner:**")
        st.code(plan_text, language="markdown")

        current_instruction = plan_text

        # Итерации
        for iteration in range(1, max_iterations + 1):
            st.markdown(f"## Итерация {iteration}")

            # ExecutorAgent
            st.markdown("### ExecutorAgent")
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
                placeholder_exec.code(exec_text, language="markdown")
            else:
                for ck in ex_gen:
                    exec_text += ck
                    placeholder_exec.code(exec_text, language="markdown")
                    time.sleep(0.02)

            log_chain("ExecutorAgent", "RESPONSE", exec_text)
            st.markdown("**Executor ответ:**")
            st.code(exec_text, language="markdown")

            # CriticAgent
            st.markdown("### CriticAgent")
            cr_text = ""
            log_chain("CriticAgent", "PROMPT", exec_text)
            placeholder_cr = st.empty()
            cr_gen = critic.criticize(exec_text, stream=True, **ollama_opts)
            if isinstance(cr_gen, str):
                cr_text = cr_gen
                placeholder_cr.code(cr_text, language="markdown")
            else:
                for ck in cr_gen:
                    cr_text += ck
                    placeholder_cr.code(cr_text, language="markdown")
                    time.sleep(0.02)
            log_chain("CriticAgent", "RESPONSE", cr_text)
            st.markdown("**Критика:**")
            st.code(cr_text, language="markdown")

            # PraiseAgent
            st.markdown("### PraiseAgent")
            pr_text = ""
            log_chain("PraiseAgent", "PROMPT", exec_text)
            placeholder_pr = st.empty()
            pr_gen = praise.praise(exec_text, stream=True, **ollama_opts)
            if isinstance(pr_gen, str):
                pr_text = pr_gen
                placeholder_pr.code(pr_text, language="markdown")
            else:
                for ck in pr_gen:
                    pr_text += ck
                    placeholder_pr.code(pr_text, language="markdown")
                    time.sleep(0.02)
            log_chain("PraiseAgent", "RESPONSE", pr_text)
            st.markdown("**Похвала:**")
            st.code(pr_text, language="markdown")

            # ArbiterAgent (ReworkInstruction)
            if iteration < max_iterations:
                st.markdown("### ArbiterAgent (Rework Instruction)")
                arb_text = ""
                arb_prompt = f"Executor: {exec_text}\nCritic: {cr_text}\nPraise: {pr_text}"
                log_chain("ArbiterAgent", "PROMPT", arb_prompt)
                placeholder_arb = st.empty()
                arb_gen = arbiter.produce_rework_instruction(exec_text, cr_text, pr_text, stream=True, **ollama_opts)
                if isinstance(arb_gen, str):
                    arb_text = arb_gen
                    placeholder_arb.code(arb_text, language="markdown")
                else:
                    for ck in arb_gen:
                        arb_text += ck
                        placeholder_arb.code(arb_text, language="markdown")
                        time.sleep(0.02)
                log_chain("ArbiterAgent", "RESPONSE", arb_text)
                st.markdown("**Rework Instruction:**")
                st.code(arb_text, language="markdown")
                current_instruction = arb_text
            else:
                st.session_state.final_answer = exec_text

        st.success("Итерации завершены.")

        st.markdown("## Итоговый ответ (ExecutorAgent)")
        if st.session_state.final_answer:
            st.markdown(
                """
                <div style="border:2px solid #2196F3; padding:10px; border-radius:5px; background-color:#E3F2FD;">
                <h4 style="color:#2196F3; margin-top:0;">Final Answer</h4>
                """,
                unsafe_allow_html=True
            )
            st.markdown(st.session_state.final_answer, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.warning("Final answer отсутствует.")

    st.markdown("## Chain-of-thought Trace")
    if st.session_state.chain_trace:
        display_chain_trace()
    else:
        st.info("Пока нет шагов в логе.")

if __name__ == "__main__":
    main()
