# streamlit_app.py

import streamlit as st
import time

from ollama_client import OllamaClient
from agents import (
    PlannerAgent,
    ExecutorAgent,
    CriticAgent,
    PraiseAgent,
    ArbiterAgent
)
from rag_db import SimpleVectorStore


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
        st.session_state.chain_trace = []  # пошаговый лог
    if "final_answer" not in st.session_state:
        st.session_state.final_answer = ""

def log_chain(agent_name: str, step_type: str, content: str):
    """
    step_type может быть PROMPT/RESPONSE или TOOL
    """
    st.session_state.chain_trace.append({
        "agent": agent_name,
        "type": step_type,
        "content": content
    })

def display_chain_trace():
    """
    Отображаем лог chain_trace шаг за шагом
    """
    for i, record in enumerate(st.session_state.chain_trace):
        step_no = i + 1
        agent = record["agent"]
        tp = record["type"]
        msg = record["content"]
        st.markdown(f"**Step {step_no}** - {agent} [{tp}]:")
        st.write(f"```\n{msg}\n```")


def main():
    st.title("Мультиагентная система (многократные итерации + Chain-of-Thought)")

    init_session_state()
    client = st.session_state.ollama_client

    with st.expander("Настройки агентов"):
        models = client.list_models()
        if not models:
            st.warning("Не удалось получить список моделей Ollama.")
            models = ["unknown"]

        model_planner = st.selectbox("PlannerAgent Model", models, index=0)
        model_executor = st.selectbox("ExecutorAgent Model", models, index=0)
        model_critic = st.selectbox("CriticAgent Model", models, index=0)
        model_praise = st.selectbox("PraiseAgent Model", models, index=0)
        model_arbiter = st.selectbox("ArbiterAgent Model", models, index=0)

        def_planner = "Ты - Агент-Планировщик..."
        def_executor = "Ты - Агент-Исполнитель..."
        def_critic = "Ты - Агент-Критик..."
        def_praise = "Ты - Агент-Похвала..."
        def_arbiter = "Ты - Агент-Арбитр. Учитывай замечания, не всё затирай..."

        sp_planner = st.text_area("PlannerAgent System Prompt:", def_planner, height=80)
        sp_executor = st.text_area("ExecutorAgent System Prompt:", def_executor, height=80)
        sp_critic   = st.text_area("CriticAgent System Prompt:", def_critic,   height=80)
        sp_praise   = st.text_area("PraiseAgent System Prompt:", def_praise,   height=80)
        sp_arbiter  = st.text_area("ArbiterAgent System Prompt:", def_arbiter, height=80)


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
            st.success("Агенты созданы / обновлены.")

    with st.expander("Загрузка документов (RAG)"):
        up_files = st.file_uploader("Загрузите текстовые файлы", accept_multiple_files=True)
        if up_files:
            for f in up_files:
                text = f.read().decode("utf-8", errors="ignore")
                st.session_state.vector_store.add_documents([text])
            st.success("Документы добавлены в векторный индекс.")

    st.sidebar.title("Параметры Ollama")
    temperature = st.sidebar.slider("temperature", 0.0, 2.0, 0.8, 0.1)
    top_p = st.sidebar.slider("top_p", 0.0, 1.0, 0.9, 0.05)
    presence_penalty = st.sidebar.slider("presence_penalty", 0.0, 2.0, 0.0, 0.1)
    frequency_penalty = st.sidebar.slider("frequency_penalty", 0.0, 2.0, 0.0, 0.1)
    num_ctx = st.sidebar.number_input("num_ctx", min_value=512, max_value=8192, value=2048, step=256)
    num_predict = st.sidebar.number_input("num_predict", min_value=64, max_value=4096, value=512, step=64)

    max_iterations = st.sidebar.number_input("Кол-во итераций", min_value=1, max_value=5, value=2, step=1)
    st.sidebar.write("В каждой итерации: Executor -> Critic+Praise -> Arbiter -> Executor")

    user_query = st.text_input("Введите ваш запрос / задачу:")

    if st.button("Запустить"):
        planner = st.session_state.planner_agent
        executor = st.session_state.executor_agent
        critic = st.session_state.critic_agent
        praise = st.session_state.praise_agent
        arbiter = st.session_state.arbiter_agent

        if not all([planner, executor, critic, praise, arbiter]):
            st.error("Сначала инициализируйте агентов!")
            return

        st.session_state.chain_trace = []
        st.session_state.final_answer = ""

        ollama_opts = dict(
            temperature=temperature,
            top_p=top_p,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            num_ctx=num_ctx,
            num_predict=num_predict
        )

        # 1) PlannerAgent => initial instruction
        st.markdown("### Шаг 1: PlannerAgent генерирует инструкцию")
        plan_text = ""
        plan_placeholder = st.empty()

        # Логируем PROMPT
        log_chain("PlannerAgent", "PROMPT", user_query)
        gen_plan = planner.generate_instruction(user_query, stream=True, **ollama_opts)
        for chunk in gen_plan:
            plan_text += chunk
            plan_placeholder.markdown(f"**Planner** (stream):\n```\n{plan_text}\n```")
            time.sleep(0.02)
        # Логируем RESPONSE
        log_chain("PlannerAgent", "RESPONSE", plan_text)
        st.markdown(f"**Итоговая инструкция Planner**:\n```\n{plan_text}\n```")

        current_instruction = plan_text
        # Запускаем цикл итераций
        for iteration in range(1, max_iterations + 1):
            st.markdown(f"## Итерация {iteration}")

            # A) ExecutorAgent
            st.markdown("### ExecutorAgent")
            log_chain("ExecutorAgent", "PROMPT", current_instruction)
            exec_text = ""
            ex_placeholder = st.empty()
            ex_gen = executor.execute_instruction(current_instruction, st.session_state.vector_store, stream=True, **ollama_opts)
            if isinstance(ex_gen, str):
                exec_text = ex_gen
                ex_placeholder.markdown(f"**Executor** (tool):\n```\n{exec_text}\n```")
            else:
                for c in ex_gen:
                    exec_text += c
                    ex_placeholder.markdown(f"**Executor** (stream):\n```\n{exec_text}\n```")
                    time.sleep(0.02)
            # Логируем RESP
            log_chain("ExecutorAgent", "RESPONSE", exec_text)
            st.markdown(f"**Ответ Executor**:\n```\n{exec_text}\n```")

            # B) CriticAgent
            st.markdown("### CriticAgent")
            log_chain("CriticAgent", "PROMPT", exec_text)
            cr_text = ""
            cr_placeholder = st.empty()
            cr_gen = critic.criticize(exec_text, stream=True, **ollama_opts)
            for c in cr_gen:
                cr_text += c
                cr_placeholder.markdown(f"**Critic** (stream):\n```\n{cr_text}\n```")
                time.sleep(0.02)
            log_chain("CriticAgent", "RESPONSE", cr_text)
            st.markdown(f"**Критика**:\n```\n{cr_text}\n```")

            # C) PraiseAgent
            st.markdown("### PraiseAgent")
            log_chain("PraiseAgent", "PROMPT", exec_text)
            pr_text = ""
            pr_placeholder = st.empty()
            pr_gen = praise.praise(exec_text, stream=True, **ollama_opts)
            for c in pr_gen:
                pr_text += c
                pr_placeholder.markdown(f"**Praise** (stream):\n```\n{pr_text}\n```")
                time.sleep(0.02)
            log_chain("PraiseAgent", "RESPONSE", pr_text)
            st.markdown(f"**Похвала**:\n```\n{pr_text}\n```")

            # D) ArbiterAgent => ReworkInstruction (если не последняя итерация)
            if iteration < max_iterations:
                st.markdown("### ArbiterAgent => ReworkInstruction")
                rework_text = ""
                arb_placeholder = st.empty()
                arb_prompt = f"(exec){exec_text}\n(critic){cr_text}\n(praise){pr_text}"
                log_chain("ArbiterAgent", "PROMPT", arb_prompt)
                arb_gen = arbiter.produce_rework_instruction(exec_text, cr_text, pr_text, stream=True, **ollama_opts)
                for c in arb_gen:
                    rework_text += c
                    arb_placeholder.markdown(f"**Arbiter** (stream):\n```\n{rework_text}\n```")
                    time.sleep(0.02)
                log_chain("ArbiterAgent", "RESPONSE", rework_text)
                st.markdown(f"**Rework Instruction**:\n```\n{rework_text}\n```")

                # E) ExecutorAgent снова будет на след. iteration
                current_instruction = rework_text
            else:
                # Если это последняя итерация — сохраним exec_text как финальный
                st.session_state.final_answer = exec_text
                st.info("Достигнут предел итераций. Можно ниже дополнительно перезапустить (Arbiter->Executor), если хотите.")
        
        # --- После цикла ---
        st.success("Многократный цикл завершён.")
        # Показываем «последний результат»
        st.markdown("## Итоговый результат после итераций")

        if st.session_state.final_answer:
            st.markdown(
                f"""
                <div style="border:2px solid #4CAF50; padding:10px; border-radius:5px; background-color:#f9fff9;">
                <h4>Финальный ответ:</h4>
                <pre style="white-space:pre-wrap;">{st.session_state.final_answer}</pre>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.warning("Нет сохранённого результата (возможно, Executor вернул пустую строку).")

        st.markdown("---")
        st.markdown("### Дополнительный финальный вызов (Arbiter->Executor)?")

        if st.button("Сделать дополнительный Rework и финальный Executor"):
            # 1) Арбитр генерирует ReworkInstruction (на основе final_answer, Critic/Praise)
            #    но для упрощения — Critic/Praise у нас уже есть? 
            #    Или можно заново запускать critic/praise? 
            #    Здесь просто возьмём последние cr_text, pr_text
            #    (Либо user сам редактирует cr_text/pr_text)
            st.info("Допустим, берём последний cr_text / pr_text.")
            # TODO: для надёжности стоило бы сохранить cr_text / pr_text в session_state.
            # Сейчас просто скажем, что cr_text = "Нет критики" ...
            # Для упрощения — если нужно, можно хранить "последние" в s.sate.

            st.error("В данном примере упрощённо нет сохранённых cr_text/pr_text на последней итерации.")
            st.warning("Чтобы передать свежие Critic/Praise, нужно хранить их. Здесь только шаблон логики.")

            # В теории:
            # rework_text = (Arbiter produce)
            # exec_text_2 = (Executor)

            st.info("Шаблон: здесь можно заново Arbiter->Executor.")
        
    st.markdown("## Chain-of-thought trace (пошаговый лог)")
    if st.session_state.chain_trace:
        display_chain_trace()
    else:
        st.info("Лог пуст. Запустите процесс, чтобы увидеть пошаговый вывод.")


if __name__ == "__main__":
    main()
