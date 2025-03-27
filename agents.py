# agents.py

from typing import List, Dict, Union, Generator
from ollama_client import OllamaClient
from rag_db import SimpleVectorStore
from windows_tools import run_system_command, list_directory

class BaseAgent:
    def __init__(self, name: str, system_prompt: str, model_name: str, client: OllamaClient):
        self.name = name
        self.system_prompt = system_prompt
        self.model_name = model_name
        self.client = client
        self.history: List[Dict[str, str]] = []
    
    def add_message(self, role: str, content: str):
        self.history.append({"role": role, "content": content})
    
    def build_prompt(self) -> str:
        parts = [f"System({self.name}): {self.system_prompt}\n"]
        for msg in self.history:
            role = msg["role"]
            text = msg["content"]
            parts.append(f"{role.upper()}: {text}\n")
        return "\n".join(parts)


class PlannerAgent(BaseAgent):
    """
    Принимает user_query, выдаёт инструкцию для Executor
    """
    def generate_instruction(
        self, 
        user_query: str, 
        stream=False, 
        **ollama_opts
    ) -> Union[str, Generator[str, None, None]]:
        self.add_message("user", user_query)
        prompt = self.build_prompt()

        if not stream:
            resp = self.client.generate(prompt=prompt, model=self.model_name, stream=False, **ollama_opts)
            self.add_message("assistant", resp)
            return resp
        else:
            gen = self.client.generate(prompt=prompt, model=self.model_name, stream=True, **ollama_opts)
            return gen


class ExecutorAgent(BaseAgent):
    """
    Получает инструкцию, если в ней есть "search:", "cmd:", "ls:", вызывает соотв. инструменты.
    Иначе - LLM ответ
    """
    def execute_instruction(
        self,
        instruction: str,
        vector_store: SimpleVectorStore,
        stream=False,
        **ollama_opts
    ) -> Union[str, Generator[str, None, None]]:
        self.add_message("user", instruction)

        # 1) search:
        if "search:" in instruction.lower():
            query = instruction.split("search:")[1].strip()
            found_docs = vector_store.search(query, k=3)
            tool_out = f"[RAG] Найдено документов: {len(found_docs)}\n"
            for i, d in enumerate(found_docs, 1):
                tool_out += f"{i}. {d[:200]}...\n"
            self.add_message("assistant", tool_out)
            return tool_out
        
        # 2) cmd:
        if "cmd:" in instruction.lower():
            cmd_text = instruction.split("cmd:")[1].strip()
            tool_out = run_system_command(cmd_text)
            self.add_message("assistant", tool_out)
            return tool_out
        
        # 3) ls:
        if "ls:" in instruction.lower():
            path = instruction.split("ls:")[1].strip()
            tool_out = list_directory(path)
            self.add_message("assistant", tool_out)
            return tool_out
        
        # 4) LLM
        prompt = self.build_prompt()
        if not stream:
            resp = self.client.generate(prompt=prompt, model=self.model_name, stream=False, **ollama_opts)
            self.add_message("assistant", resp)
            return resp
        else:
            gen = self.client.generate(prompt=prompt, model=self.model_name, stream=True, **ollama_opts)
            return gen


class CriticAgent(BaseAgent):
    """
    Критикует ответ Executor (находит ошибки)
    """
    def criticize(self, executor_result: str, stream=False, **ollama_opts) -> Union[str, Generator[str, None, None]]:
        user_msg = (
            "Вот ответ Исполнителя. Определи ошибки, слабые стороны, неточности:\n\n"
            f"{executor_result}"
        )
        self.add_message("user", user_msg)
        prompt = self.build_prompt()

        if not stream:
            resp = self.client.generate(prompt=prompt, model=self.model_name, stream=False, **ollama_opts)
            self.add_message("assistant", resp)
            return resp
        else:
            gen = self.client.generate(prompt=prompt, model=self.model_name, stream=True, **ollama_opts)
            return gen


class PraiseAgent(BaseAgent):
    """
    Выделяет плюсы ответа Executor
    """
    def praise(self, executor_result: str, stream=False, **ollama_opts) -> Union[str, Generator[str, None, None]]:
        user_msg = (
            "Вот ответ Исполнителя. Покажи, что в нём хорошего, какие сильные стороны:\n\n"
            f"{executor_result}"
        )
        self.add_message("user", user_msg)
        prompt = self.build_prompt()

        if not stream:
            resp = self.client.generate(prompt=prompt, model=self.model_name, stream=False, **ollama_opts)
            self.add_message("assistant", resp)
            return resp
        else:
            gen = self.client.generate(prompt=prompt, model=self.model_name, stream=True, **ollama_opts)
            return gen


class ArbiterAgent(BaseAgent):
    """
    Генерирует «Rework Instruction» для Executor, 
    учитывая (исходный ответ, критику, похвалу).
    """
    def produce_rework_instruction(
        self,
        executor_result: str,
        critic_text: str,
        praise_text: str,
        stream=False,
        **ollama_opts
    ) -> Union[str, Generator[str, None, None]]:
        user_msg = (
            "У нас есть:\n"
            f"1) Ответ Исполнителя:\n{executor_result}\n\n"
            f"2) Критика:\n{critic_text}\n\n"
            f"3) Похвала:\n{praise_text}\n\n"
            "Сформируй инструкцию, как улучшить ответ, "
            "не переписывая всё с нуля (если нет больших ошибок). "
            "Опиши чётко, что Исполнителю нужно доработать."
        )
        self.add_message("user", user_msg)
        prompt = self.build_prompt()

        if not stream:
            resp = self.client.generate(prompt=prompt, model=self.model_name, stream=False, **ollama_opts)
            self.add_message("assistant", resp)
            return resp
        else:
            gen = self.client.generate(prompt=prompt, model=self.model_name, stream=True, **ollama_opts)
            return gen
