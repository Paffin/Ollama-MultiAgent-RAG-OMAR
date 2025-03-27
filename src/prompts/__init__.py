"""
Модуль промптов для различных агентов
"""

from .planner import PLANNER_PROMPT
from .executor import EXECUTOR_PROMPT
from .critic import CRITIC_PROMPT
from .praise import PRAISE_PROMPT
from .arbiter import ARBITER_PROMPT

__all__ = [
    'PLANNER_PROMPT',
    'EXECUTOR_PROMPT',
    'CRITIC_PROMPT',
    'PRAISE_PROMPT',
    'ARBITER_PROMPT'
] 