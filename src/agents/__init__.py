"""
Агенты для OMAR
"""

from .base import BaseAgent
from .planner import PlannerAgent
from .executor import ExecutorAgent
from .critic import CriticAgent
from .praise import PraiseAgent
from .arbiter import ArbiterAgent

__all__ = [
    'BaseAgent',
    'PlannerAgent',
    'ExecutorAgent',
    'CriticAgent',
    'PraiseAgent',
    'ArbiterAgent'
] 