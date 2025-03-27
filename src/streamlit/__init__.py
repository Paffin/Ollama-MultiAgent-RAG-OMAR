"""
Модуль Streamlit интерфейса
"""

from .app import run
from .components import (
    AgentChain,
    AnalyticsDashboard,
    DataProcessingPanel,
    NotificationPanel,
    SettingsPanel
)

__all__ = [
    'run',
    'AgentChain',
    'AnalyticsDashboard',
    'DataProcessingPanel',
    'NotificationPanel',
    'SettingsPanel'
] 