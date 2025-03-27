"""
Модуль Streamlit интерфейса
"""

import streamlit as st
from .components import (
    AgentChain,
    AnalyticsDashboard,
    DataProcessingPanel,
    NotificationPanel,
    SettingsPanel
)

__all__ = [
    'st',
    'AgentChain',
    'AnalyticsDashboard',
    'DataProcessingPanel',
    'NotificationPanel',
    'SettingsPanel'
] 