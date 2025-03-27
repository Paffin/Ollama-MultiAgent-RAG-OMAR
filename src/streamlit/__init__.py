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
from .app import run_app

__all__ = [
    'st',
    'run_app',
    'AgentChain',
    'AnalyticsDashboard',
    'DataProcessingPanel',
    'NotificationPanel',
    'SettingsPanel'
] 