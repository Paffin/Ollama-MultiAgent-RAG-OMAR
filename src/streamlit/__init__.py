"""
Модуль Streamlit интерфейса
"""

import streamlit as st
from src.streamlit.components import (
    AgentChain,
    AnalyticsDashboard,
    DataProcessingPanel,
    NotificationPanel,
    SettingsPanel
)
from src.streamlit.app import run_app

__all__ = [
    'st',
    'run_app',
    'AgentChain',
    'AnalyticsDashboard',
    'DataProcessingPanel',
    'NotificationPanel',
    'SettingsPanel'
] 