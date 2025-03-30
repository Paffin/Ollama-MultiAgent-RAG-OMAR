import asyncio
import sys
import streamlit.web.bootstrap
from streamlit.web.server import Server

def run():
    """Запускает Streamlit приложение с правильной настройкой event loop."""
    if sys.platform == "darwin":
        asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
    
    # Создаем новый event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    # Запускаем приложение
    streamlit.web.bootstrap.run("streamlit_app.py", "", [], {})

if __name__ == "__main__":
    run() 