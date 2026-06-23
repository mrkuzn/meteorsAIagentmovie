from __future__ import annotations

import os
import uuid
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# URL вашего FastAPI бэкенда в Selectel.
# На локальной машине используется http://localhost:8000. 
# При деплое в Streamlit Cloud этот IP прописывается в Advanced Settings -> Secrets.
BACKEND_URL = st.secrets.get("BACKEND_URL", os.getenv("BACKEND_URL", "http://localhost:8000"))
CHAT_ENDPOINT = f"{BACKEND_URL}/api/chat"

st.set_page_config(page_title="Find My Movie", page_icon="🎬", layout="wide")

NEW_TITLE = "Новый чат"
ASSISTANT_AVATAR = "🎬"

# --- Функции обратного вызова (Callbacks) и управление стейтом ---

def _make_chat() -> dict:
    """Создает структуру данных для нового чата."""
    return {
        "id": uuid.uuid4().hex[:8],
        "title": NEW_TITLE,
        "history": []  # Локальная история для отображения: [{"role": "user"|"assistant", "content": str}]
    }

def _ensure_state() -> None:
    """Инициализация хранилища чатов при первом заходе."""
    if "chats" not in st.session_state:
        chat = _make_chat()
        st.session_state.chats = [chat]
        st.session_state.active = chat["id"]

def _new_chat() -> None:
    """Создает новый чат и делает его активным."""
    chat = _make_chat()
    st.session_state.chats.append(chat)
    st.session_state.active = chat["id"]

def _select(cid: str) -> None:
    """Переключает активный чат."""
    st.session_state.active = cid

def _delete(cid: str) -> None:
    """Удаляет чат. Если чатов не осталось, создает пустой."""
    st.session_state.chats = [c for c in st.session_state.chats if c["id"] != cid]
    if not st.session_state.chats:
        st.session_state.chats = [_make_chat()]
    if st.session_state.active == cid:
        st.session_state.active = st.session_state.chats[-1]["id"]

def _active() -> dict:
    """Возвращает текущий активный чат."""
    cid = st.session_state.active
    return next(c for c in st.session_state.chats if c["id"] == cid)

# --- Инициализация стейта ---
# Вызов _warmup() удален — прогрев теперь происходит на FastAPI при старте сервера Selectel!
_ensure_state()

# --- Боковая панель (Sidebar) ---
with st.sidebar:
    st.markdown("## 🎬 Find My Movie")
    st.button("➕ Новый чат", use_container_width=True, type="primary", on_click=_new_chat)
    st.divider()
    
    # Список чатов: активный подсвечен, рядом кнопка удаления (в обратном порядке)
    for c in reversed(st.session_state.chats):
        is_active = (c["id"] == st.session_state.active)
        row, dele = st.columns([5, 1])
        
        row.button(
            c["title"],
            key=f"sel_{c['id']}",
            use_container_width=True,
            type="primary" if is_active else "secondary",
            on_click=_select,
            args=(c["id"],)
        )
        
        dele.button("🗑️", key=f"del_{c['id']}", on_click=_delete, args=(c["id"],))

    st.divider()
    st.caption("Подбор по настроению, детали и похожее — из базы ~8000 фильмов.")

# --- Центр: активный чат ---
chat = _active()

for m in chat["history"]:
    avatar = ASSISTANT_AVATAR if m["role"] == "assistant" else None
    with st.chat_message(m["role"], avatar=avatar):
        st.markdown(m["content"])

# Ввод нового сообщения с кастомным плейсхолдером
if prompt := st.chat_input("Например: что-то напряжённое про выживание в космосе..."):
    prompt = prompt.strip()
    
    # Автоматическое переименование заголовка чата по вашей логике (до 30 символов)
    if chat["title"] == NEW_TITLE:
        chat["title"] = prompt[:30] + ("..." if len(prompt) > 30 else "")
        
    chat["history"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        
    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        with st.spinner("Подбираю..."):
            try:
                # HTTP-запрос к вашему новому LangGraph бэкенду на Selectel
                payload = {
                    "message": prompt,
                    "session_id": chat["id"]  # Передается как thread_id в LangGraph для изоляции памяти
                }
                response = requests.post(CHAT_ENDPOINT, json=payload, timeout=90)
                
                if response.status_code == 200:
                    reply = response.json()["response"]
                else:
                    error_detail = response.json().get("detail", "Неизвестная ошибка сервера")
                    reply = f"⚠️ Ошибка сервера ({response.status_code}): {error_detail}"
                    
            except Exception as e:  # Не валим интерфейс на ошибке модели/сети
                reply = f"⚠️ Ошибка: {e}"
                
            st.markdown(reply)

    chat["history"].append({"role": "assistant", "content": reply})
    # Подсветка нового заголовка в сайдбаре подхватится на следующем ране
    st.rerun()
