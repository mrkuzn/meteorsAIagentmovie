"""
Веб-интерфейс консультанта Find My Movie на Streamlit (стиль чата с GPT).

Слева — список чатов (у каждого своя память), в центре — активный диалог.
Каждый чат = отдельная agent.Session (своя ConversationBufferMemory + executor),
поэтому разные чаты не путают контекст.

Запуск:
    uv run streamlit run ui_streamlit.py
Откроется в браузере (http://localhost:8501).

Нужны ключи в .env: LLM_API_KEY (обязательно), TAVILY_API_KEY (для web_search).
И собранная база: uv run python indexer.py
"""

from __future__ import annotations

import os
import sys
import uuid

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Find My Movie", page_icon="🎬", layout="wide")

if not os.getenv("LLM_API_KEY"):
    st.error("Не задан LLM_API_KEY в .env — без него голова не отвечает.")
    sys.exit(1)

import agent  # noqa: E402  (после проверки ключа: импорт тянет модели)

NEW_TITLE = "Новый чат"
ASSISTANT_AVATAR = "🎬"


@st.cache_resource(show_spinner="Прогреваю эмбеддер и базу…")
def _warmup() -> bool:
    """Один раз на процесс: грузим эмбеддер и клиент базы, чтобы первый
    запрос про фильмы не ждал ~10 c загрузки модели. Кэш — на весь рантайм."""
    try:
        from embeddings import COLLECTION, embed_query
        from tools import _client

        embed_query("прогрев")
        _client().count(COLLECTION)
    except Exception as e:  # прогрев не критичен — просто будет первый запрос медленнее
        print(f"Прогрев пропущен: {e}", flush=True)
    return True


# ------------------------------------------------------------- работа с чатами -
def _make_chat() -> dict:
    """Новый чат: свежая Session (память + executor) и пустая история."""
    return {
        "id": uuid.uuid4().hex[:8],
        "title": NEW_TITLE,
        "history": [],            # [{"role": "user"|"assistant", "content": str}]
        "session": agent.Session(),
    }


def _active() -> dict:
    """Текущий активный чат (гарантированно существует)."""
    cid = st.session_state.active
    return next(c for c in st.session_state.chats if c["id"] == cid)


def _ensure_state() -> None:
    """Инициализация хранилища чатов при первом заходе."""
    if "chats" not in st.session_state:
        chat = _make_chat()
        st.session_state.chats = [chat]
        st.session_state.active = chat["id"]


def _new_chat() -> None:
    chat = _make_chat()
    st.session_state.chats.append(chat)
    st.session_state.active = chat["id"]


def _select(cid: str) -> None:
    st.session_state.active = cid


def _delete(cid: str) -> None:
    """Удалить чат (вместе с его памятью). Последний — заменяем пустым."""
    st.session_state.chats = [c for c in st.session_state.chats if c["id"] != cid]
    if not st.session_state.chats:
        st.session_state.chats = [_make_chat()]
    if st.session_state.active == cid:
        st.session_state.active = st.session_state.chats[-1]["id"]


# ------------------------------------------------------------------ интерфейс --
_warmup()
_ensure_state()

with st.sidebar:
    st.markdown("## 🎬 Find My Movie")
    st.button("➕ Новый чат", use_container_width=True, type="primary", on_click=_new_chat)
    st.divider()

    # список чатов: активный подсвечен, рядом — кнопка удаления
    for c in reversed(st.session_state.chats):
        is_active = c["id"] == st.session_state.active
        row, dele = st.columns([5, 1])
        row.button(
            c["title"],
            key=f"sel_{c['id']}",
            use_container_width=True,
            type="primary" if is_active else "secondary",
            on_click=_select,
            args=(c["id"],),
        )
        dele.button("🗑", key=f"del_{c['id']}", on_click=_delete, args=(c["id"],))

    st.divider()
    st.caption("Подбор по настроению, детали и похожее — из базы ~8000 фильмов.")

# --- центр: активный чат ---
chat = _active()

for m in chat["history"]:
    avatar = ASSISTANT_AVATAR if m["role"] == "assistant" else None
    with st.chat_message(m["role"], avatar=avatar):
        st.markdown(m["content"])

if prompt := st.chat_input("Например: что-то напряжённое про выживание в космосе…"):
    prompt = prompt.strip()
    if chat["title"] == NEW_TITLE:
        chat["title"] = prompt[:30] + ("…" if len(prompt) > 30 else "")

    chat["history"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        with st.spinner("Подбираю…"):
            try:
                reply = chat["session"].respond(prompt)
            except Exception as e:  # не валим интерфейс на ошибке модели/сети
                reply = f"⚠️ Ошибка: {e}"
        st.markdown(reply)

    chat["history"].append({"role": "assistant", "content": reply})
    # подсветка нового заголовка в сайдбаре подхватится на следующем ране
    st.rerun()
