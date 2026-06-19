"""
Веб-интерфейс для консультанта Find My Movie (Gradio).

Интерфейс в стиле чата с GPT: слева — список чатов (каждый со своей памятью),
в центре — окно активного диалога.

Запуск:
    uv run python ui.py
Откроется в браузере (http://127.0.0.1:7860).

Нужны ключи в .env: GROQ_API_KEY (обязательно), TAVILY_API_KEY (для web_search).
И собранная база: uv run python indexer.py
"""

from __future__ import annotations

import os
import sys
import uuid

from dotenv import load_dotenv

load_dotenv()

if not os.getenv("GROQ_API_KEY"):
    sys.exit("Не задан GROQ_API_KEY в .env — без него голова не отвечает.")

import gradio as gr  # noqa: E402

import agent  # noqa: E402

NEW_TITLE = "Новый чат"

# Сессии (память + executor каждого чата) держим тут, по id. В gr.State кладём
# только сериализуемые метаданные — Session с executor'ом туда не положишь.
_SESSIONS: dict[str, agent.Session] = {}


def _warmup() -> None:
    """Прогрев на старте: грузим эмбеддер и клиент базы в память,
    чтобы первый запрос про фильмы не ждал ~10 c загрузки модели."""
    print("Прогрев эмбеддера и базы…", flush=True)
    try:
        from embeddings import COLLECTION, embed_query
        from tools import _client

        embed_query("прогрев")
        _client().count(COLLECTION)
        print("Готово — первый запрос будет быстрым.", flush=True)
    except Exception as e:
        print(f"Прогрев пропущен: {e}", flush=True)


# ------------------------------------------------------------- работа с чатами -
def _make_chat() -> dict:
    """Новый чат: свежая Session в реестре + метаданные для State."""
    cid = uuid.uuid4().hex[:8]
    _SESSIONS[cid] = agent.Session()
    return {"id": cid, "title": NEW_TITLE, "history": []}


def _choices(chats: list[dict]) -> list[tuple[str, str]]:
    """Список (заголовок, id) для радиокнопок сайдбара (новые — сверху)."""
    return [(c["title"], c["id"]) for c in reversed(chats)]


def _find(chats: list[dict], cid: str) -> dict | None:
    return next((c for c in chats if c["id"] == cid), None)


def init():
    """Стартовое состояние: один пустой чат."""
    chat = _make_chat()
    chats = [chat]
    active = chat["id"]
    return chats, active, gr.update(choices=_choices(chats), value=active), []


def new_chat(chats):
    chat = _make_chat()
    chats = chats + [chat]
    active = chat["id"]
    return chats, active, gr.update(choices=_choices(chats), value=active), []


def select_chat(cid, chats):
    """Переключение на выбранный в сайдбаре чат — грузим его историю."""
    c = _find(chats, cid)
    return cid, (c["history"] if c else [])


def delete_chat(chats, active):
    """Удалить активный чат (и его память). Если был последним — заводим пустой."""
    chats = [c for c in chats if c["id"] != active]
    _SESSIONS.pop(active, None)
    if not chats:
        chats = [_make_chat()]
    active = chats[-1]["id"]
    c = _find(chats, active)
    return chats, active, gr.update(choices=_choices(chats), value=active), c["history"]


def send(message, chats, active):
    """Отправка сообщения: показываем реплику сразу, потом — ответ агента."""
    message = (message or "").strip()
    if not message:
        yield "", chats, active, _find(chats, active)["history"], gr.update()
        return

    c = _find(chats, active)
    if c is None:  # на всякий случай — активного чата нет
        c = _make_chat()
        chats = chats + [c]
        active = c["id"]

    # заголовок чата — из первой реплики пользователя
    if c["title"] == NEW_TITLE:
        c["title"] = message[:30] + ("…" if len(message) > 30 else "")

    c["history"] = c["history"] + [{"role": "user", "content": message}]
    # первый yield: очищаем поле, показываем сообщение и обновлённый заголовок
    yield "", chats, active, c["history"], gr.update(choices=_choices(chats), value=active)

    try:
        reply = _SESSIONS[c["id"]].respond(message)
    except Exception as e:
        reply = f"⚠️ Ошибка: {e}"
    c["history"] = c["history"] + [{"role": "assistant", "content": reply}]
    yield "", chats, active, c["history"], gr.update(choices=_choices(chats), value=active)


# ------------------------------------------------------------------ интерфейс --
with gr.Blocks(title="Find My Movie", fill_height=True) as demo:
    chats = gr.State([])
    active = gr.State("")

    with gr.Row(equal_height=False):
        # --- сайдбар со списком чатов ---
        with gr.Column(scale=1, min_width=220):
            gr.Markdown("### 🎬 Find My Movie")
            new_btn = gr.Button("➕ Новый чат", variant="primary")
            chat_list = gr.Radio(
                label="Чаты", choices=[], value=None, container=True
            )
            del_btn = gr.Button("🗑 Удалить чат", size="sm")

        # --- центральное окно активного чата ---
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(
                height="70vh",
                show_label=False,
                avatar_images=(None, "🎬"),
                placeholder="Спроси по настроению, попроси детали или похожее.",
            )
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Например: что-то напряжённое про выживание в космосе…",
                    show_label=False,
                    scale=8,
                    autofocus=True,
                )
                send_btn = gr.Button("Отправить", variant="primary", scale=1, min_width=120)

    # --- связи ---
    demo.load(init, outputs=[chats, active, chat_list, chatbot])
    new_btn.click(new_chat, [chats], [chats, active, chat_list, chatbot])
    del_btn.click(delete_chat, [chats, active], [chats, active, chat_list, chatbot])
    chat_list.input(select_chat, [chat_list, chats], [active, chatbot])

    send_args = ([msg, chats, active], [msg, chats, active, chatbot, chat_list])
    send_btn.click(send, *send_args)
    msg.submit(send, *send_args)


if __name__ == "__main__":
    _warmup()
    demo.launch(inbrowser=True)
