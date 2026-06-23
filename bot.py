"""
Telegram-бот поверх агента Find My Movie.

Каждому чату — своя сессия (своя память диалога). Тяжёлый respond() (LLM +
эмбеддинг + Qdrant) выполняется в отдельном потоке, чтобы не блокировать
event loop. Обращения к локальной базе сериализуются одним lock'ом —
файловый Qdrant не гарантирует потокобезопасность.

Запуск:
    uv run python bot.py

Нужны ключи в .env:
    TELEGRAM_BOT_TOKEN — токен от @BotFather (обязательно)
    LLM_API_KEY        — голова + роутер (обязательно)
    TAVILY_API_KEY     — web_search (опционально)
И собранная база: uv run python indexer.py
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import threading

from dotenv import load_dotenv

load_dotenv()

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not TOKEN:
    sys.exit("Не задан TELEGRAM_BOT_TOKEN в .env — получи токен у @BotFather.")
if not os.getenv("LLM_API_KEY"):
    sys.exit("Не задан LLM_API_KEY в .env — без него голова не отвечает.")

from telegram import Update  # noqa: E402
from telegram.constants import ChatAction  # noqa: E402
from telegram.ext import (  # noqa: E402
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

import backend.agent as agent  # noqa: E402

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s: %(message)s", level=logging.INFO
)
log = logging.getLogger("findmymovie-bot")

# chat_id -> Session (своя память на каждый чат)
_sessions: dict[int, agent.Session] = {}
# файловый Qdrant — один процесс; сериализуем обращения к базе
_db_lock = threading.Lock()

WELCOME = (
    "🎬 Привет! Я консультант по фильмам.\n\n"
    "Опиши, что хочешь посмотреть («что-то напряжённое про космос», «лёгкая комедия "
    "на вечер»), попроси детали или похожее — подберу из базы (фильмы 2019–2026). "
    "Могу и в интернет за отзывами сходить.\n\n"
    "/reset — очистить память диалога"
)


def _session(chat_id: int) -> agent.Session:
    if chat_id not in _sessions:
        _sessions[chat_id] = agent.Session()
    return _sessions[chat_id]


def _answer(chat_id: int, text: str) -> str:
    """Блокирующий вызов агента под lock'ом — гоняется в отдельном потоке."""
    with _db_lock:
        return _session(chat_id).respond(text)


async def start(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    _sessions.pop(update.effective_chat.id, None)
    await update.message.reply_text(WELCOME)


async def reset(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    _sessions.pop(update.effective_chat.id, None)
    await update.message.reply_text("🧹 Память очищена — начнём заново.")


async def on_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    text = (update.message.text or "").strip()
    if not text:
        return
    await context.bot.send_chat_action(chat_id, ChatAction.TYPING)
    try:
        reply = await asyncio.to_thread(_answer, chat_id, text)
    except Exception as e:  # не валим бота на ошибке одного запроса
        log.exception("ошибка обработки сообщения")
        reply = f"⚠️ Что-то пошло не так: {e}"
    await update.message.reply_text(reply)


def main() -> None:
    log.info("Прогрев эмбеддера и базы…")
    try:
        from backend.embeddings import COLLECTION, embed_query
        from backend.tools import _client

        embed_query("прогрев")
        _client().count(COLLECTION)
        log.info("Прогрев готов.")
    except Exception as e:
        log.warning("Прогрев пропущен: %s", e)

    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("reset", reset))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_message))
    log.info("Бот запущен. Жду сообщения (Ctrl+C — выход).")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
