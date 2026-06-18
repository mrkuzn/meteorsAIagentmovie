"""
Веб-интерфейс для консультанта Find My Movie (Gradio).

Запуск:
    uv run python ui.py
Откроется чат в браузере (http://127.0.0.1:7860).

Нужны ключи в .env: GROQ_API_KEY (обязательно), TAVILY_API_KEY (для web_search).
И собранная база: uv run python indexer.py
"""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv

load_dotenv()

if not os.getenv("GROQ_API_KEY"):
    sys.exit("Не задан GROQ_API_KEY в .env — без него голова не отвечает.")

import gradio as gr  # noqa: E402

import agent  # noqa: E402
from agent import respond  # noqa: E402


def _warmup() -> None:
    """Прогрев на старте: грузим эмбеддер и клиент базы в память,
    чтобы первый запрос про фильмы не ждал ~10 c загрузки модели."""
    print("Прогрев эмбеддера и базы…", flush=True)
    try:
        from embeddings import embed_query
        from tools import _client

        embed_query("прогрев")
        _client().count(__import__("embeddings").COLLECTION)
        print("Готово — первый запрос будет быстрым.", flush=True)
    except Exception as e:
        print(f"Прогрев пропущен: {e}", flush=True)


def chat(message: str, history: list) -> str:
    # пустая история = новый диалог или нажали «Очистить» -> сбрасываем память агента
    if not history:
        agent.memory.clear()
    try:
        return respond(message)
    except Exception as e:
        return f"⚠️ Ошибка: {e}"


demo = gr.ChatInterface(
    fn=chat,
    title="🎬 Find My Movie — консультант по фильмам",
    description=(
        "Спрашивай по настроению («что-то напряжённое про космос»), проси детали "
        "или похожее. Память держит контекст диалога. База: фильмы 2019–2026."
    ),
    examples=[
        "посоветуй напряжённый фильм про выживание в космосе",
        "хочу лёгкую комедию на вечер",
        "расскажи подробнее про первый",
        "а что похожее посоветуешь?",
    ],
)


if __name__ == "__main__":
    _warmup()
    demo.launch(inbrowser=True)
