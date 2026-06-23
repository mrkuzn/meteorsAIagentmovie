"""
Точка входа — интерактивный чат с консультантом Find My Movie.

Запуск:
    uv run python app.py

Нужны ключи в .env: LLM_API_KEY (голова+роутер), TAVILY_API_KEY (web_search).
Перед первым запуском должна быть собрана база: uv run python indexer.py
"""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv

load_dotenv()


def main() -> None:
    if not os.getenv("LLM_API_KEY"):
        sys.exit("Не задан LLM_API_KEY в .env — без него голова не отвечает.")

    # импорт после проверки ключа (agent.py поднимает модели и базу)
    from backend.agent import respond

    print("🎬 Find My Movie — консультант по фильмам.")
    print("   Спрашивай что посмотреть, проси детали или похожее. 'выход' — закончить.\n")

    while True:
        try:
            user = input("Вы: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nПока!")
            break
        if user.lower() in {"выход", "exit", "quit", "q"}:
            print("Пока!")
            break
        if not user:
            continue
        try:
            print(f"\nАгент: {respond(user)}\n")
        except Exception as e:
            print(f"\n[ошибка] {e}\n")


if __name__ == "__main__":
    main()
