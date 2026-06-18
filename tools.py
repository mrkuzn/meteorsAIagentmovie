"""
Инструменты агента Find My Movie (4 шт, как требует README).

  search_content — семантический поиск по базе (+ опциональные фильтры)
  get_details    — полная карточка фильма по ID
  find_similar   — похожие фильмы по ID
  web_search     — отзывы/свежая инфа через Tavily

Все инструменты работают поверх локальной базы Qdrant (data/db/), собранной indexer.py.
Поиск использует тот же эмбеддер e5-large (embeddings.embed_query, префикс 'query:').
"""

from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv
from langchain_core.tools import tool
from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    HasIdCondition,
    MatchValue,
    Range,
)

from embeddings import COLLECTION, DB_PATH, embed_query

load_dotenv()


@lru_cache(maxsize=1)
def _client() -> QdrantClient:
    """Единый клиент Qdrant (локальная файловая база — один процесс за раз)."""
    return QdrantClient(path=DB_PATH)


# ------------------------------------------------------------- форматирование -
def _short(payload: dict) -> str:
    """Краткая строка фильма для списков (с ID, чтобы агент мог уточнить)."""
    genres = ", ".join(payload.get("genres") or [])
    rating = payload.get("rating")
    rating_s = f"★{rating}" if rating else "★—"
    return (
        f"[ID {payload['id']}] {payload['title']} ({payload.get('year') or '—'}) "
        f"{rating_s} | {genres}"
    )


def _full(payload: dict) -> str:
    """Полная карточка фильма."""
    def join(key):
        return ", ".join(payload.get(key) or []) or "—"

    return (
        f"[ID {payload['id']}] {payload['title']}"
        + (f" / {payload['original_title']}" if payload.get("original_title") else "")
        + f"\nГод: {payload.get('year') or '—'}"
        f"\nЖанры: {join('genres')}"
        f"\nРежиссёр: {join('director')}"
        f"\nВ ролях: {join('cast')}"
        f"\nСтрана: {join('country')}"
        f"\nРейтинг: {payload.get('rating') or '—'} ({payload.get('num_ratings') or 0} оценок)"
        f"\nДлительность: {payload.get('duration_min') or '—'} мин"
        f"\nОписание: {payload.get('description') or '—'}"
        f"\nСсылка: {payload.get('url') or '—'}"
    )


# -------------------------------------------------------------- инструменты ---
@tool
def search_content(
    query: str,
    year_from: int = 0,
    year_to: int = 0,
    genre: str = "",
    min_rating: float = 0.0,
    limit: int = 5,
) -> str:
    """Семантический поиск фильмов по смысловому описанию запроса.

    Используй для запросов вида «хочу что-то про одиночество в мегаполисе»,
    «напряжённый триллер про выживание», «лёгкое и смешное на вечер».
    Опциональные фильтры:
      year_from / year_to — диапазон годов (0 = без ограничения);
      genre — точный жанр на русском (например «комедия», «драма», «фантастика»);
      min_rating — минимальный рейтинг (0 = любой);
      limit — сколько результатов вернуть (по умолчанию 5).
    Возвращает список фильмов с их ID (по ID можно запросить детали или похожие).
    """
    conditions: list = []
    if year_from or year_to:
        conditions.append(
            FieldCondition(
                key="year", range=Range(gte=year_from or None, lte=year_to or None)
            )
        )
    if genre:
        conditions.append(FieldCondition(key="genres", match=MatchValue(value=genre)))
    if min_rating:
        conditions.append(FieldCondition(key="rating", range=Range(gte=min_rating)))
    qfilter = Filter(must=conditions) if conditions else None

    hits = _client().query_points(
        COLLECTION,
        query=embed_query(query),
        query_filter=qfilter,
        limit=limit,
        with_payload=True,
    ).points
    if not hits:
        return "По запросу ничего не найдено (возможно, фильтры слишком строгие)."
    return "\n".join(_short(h.payload) for h in hits)


@tool
def get_details(movie_id: int) -> str:
    """Полная карточка фильма по его ID из базы.

    Используй, когда нужно подробно рассказать про конкретный фильм или ответить
    на уточняющий вопрос (режиссёр, актёры, длительность, рейтинг, описание).
    ID берётся из результатов search_content или find_similar.
    """
    res = _client().retrieve(COLLECTION, ids=[movie_id], with_payload=True)
    if not res:
        return f"Фильм с ID {movie_id} не найден."
    return _full(res[0].payload)


@tool
def find_similar(movie_id: int, limit: int = 5) -> str:
    """Найти фильмы, похожие на фильм с заданным ID (по смыслу/вектору).

    Используй для запросов «понравился вот этот — что ещё посмотреть»,
    «хочу что-то похожее». ID берётся из предыдущих результатов.
    """
    res = _client().retrieve(COLLECTION, ids=[movie_id], with_vectors=True, with_payload=True)
    if not res:
        return f"Фильм с ID {movie_id} не найден."
    vector = res[0].vector
    hits = _client().query_points(
        COLLECTION,
        query=vector,
        query_filter=Filter(must_not=[HasIdCondition(has_id=[movie_id])]),
        limit=limit,
        with_payload=True,
    ).points
    base = res[0].payload["title"]
    if not hits:
        return f"Похожих на «{base}» не нашлось."
    return f"Похожие на «{base}»:\n" + "\n".join(_short(h.payload) for h in hits)


@tool
def web_search(query: str) -> str:
    """Поиск в интернете: отзывы, рецензии, свежая информация о фильмах.

    Используй, когда ответа НЕТ в базе: «что пишут критики», «вышел ли новый фильм
    этого режиссёра», актуальные новости. Не используй для обычного поиска по базе.
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return "Веб-поиск недоступен: не задан TAVILY_API_KEY."
    from tavily import TavilyClient

    try:
        resp = TavilyClient(api_key=api_key).search(query, max_results=5)
    except Exception as e:  # не валим агента на сетевой ошибке
        return f"Ошибка веб-поиска: {e}"
    results = resp.get("results", [])
    if not results:
        return "В интернете ничего релевантного не нашлось."
    lines = []
    for r in results:
        content = (r.get("content") or "")[:300]
        lines.append(f"- {r.get('title', '')}\n  {content}\n  {r.get('url', '')}")
    return "\n".join(lines)


# все инструменты — для импорта в agent.py
ALL_TOOLS = [search_content, get_details, find_similar, web_search]


if __name__ == "__main__":
    # быстрый тест инструментов (только когда app.py не запущен — база эксклюзивна)
    print("== search_content ==")
    print(search_content.invoke({"query": "напряжённое выживание в космосе", "limit": 3}))
    print("\n== get_details (первый id из поиска) ==")
    print(get_details.invoke({"movie_id": 0}))
    print("\n== find_similar ==")
    print(find_similar.invoke({"movie_id": 0, "limit": 3}))
