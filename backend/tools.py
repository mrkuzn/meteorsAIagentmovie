"""
Инструменты агента Find My Movie (4 шт, как требует README).

  search_content — гибридный поиск по базе (семантика e5 + лексика BM25, RRF) + фильтры
  get_details    — полная карточка фильма по названию
  find_similar   — похожие фильмы по названию
  web_search     — отзывы/свежая инфа через Tavily

Все инструменты работают поверх локальной базы Qdrant (data/db/), собранной indexer.py.
Семантика — эмбеддер e5-large (embeddings.embed_query, префикс 'query:'),
лексика — BM25 (embeddings.embed_sparse_query); слияние рейтингов через RRF в Qdrant.
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
    Fusion,
    FusionQuery,
    HasIdCondition,
    MatchValue,
    Prefetch,
    Range,
    SparseVector,
)

from backend.embeddings import (
    COLLECTION,
    DB_PATH,
    DENSE_NAME,
    SPARSE_NAME,
    embed_query,
    embed_sparse_query,
)

load_dotenv()


@lru_cache(maxsize=1)
def _client() -> QdrantClient:
    """Единый клиент Qdrant (локальная файловая база — один процесс за раз)."""
    return QdrantClient(path=DB_PATH)


# ------------------------------------------------------------- форматирование -
def _short(payload: dict) -> str:
    """Краткая строка фильма для списков (по названию — без числовых ID)."""
    genres = ", ".join(payload.get("genres") or [])
    rating = payload.get("rating")
    rating_s = f"★{rating}" if rating else "★—"
    return (
        f"{payload['title']} ({payload.get('year') or '—'}) {rating_s} | {genres}"
    )


def _resolve(title: str):
    """Найти фильм по названию: сначала точное совпадение, иначе — семантически ближайший."""
    pts, _ = _client().scroll(
        COLLECTION,
        scroll_filter=Filter(must=[FieldCondition(key="title", match=MatchValue(value=title))]),
        limit=1,
        with_payload=True,
        with_vectors=True,
    )
    if pts:
        return pts[0]
    hits = _client().query_points(
        COLLECTION,
        query=embed_query(title),
        using=DENSE_NAME,
        limit=1,
        with_payload=True,
        with_vectors=True,
    ).points
    return hits[0] if hits else None


def _full(payload: dict) -> str:
    """Полная карточка фильма."""
    def join(key):
        return ", ".join(payload.get(key) or []) or "—"

    return (
        f"{payload['title']}"
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
    """Гибридный поиск фильмов: смысл (семантика) + точные слова (лексика BM25).

    Используй для запросов вида «хочу что-то про одиночество в мегаполисе»,
    «напряжённый триллер про выживание», «лёгкое и смешное на вечер», а также
    когда в запросе есть конкретные слова — имя актёра/режиссёра, точное название.
    Опциональные фильтры:
      year_from / year_to — диапазон годов (0 = без ограничения);
      genre — точный жанр на русском (например «комедия», «драма», «фантастика»);
      min_rating — минимальный рейтинг (0 = любой);
      limit — сколько результатов вернуть (по умолчанию 5).
    Возвращает список фильмов (название, год, рейтинг, жанры). По названию фильма
    можно затем запросить детали (get_details) или похожие (find_similar).
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

    # каждая ветвь берёт пошире, RRF сливает два рейтинга в один итоговый список
    pool = max(limit * 4, 20)
    s_idx, s_val = embed_sparse_query(query)
    hits = _client().query_points(
        COLLECTION,
        prefetch=[
            Prefetch(
                query=embed_query(query), using=DENSE_NAME, filter=qfilter, limit=pool
            ),
            Prefetch(
                query=SparseVector(indices=s_idx, values=s_val),
                using=SPARSE_NAME,
                filter=qfilter,
                limit=pool,
            ),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=limit,
        with_payload=True,
    ).points
    if not hits:
        return "По запросу ничего не найдено (возможно, фильтры слишком строгие)."
    return "\n".join(_short(h.payload) for h in hits)


@tool
def get_details(title: str) -> str:
    """Полная карточка фильма по его НАЗВАНИЮ из базы.

    Используй, когда нужно подробно рассказать про конкретный фильм или ответить
    на уточняющий вопрос (режиссёр, актёры, длительность, рейтинг, описание).
    Передавай точное название фильма из результатов search_content или find_similar.
    """
    p = _resolve(title)
    if not p:
        return f"Фильм «{title}» в базе не найден."
    return _full(p.payload)


@tool
def find_similar(title: str, limit: int = 5) -> str:
    """Найти фильмы, похожие на заданный фильм (по смыслу/вектору).

    Используй для запросов «понравился вот этот — что ещё посмотреть»,
    «хочу что-то похожее». Передавай точное название фильма-образца.
    """
    p = _resolve(title)
    if not p:
        return f"Фильм «{title}» в базе не найден."
    hits = _client().query_points(
        COLLECTION,
        query=p.vector[DENSE_NAME],
        using=DENSE_NAME,
        query_filter=Filter(must_not=[HasIdCondition(has_id=[p.id])]),
        limit=limit,
        with_payload=True,
    ).points
    base = p.payload["title"]
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
    print("== search_content (гибрид) ==")
    print(search_content.invoke({"query": "напряжённое выживание в космосе", "limit": 3}))
    print("\n== get_details (по названию) ==")
    print(get_details.invoke({"title": "Дурак"}))
    print("\n== find_similar (по названию) ==")
    print(find_similar.invoke({"title": "Дурак", "limit": 3}))
