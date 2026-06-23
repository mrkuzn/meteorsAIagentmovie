"""
Индексация фильмов из data/raw.csv в локальный Qdrant.

Для каждого фильма:
  - собираем осмысленный текст -> эмбеддинг e5-large (passage:) -> вектор
  - кладём вектор + метаданные (payload) в коллекцию 'movies'

Метаданные нормализуем по типам (year:int, rating:float, genres:list, ...),
чтобы по ним работала фильтрация в search_content.

Запуск:
    uv run python indexer.py            # переиндексировать всё
    uv run python indexer.py --check    # только проверить, что в базе (без записи)
"""

from __future__ import annotations

import csv
import sys

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    Modifier,
    PointStruct,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)
from tqdm import tqdm

from backend.embeddings import (
    COLLECTION,
    DB_PATH,
    DENSE_NAME,
    EMBED_MODEL,
    SPARSE_NAME,
    VECTOR_SIZE,
    embed_passages,
    embed_query,
    embed_sparse_passages,
)

RAW_CSV = "data/raw.csv"
BATCH = 128


# ------------------------------------------------------------- нормализация ---
def as_list(value: str) -> list[str]:
    """'драма; комедия' -> ['драма', 'комедия']."""
    return [x.strip() for x in (value or "").split(";") if x.strip()]


def as_int(value: str):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def as_float(value: str):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_text(row: dict) -> str:
    """Текст для эмбеддинга: то, по чему ищем СМЫСЛ (название + жанры + описание...)."""
    parts = [row["title"]]
    if row["original_title"] and row["original_title"] != row["title"]:
        parts.append(row["original_title"])
    if row["year"]:
        parts.append(f"{row['year']} год")
    if row["genres"]:
        parts.append("Жанры: " + row["genres"])
    if row["director"]:
        parts.append("Режиссёр: " + row["director"])
    if row["cast"]:
        parts.append("В ролях: " + row["cast"])
    if row["country"]:
        parts.append("Страна: " + row["country"])
    if row["description"]:
        parts.append(row["description"])
    return ". ".join(parts)


def build_payload(idx: int, row: dict) -> dict:
    """Всё, что нужно для get_details и фильтрации — в payload."""
    return {
        "id": idx,
        "title": row["title"],
        "original_title": row["original_title"],
        "year": as_int(row["year"]),
        "genres": as_list(row["genres"]),
        "director": as_list(row["director"]),
        "cast": as_list(row["cast"]),
        "country": as_list(row["country"]),
        "rating": as_float(row["rating"]),
        "num_ratings": as_int(row["num_ratings"]),
        "duration_min": as_int(row["duration_min"]),
        "description": row["description"],
        "poster_url": row["poster_url"],
        "url": row["url"],
    }


# --------------------------------------------------------------- индексация ---
def index() -> None:
    with open(RAW_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"Фильмов к индексации: {len(rows)}")
    print(f"Эмбеддер: {EMBED_MODEL} (dim={VECTOR_SIZE})")

    client = QdrantClient(path=DB_PATH)
    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
    client.create_collection(
        COLLECTION,
        # гибрид: плотный e5 (косинус) + разреженный BM25 (IDF считает Qdrant)
        vectors_config={DENSE_NAME: VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)},
        sparse_vectors_config={SPARSE_NAME: SparseVectorParams(modifier=Modifier.IDF)},
    )

    for start in tqdm(range(0, len(rows), BATCH), desc="Индексация", unit="batch"):
        chunk = rows[start : start + BATCH]
        texts = [build_text(r) for r in chunk]
        dense = embed_passages(texts)
        sparse = embed_sparse_passages(texts)
        points = [
            PointStruct(
                id=start + i,
                vector={
                    DENSE_NAME: dvec,
                    SPARSE_NAME: SparseVector(indices=idx, values=val),
                },
                payload=build_payload(start + i, row),
            )
            for i, (row, dvec, (idx, val)) in enumerate(zip(chunk, dense, sparse))
        ]
        client.upsert(COLLECTION, points=points)

    total = client.count(COLLECTION).count
    print(f"Готово: в коллекции '{COLLECTION}' {total} фильмов -> {DB_PATH}/")
    smoke_test(client)


def smoke_test(client: QdrantClient) -> None:
    """Быстрая проверка, что гибридный поиск (dense + BM25) реально работает."""
    from qdrant_client.models import Fusion, FusionQuery, Prefetch, SparseVector

    from backend.embeddings import embed_sparse_query

    queries = [
        "напряжённое выживание в космосе",
        "лёгкая романтическая комедия на вечер",
    ]
    print("\n--- smoke-test гибридного поиска (dense + BM25, RRF) ---")
    for q in queries:
        s_idx, s_val = embed_sparse_query(q)
        hits = client.query_points(
            COLLECTION,
            prefetch=[
                Prefetch(query=embed_query(q), using=DENSE_NAME, limit=20),
                Prefetch(
                    query=SparseVector(indices=s_idx, values=s_val),
                    using=SPARSE_NAME,
                    limit=20,
                ),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=3,
            with_payload=True,
        ).points
        print(f"\nЗапрос: «{q}»")
        for h in hits:
            p = h.payload
            print(f"  {h.score:.3f}  {p['title']} ({p['year']}) — {', '.join(p['genres'])}")


def check() -> None:
    client = QdrantClient(path=DB_PATH)
    if not client.collection_exists(COLLECTION):
        print(f"Коллекции '{COLLECTION}' нет — сначала запусти индексацию.")
        return
    print(f"В коллекции '{COLLECTION}': {client.count(COLLECTION).count} фильмов")
    smoke_test(client)


if __name__ == "__main__":
    if "--check" in sys.argv:
        check()
    else:
        index()
