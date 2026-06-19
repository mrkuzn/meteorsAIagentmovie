"""
Объективное сравнение ретриверов на КУСКЕ базы: dense-only vs hybrid (dense+BM25, RRF).

Зачем: решить, стоит ли гибрид реиндекса полной базы, или вернуться к чистой семантике.
Полную базу не трогаем — строим временную из случайной выборки фильмов из data/raw.csv.

Запросы генерируются автоматически из самих фильмов (без ручной разметки):
  - title : запрос = точное название фильма          -> проверяет ЛЕКСИКУ (точные слова)
  - desc  : запрос = кусок описания (без названия)    -> проверяет СЕМАНТИКУ (смысл)
  - cast  : запрос = имя актёра, уникального в выборке -> проверяет ЛЕКСИКУ (имена)
Для каждого запроса известен «золотой» фильм (из которого он построен). Считаем,
на каком месте ретривер его вернул: hit@5 (попал ли в топ-5) и MRR (1/ранг).

Запуск:
    uv run python eval_retriever.py            # выборка 400 фильмов
    uv run python eval_retriever.py 800 60     # 800 фильмов, по 60 запросов на тип
"""

from __future__ import annotations

import csv
import random
import sys
import tempfile

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    Fusion,
    FusionQuery,
    Modifier,
    Prefetch,
    PointStruct,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from embeddings import (
    COLLECTION,
    DENSE_NAME,
    SPARSE_NAME,
    VECTOR_SIZE,
    embed_passages,
    embed_query,
    embed_sparse_passages,
    embed_sparse_query,
)
from indexer import RAW_CSV, build_payload, build_text

K = 5
SEED = 42


def build_index(rows: list[dict]) -> QdrantClient:
    tmp = tempfile.mkdtemp(prefix="eval_db_")
    client = QdrantClient(path=tmp)
    client.create_collection(
        COLLECTION,
        vectors_config={DENSE_NAME: VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)},
        sparse_vectors_config={SPARSE_NAME: SparseVectorParams(modifier=Modifier.IDF)},
    )
    texts = [build_text(r) for r in rows]
    print(f"Эмбеддинг {len(rows)} фильмов (e5 + BM25)…", flush=True)
    dense = embed_passages(texts)
    sparse = embed_sparse_passages(texts)
    points = [
        PointStruct(
            id=i,
            vector={DENSE_NAME: d, SPARSE_NAME: SparseVector(indices=ix, values=v)},
            payload=build_payload(i, r),
        )
        for i, (r, d, (ix, v)) in enumerate(zip(rows, dense, sparse))
    ]
    client.upsert(COLLECTION, points=points)
    return client


# --------------------------------------------------------------- ретриверы ---
def search_dense(client: QdrantClient, query: str) -> list[int]:
    hits = client.query_points(
        COLLECTION, query=embed_query(query), using=DENSE_NAME, limit=K, with_payload=False
    ).points
    return [h.id for h in hits]


def search_hybrid(client: QdrantClient, query: str) -> list[int]:
    s_idx, s_val = embed_sparse_query(query)
    hits = client.query_points(
        COLLECTION,
        prefetch=[
            Prefetch(query=embed_query(query), using=DENSE_NAME, limit=20),
            Prefetch(query=SparseVector(indices=s_idx, values=s_val), using=SPARSE_NAME, limit=20),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=K,
        with_payload=False,
    ).points
    return [h.id for h in hits]


# ------------------------------------------------------------ набор запросов ---
def make_queries(rows: list[dict], per_type: int) -> list[tuple[str, str, int]]:
    """(тип, текст_запроса, gold_id). gold_id == индекс фильма в rows (== point id)."""
    rng = random.Random(SEED)
    queries: list[tuple[str, str, int]] = []

    # title: точное название
    have_title = [i for i, r in enumerate(rows) if r["title"].strip()]
    for i in rng.sample(have_title, min(per_type, len(have_title))):
        queries.append(("title", rows[i]["title"].strip(), i))

    # desc: кусок описания из середины (без названия) — длиной хватает на смысл
    have_desc = [i for i, r in enumerate(rows) if len(r["description"] or "") >= 200]
    for i in rng.sample(have_desc, min(per_type, len(have_desc))):
        d = rows[i]["description"].strip()
        snippet = d[60:260]  # из середины, чтобы не зацепить название в начале
        queries.append(("desc", snippet, i))

    # cast: имя актёра, встречающегося в выборке ровно один раз -> однозначный фильм
    from collections import Counter

    first_cast = {}
    for i, r in enumerate(rows):
        names = [x.strip() for x in (r["cast"] or "").split(";") if x.strip()]
        if names:
            first_cast[i] = names[0]
    counts = Counter(first_cast.values())
    unique = [i for i, name in first_cast.items() if counts[name] == 1]
    for i in rng.sample(unique, min(per_type, len(unique))):
        queries.append(("cast", f"фильм с актёром {first_cast[i]}", i))

    return queries


# ----------------------------------------------------------------- метрики ---
def evaluate(client: QdrantClient, queries: list[tuple[str, str, int]]) -> None:
    methods = {"dense-only": search_dense, "hybrid (RRF)": search_hybrid}
    types = ["title", "desc", "cast"]
    # stats[method][type] = [hits, rr_sum, n]
    stats = {m: {t: [0, 0.0, 0] for t in types} for m in methods}

    for qtype, text, gold in queries:
        for mname, fn in methods.items():
            ids = fn(client, text)
            rank = ids.index(gold) + 1 if gold in ids else 0
            s = stats[mname][qtype]
            s[0] += 1 if rank else 0
            s[1] += (1.0 / rank) if rank else 0.0
            s[2] += 1

    print(f"\n=== Результаты (top-{K}, выборка {client.count(COLLECTION).count} фильмов) ===\n")
    header = f"{'тип запроса':<14}{'n':>4}   " + "".join(
        f"{m:>26}" for m in methods
    )
    print(header)
    print(f"{'':14}{'':>4}   " + "".join(f"{'hit@5 / MRR':>26}" for _ in methods))
    print("-" * len(header))
    for t in types:
        n = stats["dense-only"][t][2]
        if not n:
            continue
        row = f"{t:<14}{n:>4}   "
        for m in methods:
            hits, rr, nn = stats[m][t]
            row += f"{hits / nn:>11.2f} /{rr / nn:>11.3f}   "[:26].rjust(26)
        print(row)
    # overall
    print("-" * len(header))
    row = f"{'ВСЕГО':<14}"
    total_n = sum(stats["dense-only"][t][2] for t in types)
    row += f"{total_n:>4}   "
    for m in methods:
        h = sum(stats[m][t][0] for t in types)
        rr = sum(stats[m][t][1] for t in types)
        n = sum(stats[m][t][2] for t in types)
        row += f"{h / n:>11.2f} /{rr / n:>11.3f}   "[:26].rjust(26)
    print(row)
    print("\nhit@5 — доля запросов, где нужный фильм попал в топ-5 (выше = лучше)")
    print("MRR   — средний обратный ранг нужного фильма (выше = лучше)")


def main() -> None:
    n_films = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    per_type = int(sys.argv[2]) if len(sys.argv) > 2 else 40

    with open(RAW_CSV, newline="", encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))
    rng = random.Random(SEED)
    rows = rng.sample(all_rows, min(n_films, len(all_rows)))

    client = build_index(rows)
    queries = make_queries(rows, per_type)
    by_type = {}
    for t, _, _ in queries:
        by_type[t] = by_type.get(t, 0) + 1
    print(f"Запросов: {by_type}")
    evaluate(client, queries)


if __name__ == "__main__":
    main()
