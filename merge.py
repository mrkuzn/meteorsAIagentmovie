"""
Объединение датасетов команды в общий data/raw.csv.

Читает все файлы data/raw*.csv (наш raw.csv + куски друзей: rawv/rawd/rawr/...),
склеивает, выкидывает дубли по url и пишет обратно в data/raw.csv.

Дедуп идёт по url — он уникален для фильма. Идемпотентно: повторный запуск
даёт тот же результат (в raw.csv уже лежит объединение).

Запуск:
    uv run python merge.py
Потом переиндексировать базу:
    uv run python indexer.py
"""

from __future__ import annotations

import csv
import glob
import os

DATA_DIR = "data"
OUT = os.path.join(DATA_DIR, "raw.csv")
FIELDS = [
    "title", "original_title", "year", "genres", "director", "cast",
    "country", "rating", "num_ratings", "duration_min", "description",
    "poster_url", "url",
]


def main() -> None:
    files = sorted(glob.glob(os.path.join(DATA_DIR, "raw*.csv")))
    if not files:
        print("Не найдено файлов data/raw*.csv")
        return

    by_url: dict[str, dict] = {}
    dups = 0
    for path in files:
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        print(f"{os.path.basename(path):12} записей: {len(rows)}")
        for r in rows:
            url = (r.get("url") or "").strip()
            if not url:
                continue
            if url in by_url:
                dups += 1
            else:
                by_url[url] = r

    merged = list(by_url.values())
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(merged)

    years: dict[str, int] = {}
    for r in merged:
        years[r["year"]] = years.get(r["year"], 0) + 1
    print(f"\nДублей по url отброшено: {dups}")
    print(f"Итого уникальных фильмов: {len(merged)} -> {OUT}")
    print("По годам:", dict(sorted(years.items())))


if __name__ == "__main__":
    main()
