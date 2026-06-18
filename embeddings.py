"""
Общий слой эмбеддингов для проекта Find My Movie.

Эмбеддер — multilingual-e5-large через fastembed (локально, без ключа).
e5 требует префиксы: документы кодируются как 'passage: ...', запросы как 'query: ...'.
Один источник правды и для indexer.py, и для tools.py — чтобы не разъехались.
"""

from __future__ import annotations

import os

# Фикс загрузки больших ONNX-моделей (e5-large >2ГБ хранит веса в model.onnx_data):
# новый onnxruntime отказывается читать external data через симлинки HF-кэша.
# Реальные файлы вместо симлинков снимают проблему. Ставим ДО импорта fastembed.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")

from functools import lru_cache  # noqa: E402

from fastembed import TextEmbedding  # noqa: E402

EMBED_MODEL = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-large")
VECTOR_SIZE = 1024  # размерность multilingual-e5-large
DB_PATH = os.path.join("data", "db")
COLLECTION = "movies"

# Локальная распакованная модель (tar с Google CDN). Если папка есть — грузим из неё
# (плоские файлы, без симлинков). Если нет — fastembed скачает сам (см. фикс выше).
LOCAL_MODEL_PATH = os.getenv("EMBED_LOCAL_PATH", "models/fast-multilingual-e5-large")


@lru_cache(maxsize=1)
def _model() -> TextEmbedding:
    """Ленивая загрузка модели. Локальная папка -> без скачивания; иначе скачает ~2.2 ГБ."""
    if os.path.isdir(LOCAL_MODEL_PATH):
        return TextEmbedding(model_name=EMBED_MODEL, specific_model_path=LOCAL_MODEL_PATH)
    return TextEmbedding(model_name=EMBED_MODEL)


def embed_passages(texts: list[str]) -> list[list[float]]:
    """Векторы для документов (с префиксом 'passage: ')."""
    docs = [f"passage: {t}" for t in texts]
    return [v.tolist() for v in _model().embed(docs)]


def embed_query(text: str) -> list[float]:
    """Вектор для поискового запроса (с префиксом 'query: ')."""
    return next(iter(_model().embed([f"query: {text}"]))).tolist()
