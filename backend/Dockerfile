# Использовать официальный образ Python как базовый
FROM python:3.11-slim AS builder

# Установить менеджер пакетов uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Установить рабочую директорию
WORKDIR /app

# Скопировать файлы зависимостей бэкенда
COPY pyproject.toml uv.lock ./

# Установить зависимости проекта без самого приложения в режиме frozen
RUN uv sync --frozen --no-cache --no-install-project

# --- Финальный легковесный образ ---
FROM python:3.11-slim

WORKDIR /app

# Копируем виртуальное окружение со всеми установленными библиотеками
COPY --from=builder /app/.venv /app/.venv
# Копируем исходный код бэкенда и папку с векторной базой данных
COPY . /app

# Добавляем виртуальное окружение в PATH, чтобы команды запускались корректно
ENV PATH="/app/.venv/bin:$PATH"
ENV PORT=8000

# Открываем порт для FastAPI
EXPOSE 8000

# Запускаем FastAPI через uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
