"""
Агент-консультант Find My Movie на классическом LangChain (без LangGraph).

Архитектура (как в README):
  Router (llama-3.1-8b-instant) — болтовня или запрос на поиск?
    ├─ small talk -> прямой ответ головы, без инструментов и без базы
    └─ поиск      -> AgentExecutor (gpt-oss-120b) с инструментами:
                       search_content / get_details / find_similar / web_search
  Память (ConversationBufferMemory) общая для обоих путей — держит весь диалог.

Голова и роутер — обе модели на Groq (OpenAI-совместимый API) через langchain-openai.
"""

from __future__ import annotations

import logging
import os

from dotenv import load_dotenv
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_classic.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from tools import ALL_TOOLS

load_dotenv()

GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
ROUTER_MODEL = os.getenv("ROUTER_MODEL", "llama-3.1-8b-instant")

SYSTEM = (
    "Ты — MovieAI, дружелюбный консультант по фильмам в Telegram. Помогаешь подобрать "
    "фильм, рассказываешь детали, находишь похожее. Отвечай по-русски, живо и по делу.\n\n"
    "О БАЗЕ:\n"
    "- В базе ~8000 фильмов 1995–2026 годов (с film.ru). Фильмов вне этого периода "
    "там нет — совсем старую классику (до 1995) не ищи.\n"
    "- Рейтинги обычно в диапазоне 5–7. НЕ задавай min_rating без явной просьбы "
    "пользователя — строгий фильтр часто возвращает пусто.\n\n"
    "КАК РАБОТАТЬ С ИНСТРУМЕНТАМИ:\n"
    "- Для подбора/похожего/деталей используй инструменты, не выдумывай.\n"
    "- Сначала вызывай search_content только с query, БЕЗ фильтров. Фильтры "
    "(year_from/year_to/genre/min_rating) добавляй, лишь если пользователь их явно назвал.\n"
    "- Достаточно 1–2 поисков. Получив результаты — СРАЗУ представь их пользователю "
    "списком, не повторяй поиск много раз и не перебирай параметры.\n"
    "- Если ничего не нашлось — честно скажи и предложи переформулировать запрос.\n"
    "- web_search — для АКТУАЛЬНОЙ информации о фильме из интернета: отзывы и оценки "
    "критиков, где посмотреть, свежие новости, инфо о режиссёре/актёрах. Если не знаешь "
    "фактов наверняка — не выдумывай, а сходи в web_search и возьми реальные данные.\n"
    "- ВАЖНО: подбор и рекомендации фильмов — ТОЛЬКО из нашей базы (search_content/"
    "find_similar). web_search не для поиска новых фильмов, а для деталей о тех, что "
    "уже есть в базе. Не предлагай пользователю фильмы, которых нет в базе.\n\n"
    "КАК ОФОРМЛЯТЬ ОТВЕТ (это Telegram — таблицы и markdown НЕ поддерживаются):\n"
    "- НИКОГДА не используй таблицы и символ | , не используй ** и * для выделения "
    "(они показываются как обычные символы). Пиши простым текстом с эмодзи.\n"
    "- Каждый фильм — отдельным блоком, между блоками пустая строка:\n"
    "  🎬 Название (год)\n"
    "  ⭐ рейтинг · жанр1, жанр2\n"
    "- ID фильма — ВНУТРЕННИЙ, только для вызова инструментов get_details/find_similar. "
    "НИКОГДА не показывай ID пользователю и не пиши его в ответе.\n"
    "- Перед списком — короткая фраза-подводка (1 строка), после — короткий вопрос/CTA.\n"
    "- Без воды и канцелярита, живо и дружелюбно.\n\n"
    "ОБЩЕЕ:\n"
    "- Опирайся только на данные инструментов. В ответах упоминай год и жанры.\n"
    "- Учитывай предпочтения и отказы пользователя из истории диалога."
)
SMALLTALK_SYSTEM = (
    "Ты — MovieAI, дружелюбный ассистент в Telegram. Сейчас обычный разговор: "
    "приветствие, болтовня, благодарность, общие вопросы. Отвечай естественно, "
    "по-человечески и коротко — да, можешь просто поболтать.\n"
    "Ненавязчиво, одной короткой фразой, напомни, что твоя сильная сторона — помочь "
    "подобрать фильм, рассказать детали или найти похожее, если захочется. "
    "Без напора и не в каждом сообщении — мягко."
)


def _llm(model: str, temperature: float) -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        base_url=GROQ_BASE_URL,
        api_key=GROQ_API_KEY,
        temperature=temperature,
    )


llm = _llm(LLM_MODEL, 0.3)
router_llm = _llm(ROUTER_MODEL, 0.0)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# --- Router: одно слово на выходе -------------------------------------------
_router_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Классифицируй сообщение пользователя ОДНИМ словом:\n"
            "- 'search' — хочет найти/подобрать фильм, спрашивает детали, похожее, "
            "отзывы, рейтинг, режиссёра и т.п.;\n"
            "- 'smalltalk' — приветствие, болтовня, благодарность, общий разговор "
            "без запроса про конкретные фильмы.\n"
            "Ответь только одним словом: search или smalltalk.",
        ),
        ("human", "{input}"),
    ]
)
_router = _router_prompt | router_llm | StrOutputParser()

# --- Agent: tool-calling + память -------------------------------------------
_agent_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ]
)
# Сам runnable-агент без состояния — общий; память живёт в executor (см. ниже).
_agent = create_tool_calling_agent(llm, ALL_TOOLS, _agent_prompt)


def _new_memory() -> ConversationBufferMemory:
    return ConversationBufferMemory(memory_key="chat_history", return_messages=True)


_log = logging.getLogger("findmymovie.agent")


def _new_executor(mem: ConversationBufferMemory) -> AgentExecutor:
    return AgentExecutor(
        agent=_agent,
        tools=ALL_TOOLS,
        memory=mem,
        verbose=False,
        handle_parsing_errors=True,
        max_iterations=3,
        return_intermediate_steps=True,
    )


def route(user_input: str) -> str:
    """'smalltalk' или 'search' (по умолчанию search — лучше лишний раз поискать)."""
    try:
        raw = _router.invoke({"input": user_input}).strip().lower()
        return "smalltalk" if "smalltalk" in raw else "search"
    except Exception:
        return "search"


def _respond_with(user_input: str, mem: ConversationBufferMemory, executor: AgentExecutor) -> str:
    """Маршрутизирует и отвечает поверх ПЕРЕДАННОЙ памяти (общая на оба пути)."""
    if route(user_input) == "smalltalk":
        _log.info("ветка=smalltalk инструменты: (нет)")
        history = mem.load_memory_variables({})["chat_history"]
        msgs = [SystemMessage(content=SMALLTALK_SYSTEM), *history, HumanMessage(content=user_input)]
        reply = llm.invoke(msgs).content
        mem.save_context({"input": user_input}, {"output": reply})
        return reply
    result = executor.invoke({"input": user_input})
    used = [action.tool for action, _ in result.get("intermediate_steps", [])]
    _log.info("ветка=search инструменты: %s", ", ".join(used) if used else "(ни одного)")
    return result["output"]


class Session:
    """Изолированный диалог: своя память + executor. Один на чат/пользователя."""

    def __init__(self) -> None:
        self.memory = _new_memory()
        self.executor = _new_executor(self.memory)

    def respond(self, user_input: str) -> str:
        return _respond_with(user_input, self.memory, self.executor)

    def clear(self) -> None:
        self.memory.clear()


# --- Глобальная сессия для одиночных интерфейсов (app.py, ui.py) -------------
memory = _new_memory()
agent_executor = _new_executor(memory)


def respond(user_input: str) -> str:
    """Главная точка для одиночного диалога (общая глобальная память)."""
    return _respond_with(user_input, memory, agent_executor)
