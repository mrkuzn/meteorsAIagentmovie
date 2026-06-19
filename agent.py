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
# gpt-oss: low/medium/high. low = меньше "размышлений" -> быстрее и дешевле по токенам.
REASONING_EFFORT = os.getenv("REASONING_EFFORT", "low")

SYSTEM = (
    "Ты — MovieAI, дружелюбный консультант по фильмам в Telegram. Отвечай по-русски, "
    "живо и по делу.\n\n"
    "БАЗА: ~8000 фильмов 1995–2026 (film.ru), вне этого периода ничего нет. Рейтинги "
    "обычно 5–7, выше 8 почти не бывает.\n\n"
    "ГЛАВНОЕ ПРАВИЛО: ты НЕ знаешь фильмы из головы. Любой факт (название, год, жанр, "
    "рейтинг, режиссёр, актёры, сюжет) бери ТОЛЬКО из ответов инструментов — никогда по "
    "памяти. Нет данных в ответе инструмента → честно скажи «не нашёл», не выдумывай. "
    "Рекомендуй только фильмы из результатов поиска.\n\n"
    "ИНСТРУМЕНТЫ (всегда вызывай, не отвечай по памяти):\n"
    "- search_content — подбор по описанию/настроению/жанру. Сначала только query; "
    "year/genre/min_rating добавляй, лишь если пользователь явно их назвал. Про «высокие "
    "оценки» ставь min_rating не больше 7; если пусто — повтори без min_rating.\n"
    "- get_details(название) — детали конкретного фильма (режиссёр, актёры, длительность, "
    "сюжет) и для сравнения нескольких (возьми детали каждого и сопоставь).\n"
    "- find_similar(название) — похожее на понравившийся фильм.\n"
    "- web_search — отзывы/оценки критиков, где посмотреть, новости; НЕ для подбора новых "
    "фильмов, только доп. инфа о тех, что в базе.\n"
    "Хватает 1–2 поисков: получил результаты — сразу покажи списком, не ищи по кругу.\n\n"
    "ФОРМАТ (Telegram, без markdown): не используй таблицы, | , ** и * НИГДЕ, включая "
    "ответы с отзывами и данными из web_search. Каждый фильм — "
    "блоком, между блоками пустая строка:\n"
    "  🎬 Название (год)\n"
    "  ⭐ рейтинг · жанр1, жанр2\n"
    "Ссылайся на фильмы по названию (без ID). Перед списком — строка-подводка, после — "
    "короткий вопрос. Живо, без воды.\n\n"
    "ПАМЯТЬ: помни весь диалог и вкусы пользователя. Если сказал «это не моё» или что-то "
    "не любит — больше такое не предлагай. Учитывай прошлые предпочтения в новых советах."
)
SMALLTALK_SYSTEM = (
    "Ты — MovieAI, дружелюбный ассистент в Telegram. Сейчас обычный разговор: "
    "приветствие, болтовня, благодарность, общие вопросы. Отвечай естественно, "
    "по-человечески и коротко — да, можешь просто поболтать.\n"
    "Ненавязчиво, одной короткой фразой, напомни, что твоя сильная сторона — помочь "
    "подобрать фильм, рассказать детали или найти похожее, если захочется. "
    "Без напора и не в каждом сообщении — мягко.\n"
    "ВАЖНО: здесь у тебя НЕТ доступа к базе фильмов. Поэтому НИКОГДА не называй "
    "конкретные фильмы, годы, рейтинги и не советуй кино по памяти — ты можешь "
    "ошибиться. Если пользователь спрашивает что-то про фильмы/подбор/оценки — не "
    "выдумывай ответ, а предложи: «давай подберу из базы — уточни, что хочется»."
)


def _llm(model: str, temperature: float, reasoning_effort: str | None = None) -> ChatOpenAI:
    extra = {"model_kwargs": {"reasoning_effort": reasoning_effort}} if reasoning_effort else {}
    return ChatOpenAI(
        model=model,
        base_url=GROQ_BASE_URL,
        api_key=GROQ_API_KEY,
        temperature=temperature,
        **extra,
    )


# голова — с пониженным reasoning_effort (быстрее); роутер (llama) этот параметр не поддерживает
llm = _llm(LLM_MODEL, 0.3, reasoning_effort=REASONING_EFFORT)
router_llm = _llm(ROUTER_MODEL, 0.0)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# --- Router: одно слово на выходе -------------------------------------------
_router_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Ты классифицируешь сообщение пользователя в чате про подбор фильмов. "
            "Ответь ОДНИМ словом: search или smalltalk.\n"
            "- 'search' — хочет найти/подобрать фильм, спрашивает детали, похожее, "
            "отзывы, рейтинг, режиссёра, ИЛИ уточняет/сужает предыдущий запрос про "
            "фильмы (например: «оценку больше 8», «только комедии», «а подлиннее», "
            "«ещё такие же», «за 2020 год»). Короткие уточнения после разговора о "
            "фильмах — это ВСЕГДА search.\n"
            "- 'smalltalk' — ТОЛЬКО приветствие, благодарность, болтовня о жизни, не "
            "связанная с фильмами/подбором.\n"
            "Если сомневаешься — отвечай search.\n"
            "Ниже последние реплики диалога для контекста (могут быть пустыми):\n"
            "{history}",
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
        max_iterations=5,
        return_intermediate_steps=True,
    )


def _history_text(mem: ConversationBufferMemory, last: int = 4) -> str:
    """Короткая выжимка последних реплик диалога для контекста роутера."""
    msgs = mem.load_memory_variables({})["chat_history"][-last:]
    lines = []
    for m in msgs:
        role = "Пользователь" if m.type == "human" else "Бот"
        lines.append(f"{role}: {m.content[:160]}")
    return "\n".join(lines)


def route(user_input: str, history: str = "") -> str:
    """'smalltalk' или 'search' (по умолчанию search — лучше лишний раз поискать)."""
    try:
        raw = _router.invoke({"input": user_input, "history": history}).strip().lower()
        return "smalltalk" if "smalltalk" in raw else "search"
    except Exception:
        return "search"


def _respond_with(user_input: str, mem: ConversationBufferMemory, executor: AgentExecutor) -> str:
    """Маршрутизирует и отвечает поверх ПЕРЕДАННОЙ памяти (общая на оба пути)."""
    if route(user_input, _history_text(mem)) == "smalltalk":
        _log.info("ветка=smalltalk инструменты: (нет)")
        history = mem.load_memory_variables({})["chat_history"]
        msgs = [SystemMessage(content=SMALLTALK_SYSTEM), *history, HumanMessage(content=user_input)]
        reply = llm.invoke(msgs).content
        mem.save_context({"input": user_input}, {"output": reply})
        return reply
    result = executor.invoke({"input": user_input})
    used = [action.tool for action, _ in result.get("intermediate_steps", [])]
    _log.info("ветка=search инструменты: %s", ", ".join(used) if used else "(ни одного)")
    output = result["output"]
    if output.strip().lower().startswith("agent stopped"):
        _log.info("упёрлись в max_iterations — отдаём дружелюбный fallback")
        return (
            "Кажется, я перемудрил с поиском 😅 Попробуй сформулировать чуть проще — "
            "например, без жёстких условий по рейтингу, и я подберу варианты."
        )
    return output


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
