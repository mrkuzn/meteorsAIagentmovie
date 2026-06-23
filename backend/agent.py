from __future__ import annotations

import os
from typing import Annotated, Literal
from typing_extensions import TypedDict

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from backend.tools import ALL_TOOLS

load_dotenv()

# --- Настройки провайдера Polza.ai ---
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://polza.ai")
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek/deepseek-v4-flash")
ROUTER_MODEL = os.getenv("ROUTER_MODEL", "deepseek/deepseek-v4-flash")

# --- 1. Определение состояния графа ---
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    route: str  # Поле для хранения вердикта роутера

# --- 2. Инициализация моделей ---
router_llm = ChatOpenAI(
    base_url=LLM_BASE_URL, 
    api_key=LLM_API_KEY, 
    model=ROUTER_MODEL, 
    temperature=0.0
)

agent_llm = ChatOpenAI(
    base_url=LLM_BASE_URL, 
    api_key=LLM_API_KEY, 
    model=LLM_MODEL, 
    temperature=0.1
).bind_tools(ALL_TOOLS)

smalltalk_llm = ChatOpenAI(
    base_url=LLM_BASE_URL, 
    api_key=LLM_API_KEY, 
    model=LLM_MODEL, 
    temperature=0.5
)

# --- 3. Системные промпты ---
SMALLTALK_SYSTEM = SystemMessage(
    content=(
        "Ты – MovieAI, дружелюбный ассистент в Telegram. Сейчас обычный разговор: "
        "приветствие, болтовня, благодарность, общие вопросы. Отвечай естественно, "
        "по-человечески и коротко – да, можете просто поболтать.\n"
        "Ненавязчиво, одной короткой фразой, напомни, что твоя сильная сторона – помочь "
        "подобрать фильм, рассказать детали или найти похожее, если захочется. "
        "Без напора и не в каждом сообщении – мягко.\n"
        "ВАЖНО: здесь у тебя НЕТ доступа к базе фильмов. Поэтому НИКОГДА не называй "
        "конкретные фильмы, годы, рейтинги и не советуй кино по памяти – ты можешь "
        "ошибиться. Если пользователь спрашивает что-то про фильмы/подбор/оценки – не "
        "выдумывай ответ, а предложи: «давай подберу из базы – уточни, что хочется»."
    )
)

MOVIE_SYSTEM = SystemMessage(
    content=(
        "Ты — MovieAI, дружелюбный консультант по фильмам в Telegram. Отвечай по-русски, "
        "живо и по делу.\n\n"
        "БАЗА: ~8000 фильмов 1995–2026 (film.ru), вне этого периода ничего нет. Рейтинги "
        "обычно 5–7, выше 8 почти не бывает.\n\n"
        "ГЛАВНОЕ ПРАВИЛО: ты НЕ знаешь фильмы из головы. Любой факт (название, год, жанр, "
        "рейтинг, режиссёр, актёры, сюжет) бери ТОЛЬКО из ответов инструментов – никогда по "
        "памяти. Нет данных в ответе инструмента -> честно скажи «не нашёл», не выдумывай. "
        "Рекомендуй только фильмы из результатов поиска.\n\n"
        "ИНСТРУМЕНТЫ (всегда вызывай, не отвечай по памяти):\n"
        "- search_content – подбор по описанию/настроению/жанру. Сначала только query; "
        "year/genre/min_rating добавляй, лишь если пользователь явно их назвал. Про «высокие "
        "оценки» ставь min_rating не больше 7; если пусто – повтори без min_rating.\n"
        "- get_details(название) – детали конкретного фильма (режиссёр, актёры, длительность, "
        "сюжет) и для сравнения нескольких (возьми детали каждого и сопоставь).\n"
        "- find_similar(название) – похожее на понравившийся фильм.\n"
        "- web_search – отзывы/оценки критиков, где посмотреть, новости; НЕ для подбора новых "
        "фильмов, только доп. инфа о тех, что в базе.\n"
        "Хватает 1–2 поисков: получил результаты – сразу покажи списком, не ищи по кругу.\n\n"
        "ФОРМАТ (Telegram, без markdown): не используй таблицы, | , ** и * НИГДЕ, включая "
        "ответы с отзывами и данными из web_search. Каждый фильм — "
        "блоком, между блоками пустая строка:\n"
        "🎬 Название (год)\n"
        "⭐️ рейтинг · жанр1, жанр2\n"
        "Ссылайся на фильмы по названию (без ID). Перед списком — строка-подводка, после — "
        "короткий вопрос. Живо, без воды.\n\n"
        "ПАМЯТЬ: помни весь диалог и вкусы пользователя. Если сказал «это не моё» или что-то "
        "не любит – больше такое не предлагай. Учитывай прошлые предпочтения в новых советах."
    )
)

# --- 4. Определение узлов графа (Nodes) ---

def router_node(state: AgentState):
    """Узел роутера: классифицирует запрос пользователя с учетом истории."""
    # Форматируем историю диалога (все сообщения, кроме последнего входящего)
    history_messages = state["messages"][:-1]
    formatted_history = ""
    for msg in history_messages:
        role = "Пользователь" if isinstance(msg, HumanMessage) else "Ассистент"
        formatted_history += f"{role}: {msg.content}\n"
        
    if not formatted_history:
        formatted_history = "История пуста."

    current_input = state["messages"][-1].content

    router_prompt_text = (
        "Ты классифицируешь сообщение пользователя в чате про подбор фильмов.\n"
        "Ответь ОДНИМ словом: search или smalltalk.\n"
        "- 'search' – хочет найти/подобрать фильм, спрашивает детали, похожее, "
        "отзывы, рейтинг, режиссёра, или уточняет/сужает предыдущий запрос про "
        "фильмы (например: «оценку больше 8», «только комедии», «а подлиннее», "
        "«ещё такие же», «за 2020 год»). Короткие уточнения после разговора о "
        "фильмах — это ВСЕГДА search.\n"
        "- 'smalltalk' – ТОЛЬКО приветствие, благодарность, болтовня о жизни, не "
        "связанная с фильмами/подбором.\n"
        "Если сомневаешься – отвечай search.\n\n"
        "Ниже последние реплики диалога для контекста (могут быть пустыми):\n"
        f"{formatted_history}"
    )

    messages = [
        SystemMessage(content=router_prompt_text),
        HumanMessage(content=current_input)
    ]
    
    response = router_llm.invoke(messages)
    decision = response.content.strip().lower()

    # Маппим ответ на имена наших узлов графа
    return {"route": "movie" if "search" in decision else "smalltalk"}

def smalltalk_node(state: AgentState):
    """Узел для ведения обычной светской беседы."""
    messages = [SMALLTALK_SYSTEM] + state["messages"]
    response = smalltalk_llm.invoke(messages)
    return {"messages": [response]}

def movie_node(state: AgentState):
    """Основной узел агента для работы с базой и инструментами."""
    messages = [MOVIE_SYSTEM] + state["messages"]
    response = agent_llm.invoke(messages)
    return {"messages": [response]}

# --- 5. Логика условных переходов (Conditional Edges) ---

def route_initial(state: AgentState) -> Literal["smalltalk", "movie"]:
    """Направляет из роутера в целевой узел общения."""
    return state["route"]

def route_after_movie(state: AgentState) -> Literal["tools", END]:
    """Проверяет, сгенерировала ли модель вызовы инструментов."""
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "tools"
    return END

# --- 6. Сборка и компиляция графа ---
workflow = StateGraph(AgentState)

# Регистрация узлов
workflow.add_node("router", router_node)
workflow.add_node("smalltalk", smalltalk_node)
workflow.add_node("movie", movie_node)
workflow.add_node("tools", ToolNode(ALL_TOOLS))

# Настройка граней и логики переходов
workflow.add_edge(START, "router")
workflow.add_conditional_edges("router", route_initial)
workflow.add_conditional_edges("movie", route_after_movie)
workflow.add_edge("tools", "movie")
workflow.add_edge("smalltalk", END)

# Подключение встроенной оперативной памяти для хранения контекста сессии
memory = MemorySaver()
compiled_agent = workflow.compile(checkpointer=memory)

