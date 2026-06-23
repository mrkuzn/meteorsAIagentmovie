import os
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

# Импортируем скомпилированного агента из нашего нового agent.py
from backend.agent import compiled_agent

load_dotenv()

app = FastAPI(
    title="MovieAI API",
    description="FastAPI Backend для HR-агента по фильмам на базе LangGraph",
    version="1.0.0"
)

# Описываем структуру входящего запроса от Streamlit
class ChatRequest(BaseModel):
    message: str
    session_id: str = "default_user"  # Используется в качестве thread_id для памяти графа

# Описываем структуру ответа сервера
class ChatResponse(BaseModel):
    response: str

@app.get("/")
def read_root():
    """Эндпоинт для проверки работоспособности сервера."""
    return {"status": "healthy", "agent": "MovieAI LangGraph Server"}

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Основной эндпоинт для отправки сообщений агенту."""
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Сообщение не может быть пустым")

    try:
        # Конфигурируем thread_id, чтобы MemorySaver внутри LangGraph 
        # автоматически подгружал историю именно этого пользователя
        config = {"configurable": {"thread_id": request.session_id}}
        
        # Передаем новое сообщение пользователя в граф
        input_data = {"messages": [HumanMessage(content=request.message)]}
        
        # Запускаем выполнение графа синхронно или асинхронно (.ainvoke)
        # Так как наши кастомные тулзы поиска, скорее всего, синхронные, используем invoke
        result = compiled_agent.invoke(input_data, config)
        
        # Извлекаем последнее сообщение из обновленного состояния графа (ответ модели)
        final_message = result["messages"][-1].content
        
        return ChatResponse(response=final_message)
        
    except Exception as e:
        # Логируем ошибку, если что-то пошло не так внутри графа или тулзов
        raise HTTPException(status_code=500, detail=f"Ошибка работы агента: {str(e)}")

if __name__ == "__main__":
    # Получаем порт из переменных окружения (Selectel), по умолчанию 8000
    port = int(os.getenv("PORT", 8000))
    # Запускаем uvicorn-сервер на хосте 0.0.0.0, чтобы он был доступен извне
    uvicorn.run("main import app", host="0.0.0.0", port=port, reload=True)
