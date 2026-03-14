import json
import os
from typing import List, Literal

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from openai import OpenAI
from pydantic import BaseModel

load_dotenv()

PORT = int(os.getenv("PORT", "8000"))
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")
NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "openai/gpt-oss-120b")

MODEL_OPTIONS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "moonshotai/kimi-k2-instruct"
]

if not NVIDIA_API_KEY:
    print("WARNING: NVIDIA_API_KEY is missing. Please create .env first.")

client = OpenAI(
    api_key=NVIDIA_API_KEY,
    base_url=NVIDIA_BASE_URL,
)

app = FastAPI(title="HW01 Your Own ChatGPT - NVIDIA NIM")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    model: str
    systemPrompt: str
    messages: List[Message]
    temperature: float = 0.7
    top_p: float = 1.0
    max_tokens: int = 512


def sanitize_model(model: str) -> str:
    if model in MODEL_OPTIONS:
        return model
    return DEFAULT_MODEL


def sanitize_messages(messages: List[Message]) -> List[dict]:
    safe_messages = []
    for m in messages:
        safe_messages.append({
            "role": m.role,
            "content": str(m.content)[:12000]
        })
    return safe_messages


def clamp(value, low, high, fallback):
    try:
        v = float(value)
        if v < low:
            return low
        if v > high:
            return high
        return v
    except Exception:
        return fallback


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "models": MODEL_OPTIONS,
            "default_model": DEFAULT_MODEL,
        },
    )


@app.get("/api/models")
async def get_models():
    return {"models": MODEL_OPTIONS, "default_model": DEFAULT_MODEL}


@app.get("/api/health")
async def health():
    return {"ok": True}


@app.post("/api/chat")
async def chat(req: ChatRequest):
    try:
        model = sanitize_model(req.model)
        safe_messages = sanitize_messages(req.messages)

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": req.systemPrompt[:4000]},
                *safe_messages,
            ],
            temperature=clamp(req.temperature, 0.0, 2.0, 0.7),
            top_p=clamp(req.top_p, 0.1, 1.0, 1.0),
            max_tokens=int(clamp(req.max_tokens, 64, 2048, 512)),
        )

        text = response.choices[0].message.content if response.choices else ""

        return {"output": text or ""}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Chat request failed: {str(e)}"}
        )


@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest):
    try:
        model = sanitize_model(req.model)
        safe_messages = sanitize_messages(req.messages)

        stream = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": req.systemPrompt[:4000]},
                *safe_messages,
            ],
            temperature=clamp(req.temperature, 0.0, 2.0, 0.7),
            top_p=clamp(req.top_p, 0.1, 1.0, 1.0),
            max_tokens=int(clamp(req.max_tokens, 64, 2048, 512)),
            stream=True,
        )

        def generate():
            try:
                for chunk in stream:
                    delta = None
                    if chunk.choices and chunk.choices[0].delta:
                        delta = chunk.choices[0].delta.content
                    if delta:
                        yield json.dumps({"type": "delta", "content": delta}, ensure_ascii=False) + "\n"

                yield json.dumps({"type": "done"}, ensure_ascii=False) + "\n"
            except Exception as e:
                yield json.dumps({"type": "error", "content": str(e)}, ensure_ascii=False) + "\n"

        return StreamingResponse(generate(), media_type="application/x-ndjson")
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Streaming request failed: {str(e)}"}
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=PORT, reload=True)