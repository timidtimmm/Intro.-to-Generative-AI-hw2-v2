import base64
import json
import os
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Literal, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, Request, UploadFile
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
    "moonshotai/kimi-k2-instruct",
    "meta/llama-3.2-11b-vision-instruct",   # ← vision model
    "microsoft/phi-3.5-vision-instruct",    # ← vision model
]

# Vision-capable models (must stay in sync with frontend VISION_MODELS)
VISION_MODELS = {
    "meta/llama-3.2-11b-vision-instruct",
    "microsoft/phi-3.5-vision-instruct",
}

if not NVIDIA_API_KEY:
    print("WARNING: NVIDIA_API_KEY is missing. Please create .env first.")

client = OpenAI(
    api_key=NVIDIA_API_KEY,
    base_url=NVIDIA_BASE_URL,
)

app = FastAPI(title="HW01 Your Own ChatGPT - NVIDIA NIM")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ── SQLite persistent memory ─────────────────────────────────────────────────

DB_PATH = Path("chat_history.db")


def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            title TEXT,
            created_at TEXT,
            updated_at TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            role TEXT,
            content TEXT,
            created_at TEXT,
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        )
    """)
    conn.commit()
    conn.close()


init_db()


def db_create_session(title: str = "New Chat") -> str:
    sid = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO sessions (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
        (sid, title, now, now),
    )
    conn.commit()
    conn.close()
    return sid


def db_list_sessions() -> list:
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT id, title, created_at, updated_at FROM sessions ORDER BY updated_at DESC"
    ).fetchall()
    conn.close()
    return [{"id": r[0], "title": r[1], "created_at": r[2], "updated_at": r[3]} for r in rows]


def db_get_messages(session_id: str) -> list:
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT role, content FROM messages WHERE session_id=? ORDER BY id ASC",
        (session_id,),
    ).fetchall()
    conn.close()
    return [{"role": r[0], "content": r[1]} for r in rows]


def db_add_message(session_id: str, role: str, content: str):
    now = datetime.utcnow().isoformat()
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
        (session_id, role, content, now),
    )
    conn.execute(
        "UPDATE sessions SET updated_at=? WHERE id=?",
        (now, session_id),
    )
    conn.commit()
    conn.close()


def db_rename_session(session_id: str, title: str):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("UPDATE sessions SET title=? WHERE id=?", (title, session_id))
    conn.commit()
    conn.close()


def db_delete_session(session_id: str):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM messages WHERE session_id=?", (session_id,))
    conn.execute("DELETE FROM sessions WHERE id=?", (session_id,))
    conn.commit()
    conn.close()


# ── Pydantic models ───────────────────────────────────────────────────────────

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
    thinking: bool = False
    thinking_budget: int = 5000
    session_id: Optional[str] = None
    image_b64: Optional[str] = None   # base64-encoded image
    image_mime: Optional[str] = None  # e.g. "image/png"


class RenameRequest(BaseModel):
    title: str


# ── Helpers ───────────────────────────────────────────────────────────────────

def sanitize_model(model: str) -> str:
    return model if model in MODEL_OPTIONS else DEFAULT_MODEL


def sanitize_messages(messages: List[Message]) -> List[dict]:
    return [{"role": m.role, "content": str(m.content)[:12000]} for m in messages]


def clamp(value, low, high, fallback):
    try:
        v = float(value)
        return max(low, min(high, v))
    except Exception:
        return fallback


def build_user_content(text: str, image_b64: Optional[str], image_mime: Optional[str], model: str):
    """Build the content field for the last user message, optionally with image."""
    if image_b64 and model in VISION_MODELS:
        return [
            {
                "type": "image_url",
                "image_url": {"url": f"data:{image_mime};base64,{image_b64}"},
            },
            {"type": "text", "text": text},
        ]
    return text


def build_thinking_system(original: str, budget: int) -> str:
    return (
        f"{original}\n\n"
        f"[THINKING MODE] Before answering, reason step-by-step inside <think>...</think> tags. "
        f"Use up to {budget} tokens to think carefully. "
        f"Then provide your final answer after the closing </think> tag."
    )


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "models": MODEL_OPTIONS, "default_model": DEFAULT_MODEL},
    )


@app.get("/api/models")
async def get_models():
    return {"models": MODEL_OPTIONS, "default_model": DEFAULT_MODEL}


@app.get("/api/health")
async def health():
    return {"ok": True}


# ── Session endpoints ─────────────────────────────────────────────────────────

@app.get("/api/sessions")
async def list_sessions():
    return {"sessions": db_list_sessions()}


@app.post("/api/sessions")
async def create_session():
    sid = db_create_session()
    return {"session_id": sid}


@app.get("/api/sessions/{session_id}/messages")
async def get_session_messages(session_id: str):
    return {"messages": db_get_messages(session_id)}


@app.patch("/api/sessions/{session_id}")
async def rename_session(session_id: str, body: RenameRequest):
    db_rename_session(session_id, body.title)
    return {"ok": True}


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    db_delete_session(session_id)
    return {"ok": True}


# ── File upload ───────────────────────────────────────────────────────────────

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """Accept image or text file, return base64 + mime for vision or text extraction."""
    MAX_SIZE = 10 * 1024 * 1024  # 10 MB
    data = await file.read()
    if len(data) > MAX_SIZE:
        return JSONResponse(status_code=400, content={"error": "File too large (max 10 MB)"})

    mime = file.content_type or "application/octet-stream"

    if mime.startswith("image/"):
        b64 = base64.b64encode(data).decode()
        return {"type": "image", "mime": mime, "b64": b64, "filename": file.filename}

    # For text / code / pdf – return decoded text (best-effort)
    if mime.startswith("text/") or mime in ("application/json",):
        try:
            text = data.decode("utf-8", errors="replace")
            return {"type": "text", "mime": mime, "text": text[:20000], "filename": file.filename}
        except Exception:
            pass

    # PDF: extract text via basic approach (just send raw bytes decoded)
    if mime == "application/pdf":
        b64 = base64.b64encode(data).decode()
        return {"type": "pdf", "mime": mime, "b64": b64, "filename": file.filename,
                "note": "PDF uploaded. Vision model will process it."}

    return JSONResponse(status_code=415, content={"error": f"Unsupported file type: {mime}"})


# ── Chat endpoints ────────────────────────────────────────────────────────────

def _build_messages(req: ChatRequest, user_text: str) -> list:
    """
    Build the messages list to send to the API.

    Strategy:
    - History messages (all but the last user turn) are included as-is for context.
    - The final user turn is always rebuilt fresh so we can inject the image correctly.
    - If an image is attached and the model supports vision, the final user message
      becomes a multipart content list: [image_url, text].
    - Otherwise it's plain text.
    """
    model = sanitize_model(req.model)
    system_text = (
        build_thinking_system(req.systemPrompt[:4000], req.thinking_budget)
        if req.thinking
        else req.systemPrompt[:4000]
    )

    # Build history: everything except the last user message
    history = []
    all_msgs = sanitize_messages(req.messages)
    context_msgs = all_msgs[:-1] if all_msgs else []
    for m in context_msgs:
        history.append(m)

    # Build the final user message
    if req.image_b64 and model in VISION_MODELS:
        # Multipart: image first, then text (NVIDIA NIM spec)
        final_content = [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{req.image_mime};base64,{req.image_b64}"
                },
            },
            {
                "type": "text",
                "text": user_text or "請描述這張圖片。",
            },
        ]
    else:
        final_content = user_text

    final_user_msg = {"role": "user", "content": final_content}

    return [
        {"role": "system", "content": system_text},
        *history,
        final_user_msg,
    ]


@app.post("/api/chat")
async def chat(req: ChatRequest):
    try:
        model = sanitize_model(req.model)
        user_text = req.messages[-1].content if req.messages else ""
        api_messages = _build_messages(req, user_text)

        response = client.chat.completions.create(
            model=model,
            messages=api_messages,
            temperature=clamp(req.temperature, 0.0, 2.0, 0.7),
            top_p=clamp(req.top_p, 0.1, 1.0, 1.0),
            max_tokens=int(clamp(req.max_tokens, 64, 4096, 512)),
        )
        text = response.choices[0].message.content if response.choices else ""

        # Persist to DB
        if req.session_id:
            db_add_message(req.session_id, "user", user_text)
            db_add_message(req.session_id, "assistant", text or "")

        return {"output": text or "", "usage": {
            "prompt_tokens":     getattr(response.usage, "prompt_tokens", 0),
            "completion_tokens": getattr(response.usage, "completion_tokens", 0),
            "total_tokens":      getattr(response.usage, "total_tokens", 0),
        }}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Chat request failed: {str(e)}"})


@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest):
    try:
        model = sanitize_model(req.model)
        user_text = req.messages[-1].content if req.messages else ""
        api_messages = _build_messages(req, user_text)

        stream = client.chat.completions.create(
            model=model,
            messages=api_messages,
            temperature=clamp(req.temperature, 0.0, 2.0, 0.7),
            top_p=clamp(req.top_p, 0.1, 1.0, 1.0),
            max_tokens=int(clamp(req.max_tokens, 64, 4096, 512)),
            stream=True,
        )

        def generate():
            full_text = []
            try:
                for chunk in stream:
                    delta = None
                    if chunk.choices and chunk.choices[0].delta:
                        delta = chunk.choices[0].delta.content
                    if delta:
                        full_text.append(delta)
                        yield json.dumps({"type": "delta", "content": delta}, ensure_ascii=False) + "\n"

                # Persist after stream completes
                if req.session_id:
                    db_add_message(req.session_id, "user", user_text)
                    db_add_message(req.session_id, "assistant", "".join(full_text))

                # Estimate token usage (streaming doesn't return usage directly)
                total_chars = sum(len(m.content) for m in req.messages)
                est_prompt  = total_chars // 4
                est_comp    = len("".join(full_text)) // 4
                yield json.dumps({"type": "done", "usage": {
                    "prompt_tokens":     est_prompt,
                    "completion_tokens": est_comp,
                    "total_tokens":      est_prompt + est_comp,
                    "estimated":         True,
                }}, ensure_ascii=False) + "\n"
            except Exception as e:
                yield json.dumps({"type": "error", "content": str(e)}, ensure_ascii=False) + "\n"

        return StreamingResponse(generate(), media_type="application/x-ndjson")
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Streaming request failed: {str(e)}"})


# ── Prompt template endpoints ─────────────────────────────────────────────────

def init_prompt_templates_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS prompt_templates (
            id   TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT
        )
    """)
    # Seed built-in presets if table is empty
    count = conn.execute("SELECT COUNT(*) FROM prompt_templates").fetchone()[0]
    if count == 0:
        now = datetime.utcnow().isoformat()
        presets = [
            ("Default",     "You are a helpful assistant. Answer clearly and accurately."),
            ("Teacher",     "You are a patient teacher. Explain step by step with simple examples."),
            ("Coder",       "You are a senior software engineer. Write robust and maintainable code."),
            ("Translator",  "You are a professional translator. Translate accurately and preserve tone."),
        ]
        for name, content in presets:
            conn.execute(
                "INSERT INTO prompt_templates (id, name, content, created_at) VALUES (?,?,?,?)",
                (str(uuid.uuid4()), name, content, now)
            )
    conn.commit()
    conn.close()

init_prompt_templates_db()


class PromptTemplateCreate(BaseModel):
    name: str
    content: str


@app.get("/api/prompt-templates")
async def list_prompt_templates():
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT id, name, content FROM prompt_templates ORDER BY created_at ASC"
    ).fetchall()
    conn.close()
    return {"templates": [{"id": r[0], "name": r[1], "content": r[2]} for r in rows]}


@app.post("/api/prompt-templates")
async def create_prompt_template(body: PromptTemplateCreate):
    tid = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO prompt_templates (id, name, content, created_at) VALUES (?,?,?,?)",
        (tid, body.name[:80], body.content[:4000], now)
    )
    conn.commit()
    conn.close()
    return {"id": tid, "name": body.name, "content": body.content}


@app.delete("/api/prompt-templates/{tid}")
async def delete_prompt_template(tid: str):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM prompt_templates WHERE id=?", (tid,))
    conn.commit()
    conn.close()
    return {"ok": True}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=PORT, reload=True)
