import base64
import json
import os
import re
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Literal, Optional
from rag import rag_manager
from dotenv import load_dotenv
from fastapi import FastAPI, File, Request, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from openai import OpenAI
from pydantic import BaseModel
from tool_cache import tool_cache
import httpx

import memory as mem
import router as rt
from tools import registry as tool_registry
from mcp_server import router as mcp_router  # MCP integration

load_dotenv()

PORT = int(os.getenv("PORT", "8000"))
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")
NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_API_BASE = os.getenv("GEMINI_API_BASE", "https://generativelanguage.googleapis.com/v1beta")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "openai/gpt-oss-120b")
GEMINI_DEFAULT_MODEL = os.getenv("GEMINI_DEFAULT_MODEL", "gemini/gemini-2.5-flash")
GEMINI_PRO_MODEL = os.getenv("GEMINI_PRO_MODEL", "gemini/gemini-2.5-pro")

MODEL_OPTIONS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "moonshotai/kimi-k2-instruct",
    "gemini/gemini-2.5-flash",
    "gemini/gemini-2.5-pro",
    "meta/llama-3.2-11b-vision-instruct",
    "microsoft/phi-3.5-vision-instruct",
]

# Vision-capable / multimodal models (must stay in sync with frontend VISION_MODELS)
VISION_MODELS = {
    "gemini/gemini-2.5-flash",
    "gemini/gemini-2.5-pro",
    "meta/llama-3.2-11b-vision-instruct",
    "microsoft/phi-3.5-vision-instruct",
}

if not NVIDIA_API_KEY:
    print("WARNING: NVIDIA_API_KEY is missing. NVIDIA text models may not work until .env is configured.")
if not GEMINI_API_KEY:
    print("WARNING: GEMINI_API_KEY is missing. Gemini multimodal / image features may not work until .env is configured.")

client = OpenAI(
    api_key=NVIDIA_API_KEY,
    base_url=NVIDIA_BASE_URL,
)

app = FastAPI(title="HW02 Your Own ChatGPT - Multi Provider")
app.include_router(mcp_router)  # MCP JSON-RPC + SSE + REST
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# ── Unified error handler ─────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch-all handler — every unhandled exception returns the same JSON shape."""
    import traceback
    tb = traceback.format_exc()
    print(f"[ERROR] {request.method} {request.url.path}\n{tb}")
    return JSONResponse(
        status_code=500,
        content={
            "error": type(exc).__name__,
            "detail": str(exc),
            "path": str(request.url.path),
        },
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Normalise FastAPI HTTP errors into the same JSON envelope."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "HTTPException",
            "detail": exc.detail,
            "path": str(request.url.path),
        },
    )

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
    c.execute("""
        CREATE TABLE IF NOT EXISTS shared_sessions (
            token TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            created_at TEXT,
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        )
    """)
    conn.commit()
    conn.close()


init_db()
mem.init_memory_db()


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
    conn.execute("DELETE FROM shared_sessions WHERE session_id=?", (session_id,))
    conn.execute("DELETE FROM sessions WHERE id=?", (session_id,))
    conn.commit()
    conn.close()


def db_create_share_token(session_id: str) -> str:
    token = uuid.uuid4().hex[:24]
    now = datetime.utcnow().isoformat()
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute("SELECT id FROM sessions WHERE id=?", (session_id,)).fetchone()
    if not row:
        conn.close()
        raise ValueError("Session not found")
    conn.execute(
        "INSERT INTO shared_sessions (token, session_id, created_at) VALUES (?, ?, ?)",
        (token, session_id, now),
    )
    conn.commit()
    conn.close()
    return token


def db_get_shared_session(token: str) -> Optional[dict]:
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute(
        """SELECT s.id, s.title, sh.created_at
           FROM shared_sessions sh
           JOIN sessions s ON s.id = sh.session_id
           WHERE sh.token=?""",
        (token,),
    ).fetchone()
    if not row:
        conn.close()
        return None
    messages = conn.execute(
        "SELECT role, content, created_at FROM messages WHERE session_id=? ORDER BY id ASC",
        (row[0],),
    ).fetchall()
    conn.close()
    return {
        "session_id": row[0],
        "title": row[1],
        "shared_at": row[2],
        "messages": [
            {"role": r[0], "content": r[1], "created_at": r[2]}
            for r in messages
        ],
    }


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
    attachment_type: Optional[str] = None  # image / text / pdf / document

    # HW2 additions
    auto_route: bool = True           # automatically select the best model
    use_memory: bool = True           # retrieve/store long-term memory
    tools_enabled: bool = True        # allow model tool calls
    agent_mode: bool = False          # show Plan → Act → Answer prompt-chaining steps
    tool_max_iterations: int = 20      # max tool-calling loop iterations (1–10)
    user_id: str = "default"
    use_rag: bool = True


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


def is_gemini_model(model: str) -> bool:
    return str(model or "").startswith("gemini/")


def gemini_model_name(model: str) -> str:
    return str(model or GEMINI_DEFAULT_MODEL).split("/", 1)[1] if str(model or "").startswith("gemini/") else str(model)


def _parse_data_url(url: str) -> tuple[Optional[str], Optional[str]]:
    if not url or not url.startswith("data:") or ";base64," not in url:
        return None, None
    header, data = url.split(",", 1)
    mime = header[5:].split(";", 1)[0] or "application/octet-stream"
    return mime, data


def _gemini_part_from_content_part(part: dict) -> Optional[dict]:
    if not isinstance(part, dict):
        return None
    if part.get("type") == "text":
        return {"text": part.get("text", "")}
    if part.get("type") == "image_url":
        image_url = (part.get("image_url") or {}).get("url", "")
        mime, b64 = _parse_data_url(image_url)
        if mime and b64:
            return {"inlineData": {"mimeType": mime, "data": b64}}
    return None


def _messages_to_gemini_payload(api_messages: list, max_tokens: int, temperature: float, top_p: float) -> dict:
    system_text = ""
    contents = []
    for msg in api_messages:
        role = msg.get("role", "user")
        content = msg.get("content")
        if role == "system":
            if isinstance(content, str):
                system_text = content
            elif isinstance(content, list):
                texts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"]
                system_text = "\n".join(t for t in texts if t)
            continue

        parts = []
        if isinstance(content, str):
            parts.append({"text": content})
        elif isinstance(content, list):
            for part in content:
                gp = _gemini_part_from_content_part(part)
                if gp:
                    parts.append(gp)

        if not parts:
            continue

        gemini_role = "model" if role == "assistant" else "user"
        contents.append({"role": gemini_role, "parts": parts})

    payload = {
        "contents": contents,
        "generationConfig": {
            "temperature": temperature,
            "topP": top_p,
            "maxOutputTokens": max_tokens,
        },
    }
    if system_text:
        payload["system_instruction"] = {"parts": [{"text": system_text}]}
    return payload


def _extract_gemini_text(data: dict) -> str:
    texts = []
    for cand in data.get("candidates", []) or []:
        content = cand.get("content", {}) or {}
        for part in content.get("parts", []) or []:
            if isinstance(part, dict) and part.get("text"):
                texts.append(part["text"])
    return "\n".join(t.strip() for t in texts if t and t.strip()).strip()


def _gemini_usage_from_response(data: dict) -> dict:
    usage = data.get("usageMetadata", {}) or {}
    if usage:
        return {
            "prompt_tokens": usage.get("promptTokenCount", 0),
            "completion_tokens": usage.get("candidatesTokenCount", 0),
            "total_tokens": usage.get("totalTokenCount", 0),
        }
    return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "estimated": True}


def _gemini_generate_once(model: str, api_messages: list, req: ChatRequest) -> tuple[str, dict]:
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is missing")
    payload = _messages_to_gemini_payload(
        api_messages,
        max_tokens=int(clamp(req.max_tokens, 64, 4096, 512)),
        temperature=clamp(req.temperature, 0.0, 2.0, 0.7),
        top_p=clamp(req.top_p, 0.1, 1.0, 1.0),
    )
    model_name = gemini_model_name(model)
    url = f"{GEMINI_API_BASE}/models/{model_name}:generateContent?key={GEMINI_API_KEY}"
    resp = httpx.post(url, json=payload, timeout=120.0)
    data = resp.json()
    if resp.status_code >= 400:
        raise RuntimeError(data.get("error", {}).get("message", resp.text[:1000]))
    return _extract_gemini_text(data), _gemini_usage_from_response(data)


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


@app.post("/api/sessions/{session_id}/share")
async def share_session(session_id: str, request: Request):
    try:
        token = db_create_share_token(session_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Session not found")
    url = str(request.url_for("view_shared_session", token=token))
    return {"token": token, "url": url}


@app.get("/api/share/{token}")
async def get_shared_session_json(token: str):
    data = db_get_shared_session(token)
    if not data:
        raise HTTPException(status_code=404, detail="Share link not found")
    return data


@app.get("/share/{token}", response_class=HTMLResponse, name="view_shared_session")
async def view_shared_session(token: str):
    data = db_get_shared_session(token)
    if not data:
        return HTMLResponse("<h1>Share link not found</h1>", status_code=404)
    items = []
    for m in data["messages"]:
        role = "You" if m["role"] == "user" else "Assistant"
        content = (m["content"] or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        items.append(f'<div class="msg {m["role"]}"><b>{role}</b><pre>{content}</pre></div>')
    html = f"""
    <!doctype html><html lang="zh-Hant"><head><meta charset="utf-8">
    <meta name="viewport" content="width=device-width,initial-scale=1">
    <title>Shared Chat - {data['title']}</title>
    <style>
      body{{font-family:system-ui,-apple-system,'Segoe UI',sans-serif;background:#f7f7f8;margin:0;padding:32px;color:#222}}
      .wrap{{max-width:900px;margin:auto}}
      h1{{font-size:24px}}
      .msg{{background:white;border:1px solid #ddd;border-radius:14px;padding:16px;margin:14px 0;box-shadow:0 8px 20px rgba(0,0,0,.04)}}
      .msg.user{{background:#fff0f4}}
      pre{{white-space:pre-wrap;word-break:break-word;font-family:inherit;line-height:1.6}}
      .meta{{color:#777;font-size:13px}}
    </style></head><body><div class="wrap">
      <h1>Shared Chat: {data['title']}</h1>
      <p class="meta">唯讀分享連結 · token: {token}</p>
      {''.join(items)}
    </div></body></html>
    """
    return HTMLResponse(html)


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

    # PDF: extract text when possible. Most chat/vision APIs do not accept a raw PDF
    # in an image_url field, so sending extracted text is more reliable.
    if mime == "application/pdf":
        try:
            import io
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(data))
            pages = []
            for i, page in enumerate(reader.pages[:20], start=1):
                pages.append(f"[Page {i}]\n" + (page.extract_text() or ""))
            text = "\n\n".join(pages).strip()
            if text:
                return {"type": "text", "mime": mime, "text": text[:30000], "filename": file.filename,
                        "note": "PDF text extracted on backend."}
        except Exception:
            pass
        return {"type": "text", "mime": mime,
                "text": "[PDF text extraction failed. Please paste the text or upload images of the pages.]",
                "filename": file.filename}

    return JSONResponse(status_code=415, content={"error": f"Unsupported file type: {mime}"})


# ── Chat endpoints ────────────────────────────────────────────────────────────

def _build_messages(
    req: ChatRequest,
    user_text: str,
    model: Optional[str] = None,
    memory_context: str = "",
) -> list:
    """
    Build the messages list to send to the API.

    HW2 additions:
    - The chosen model may come from auto routing rather than the dropdown.
    - Long-term memory is appended to the system prompt.
    - The final user message can be multipart when an image is attached.
    """
    model = sanitize_model(model or req.model)
    rag_chunks = rag_manager.retrieve(user_text, req.session_id or "", top_k=4) if req.session_id and req.use_rag else []
    rag_context = rag_manager.format_context(rag_chunks) if rag_chunks else ""
    system_base = (rag_context + "\n\n" if rag_context else "") + req.systemPrompt[:4000]


    if memory_context:
        system_base += (
            "\n\n[CRITICAL USER MEMORY]\n"
            f"{memory_context}\n"
            "請務必遵守上述長期記憶，尤其是使用者偏好。"
        )
    system_text = (
        build_thinking_system(system_base, req.thinking_budget)
        if req.thinking
        else system_base
    )

    history = []
    all_msgs = sanitize_messages(req.messages)
    context_msgs = all_msgs[:-1] if all_msgs else []
    for m in context_msgs:
        history.append(m)

    if req.image_b64 and model in VISION_MODELS:
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

    return [
        {"role": "system", "content": system_text},
        *history,
        {"role": "user", "content": final_content},
    ]


def _json_line(obj: dict) -> str:
    """NDJSON line for the existing HW1 frontend streaming parser."""
    return json.dumps(obj, ensure_ascii=False) + "\n"


def _decide_route(req: ChatRequest, user_text: str):
    """Return (route decision, selected model)."""
    decision = rt.route_request(
        user_text,
        has_image=bool(req.image_b64),
        preferred_model=req.model,
        auto_route=req.auto_route,
        attachment_kind=req.attachment_type,
    )
    selected_model = sanitize_model(decision.model)

    # Safety fallback: image input must use a vision-capable model.
    if (req.image_b64 or req.attachment_type in {"text", "pdf", "document"}) and selected_model not in VISION_MODELS:
        selected_model = GEMINI_DEFAULT_MODEL

    return decision, selected_model


def _agent_steps_for(req: ChatRequest, route_decision, user_text: str, has_memory: bool) -> list:
    """Visible prompt-chaining / agent-mode plan for the UI.

    This is a lightweight agent workflow: the plan is generated deterministically
    from the routed task type, while actual model/tool execution still happens
    normally afterward.  The goal is to make the Plan → Tool → Verify → Answer
    process visible during demo without adding another planning API call.
    """
    task = getattr(route_decision, "task_type", "general") or "general"
    text_l = (user_text or "").lower()

    def _topic_from_text() -> str:
        text = (user_text or "").strip()
        if not text:
            return "使用者輸入內容"
        compact = " ".join(text.split())
        return compact[:36] + ("…" if len(compact) > 36 else "")

    def _analyze_detail() -> str:
        topic = _topic_from_text()
        if req.image_b64:
            return f"判斷使用者想要針對附件 / 圖片進行分析：{topic}"
        if task == "image_gen":
            return f"判斷使用者想要生成圖片：{topic}"
        if task == "weather":
            return "判斷使用者想要查詢目前天氣；需要確認城市或地區"
        if task == "math":
            return f"判斷使用者想要進行計算或數值推導：{topic}"
        if task == "search" or any(k in text_l for k in ["apple pencil", "相容", "支援機型", "官方", "最新", "目前"]):
            return f"判斷使用者想要取得較新的外部資訊或產品資料：{topic}"
        if task == "coding":
            return f"判斷使用者想要程式撰寫、除錯或整合：{topic}"
        return f"解析使用者目標與輸入格式：{topic}"

    def _decide_detail() -> str:
        if req.image_b64:
            return "需要切換到 Vision-capable model，並將附件作為 multimodal input 傳入"
        if task == "image_gen":
            return "需要呼叫圖片生成工具，而不是只回覆文字 prompt"
        if task == "weather":
            return "若輸入包含地點，呼叫天氣工具；若缺少地點，先詢問使用者補充"
        if task == "math":
            return "需要呼叫 calculator tool，避免只靠語言模型心算"
        if task == "search" or any(k in text_l for k in ["apple pencil", "相容", "支援機型", "官方", "最新", "目前"]):
            return "需要查詢目前資料，再整理成回答，避免使用過時知識"
        if task == "coding":
            return "需要使用程式能力較穩定的文字模型，並保留上下文"
        return f"選擇模型 / 任務類型：{task}"

    def _tool_detail() -> str:
        if req.image_b64:
            return "不一定需要外部工具；主要使用 Vision 模型讀取附件內容"
        if not req.tools_enabled:
            return "工具使用已關閉，改由模型直接回答"
        if task == "image_gen":
            return "呼叫 generate_image 產生圖片"
        if task == "weather":
            return "呼叫 get_weather 查詢即時天氣；若缺少地點則先回覆澄清問題"
        if task == "datetime":
            return "呼叫 get_datetime 取得系統時間（不需網路）"
        if task == "presentation":
            return "呼叫 create_presentation：先讓 LLM 產生投影片大綱 JSON，再傳入工具產生 .pptx 檔案"
        if task == "math":
            return "呼叫 calculator 執行精確計算"
        if task == "search" or "apple pencil" in text_l or "官方" in text_l or "相容" in text_l:
            return "呼叫 web_search 查詢官方或近期資料"
        if has_memory:
            return "檢索 long-term memory，將相關偏好與事實注入 context"
        return "本任務不需外部工具；直接進入回答生成"

    def _synthesize_detail() -> str:
        if task == "image_gen":
            return "整理工具回傳的圖片結果，嵌入回覆中展示"
        if req.image_b64:
            return "整合圖片 / 文件內容與使用者問題，產生可讀的分析"
        if task in {"search", "weather"} or any(k in text_l for k in ["apple pencil", "官方", "最新", "目前", "相容"]):
            return "整理查詢結果，萃取重點並用繁體中文表達"
        return "整合模型推理、短期記憶與長期記憶內容"

    def _verify_detail() -> str:
        if task in {"search", "weather"} or any(k in text_l for k in ["apple pencil", "官方", "最新", "目前", "相容"]):
            return "檢查資訊是否可能過時、是否遺漏型號 / 條件 / 限制"
        if task == "math":
            return "檢查計算式、單位與結果格式是否一致"
        if task == "coding":
            return "檢查程式碼是否可執行、變數是否一致、是否破壞原功能"
        return "檢查回答是否符合使用者需求，並避免無關內容"

    def _final_detail() -> str:
        if task == "image_gen":
            return "輸出圖片與簡短說明"
        if req.image_b64:
            return "產生針對附件的最終回答"
        return "產生最終回覆"

    return [
        {"title": "Analyze Task", "detail": _analyze_detail()},
        {"title": "Decide Actions", "detail": _decide_detail()},
        {"title": "Tool Call", "detail": _tool_detail()},
        {"title": "Synthesize", "detail": _synthesize_detail()},
        {"title": "Verify", "detail": _verify_detail()},
        {"title": "Final Answer", "detail": _final_detail()},
    ]


def _get_memory_context(req: ChatRequest, decision, user_text: str):
    if not (req.use_memory and getattr(decision, "use_memory", True)):
        return "", []

    user_id = req.user_id or "default"

    # 原本語意搜尋
    memories = mem.search_memory(user_text, user_id, top_k=4) or []

    # 固定抓 profile 類偏好
    all_memories = mem.list_memories(user_id) or []
    profile_memories = [
        m for m in all_memories
        if m.get("type") == "profile" or m.get("memory_type") == "profile"
    ]

    # 去重
    seen = set()
    merged = []
    for m in memories + profile_memories:
        mid = m.get("id") or m.get("content")
        if mid not in seen:
            seen.add(mid)
            merged.append(m)

    print("=== MEMORY DEBUG ===")
    print("USER ID:", user_id)
    print("MERGED MEMORIES:", merged)

    return mem.format_memory_context(merged), merged

def _maybe_store_memory(req: ChatRequest, user_text: str):
    if not req.use_memory or not user_text.strip():
        return None

    force_patterns = [
        "請記住", "記住", "幫我記", "請記得", "以後請","請記憶"
        "remember", "note that", "from now on"
    ]

    if any(p.lower() in user_text.lower() for p in force_patterns):
        return mem.upsert_memory(
            user_id=req.user_id,
            session_id=req.session_id or "no-session",
            memory_type="profile",
            content=user_text,
            confidence=1.0,
            ttl_days=365,
        )

    decision = mem.should_store(user_text)
    if not decision.get("store"):
        return None

    return mem.upsert_memory(
        user_id=req.user_id,
        session_id=req.session_id or "no-session",
        memory_type=decision["type"],
        content=user_text,
        confidence=decision.get("confidence", 0.85),
        ttl_days=decision.get("ttl_days", 90),
    )

def _display_route_model(route_decision, model: str) -> str:
    """Human-facing routed target shown in the UI."""
    task = getattr(route_decision, "task_type", "")
    if task == "image_gen":
        return "generate_image / Gemini Image (Nano Banana)"
    if task == "weather":
        return "get_weather / wttr.in"
    if task == "presentation":
        return "create_presentation / python-pptx"
    if task == "datetime":
        return "get_datetime / system clock"
    return model


def _run_datetime_tool(user_text: str, req: ChatRequest) -> tuple[str, list, dict]:
    """Deterministically call get_datetime and format a friendly reply."""
    # Best-effort timezone detection from user text
    tz = "Asia/Taipei"
    lower = (user_text or "").lower()
    tz_map = {
        "tokyo": "Asia/Tokyo", "japan": "Asia/Tokyo", "日本": "Asia/Tokyo",
        "new york": "America/New_York", "紐約": "America/New_York",
        "london": "Europe/London", "倫敦": "Europe/London",
        "paris": "Europe/Paris", "巴黎": "Europe/Paris",
        "beijing": "Asia/Shanghai", "北京": "Asia/Shanghai", "shanghai": "Asia/Shanghai",
        "seoul": "Asia/Seoul", "首爾": "Asia/Seoul",
        "los angeles": "America/Los_Angeles", "洛杉磯": "America/Los_Angeles",
    }
    for keyword, zone in tz_map.items():
        if keyword in lower:
            tz = zone
            break

    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "estimated": True}
    events = [{"type": "tool_start", "name": "get_datetime", "args": {"timezone": tz}}]
    result = tool_registry.execute("get_datetime", {"timezone": tz})
    latency = result.pop("_latency_ms", 0)
    events.append({"type": "tool_end", "name": "get_datetime", "result": result, "latency_ms": latency})

    if req.session_id:
        mem.log_tool_call(req.session_id, "get_datetime", {"timezone": tz}, result,
                          "ok" if "error" not in result else "error", latency)

    if "error" in result:
        text = f"時間查詢工具發生錯誤：{result.get('error')}"
    else:
        text = (
            f"已透過 `get_datetime` 工具取得即時時間。\n\n"
            f"| 項目 | 數值 |\n|---|---|\n"
            f"| 📅 日期 | {result['date']} |\n"
            f"| 🕐 時間 | {result['time']} |\n"
            f"| 📆 星期 | {result['weekday_zh']} ({result['weekday']}) |\n"
            f"| 🌍 時區 | {result['timezone']} |"
        )
    return text, events, usage


def _extract_weather_location(user_text: str) -> str:
    """Best-effort city/location extraction for demo-friendly weather routing."""
    text = (user_text or "").strip()
    if not text:
        return ""

    # Common Taiwan city aliases first.
    aliases = {
        "台北": "Taipei", "臺北": "Taipei", "新北": "New Taipei", "桃園": "Taoyuan",
        "台中": "Taichung", "臺中": "Taichung", "台南": "Tainan", "臺南": "Tainan",
        "高雄": "Kaohsiung", "基隆": "Keelung", "新竹": "Hsinchu", "苗栗": "Miaoli",
        "彰化": "Changhua", "南投": "Nantou", "雲林": "Yunlin", "嘉義": "Chiayi",
        "屏東": "Pingtung", "宜蘭": "Yilan", "花蓮": "Hualien", "台東": "Taitung",
        "臺東": "Taitung", "澎湖": "Penghu", "金門": "Kinmen", "馬祖": "Matsu",
        "Taipei": "Taipei", "Tokyo": "Tokyo", "Osaka": "Osaka", "Kyoto": "Kyoto",
        "Seoul": "Seoul", "New York": "New York", "London": "London",
    }
    lower = text.lower()
    for k, v in aliases.items():
        if k.lower() in lower:
            return v

    # English pattern: weather in/at <location>
    m = re.search(r"(?:weather|forecast|temperature)\s+(?:in|at|for)\s+([A-Za-z][A-Za-z .'-]{1,40})", text, re.I)
    if m:
        return m.group(1).strip()

    # Chinese pattern: <location> 的/今天/現在 天氣
    m = re.search(r"([\u4e00-\u9fffA-Za-z .'-]{2,20})(?:今天|今日|現在|明天|的)?(?:天氣|氣溫|溫度|降雨|預報)", text)
    if m:
        loc = re.sub(r"^(請問|幫我查|查一下|我要知道|想知道)", "", m.group(1)).strip()
        return loc
    return ""

def _format_weather_text(location: str, result: dict) -> str:
    """抽出來讓快取命中時也能重用相同格式。"""
    if "error" in result:
        return f"天氣工具查詢失敗：{result.get('error')}"
    loc   = result.get("location", location)
    desc  = result.get("description", "?")
    temp  = result.get("temperature_c", "?")
    feels = result.get("feels_like_c", "?")
    humid = result.get("humidity_pct", "?")
    wind  = result.get("wind_kmph", "?")
    cache_note = "⚡ 快取結果" if result.get("_cache_hit") else "即時查詢"
    return (
        f"已自動路由到天氣工具 `get_weather / wttr.in`（{cache_note}）。\n\n"
        f"**{loc} 目前天氣**\n\n"
        f"| 項目 | 數值 |\n|---|---|\n"
        f"| 天氣狀況 | {desc} |\n"
        f"| 溫度 | {temp} °C |\n"
        f"| 體感溫度 | {feels} °C |\n"
        f"| 濕度 | {humid}% |\n"
        f"| 風速 | {wind} km/h |\n\n"
        "資料來源：weather tool。"
    )
 
def _run_weather_tool(user_text: str, req: ChatRequest) -> tuple[str, list, dict]:
    location = _extract_weather_location(user_text)
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "estimated": True}
 
    if not location:
        return "請告訴我要查詢哪個城市或地區的天氣，例如：台北今天的天氣。", [], usage
 
    args = {"location": location}
 
    # ★ 快取命中
    cached = tool_cache.get("get_weather", args)
    if cached:
        events = [
            {"type": "tool_start", "name": "get_weather", "args": args, "cache_hit": True},
            {"type": "tool_end",   "name": "get_weather", "result": cached, "latency_ms": 0},
        ]
        return _format_weather_text(location, cached), events, usage
 
    events = [{"type": "tool_start", "name": "get_weather", "args": args}]
    result = tool_registry.execute("get_weather", args)
    latency = result.pop("_latency_ms", 0)
    events.append({"type": "tool_end", "name": "get_weather", "result": result, "latency_ms": latency})
 
    # ★ 寫入快取（TTL 10 分鐘）
    if "error" not in result:
        tool_cache.set("get_weather", args, result, ttl=600)
 
    if req.session_id:
        mem.log_tool_call(req.session_id, "get_weather", args, result,
                          "ok" if "error" not in result else "error", latency)
 
    return _format_weather_text(location, result), events, usage


def _run_image_generation_tool(user_text: str, req: ChatRequest) -> tuple[str, list, dict]:
    """Deterministically execute image generation for image_gen tasks.

    Tool-calling with tool_choice='auto' may decide to answer with text instead of
    actually calling the image tool. For HW2 auto-routing, image generation should
    route directly to the image tool so the behavior is visible and reliable.
    """
    prompt = (user_text or "").strip() or "Generate a high-quality image."
    events = [{"type": "tool_start", "name": "generate_image", "args": {"prompt": prompt}}]
    result = tool_registry.execute("generate_image", {"prompt": prompt})
    latency = result.pop("_latency_ms", 0)
    event_result = dict(result)
    if "image_b64" in event_result:
        event_result["image_b64"] = "[base64 image omitted]"
    events.append({
        "type": "tool_end",
        "name": "generate_image",
        "result": event_result,
        "latency_ms": latency,
    })

    if req.session_id:
        mem.log_tool_call(
            req.session_id,
            "generate_image",
            {"prompt": prompt},
            result,
            "ok" if "error" not in result else "error",
            latency,
        )

    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "estimated": True}
    if "error" in result:
        text = f"圖片生成工具發生錯誤：{result.get('error')}"
    else:
        mime = result.get("mime", "image/png")
        b64 = result.get("image_b64", "")
        if b64:
            text = (
                "已自動路由到圖片生成工具 `generate_image / Gemini Image (Nano Banana)`。\n\n"
                f"![Generated image](data:{mime};base64,{b64})"
            )
        else:
            text = f"圖片生成工具沒有回傳圖片：{result}"
    return text, events, usage


def _run_presentation_tool(user_text: str, req: ChatRequest) -> tuple[str, list, dict]:
    """Two-phase deterministic presentation generator.

    Phase 1 — Ask the LLM to output a structured JSON slide outline.
    Phase 2 — Pass the outline directly to create_presentation tool.

    This avoids relying on tool_choice='auto' where the model may decide
    to answer with text instead of actually calling the tool.
    """
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "estimated": True}
    events: list = []

    # ── Phase 1: generate slide outline ──────────────────────────────────────
    outline_prompt = (
        "你是一個簡報設計師，請根據使用者的需求產生一份簡報大綱。\n"
        "請只回傳 JSON，不要有任何說明文字或 markdown 程式碼區塊。\n"
        "JSON 格式必須完全符合以下結構：\n"
        "{\n"
        '  "title": "簡報標題",\n'
        '  "subtitle": "副標題（可省略）",\n'
        '  "filename": "英文檔名不含副檔名",\n'
        '  "slides": [\n'
        '    {"title": "投影片標題", "bullets": ["重點1", "重點2", "重點3"], "notes": "講者備註（可省略）"},\n'
        "    ...\n"
        "  ]\n"
        "}\n\n"
        f"使用者需求：{user_text}"
    )

    try:
        response = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[{"role": "user", "content": outline_prompt}],
            temperature=0.7,
            max_tokens=2048,
        )
        raw = response.choices[0].message.content or ""
        if getattr(response, "usage", None):
            usage = {
                "prompt_tokens": getattr(response.usage, "prompt_tokens", 0),
                "completion_tokens": getattr(response.usage, "completion_tokens", 0),
                "total_tokens": getattr(response.usage, "total_tokens", 0),
            }
    except Exception as e:
        return f"簡報大綱生成失敗：{e}", events, usage

    # Strip possible markdown fences the model may add despite instructions
    import re as _re
    clean = _re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=_re.I)
    clean = _re.sub(r"\s*```$", "", clean.strip())

    try:
        outline = json.loads(clean)
    except Exception:
        return (
            f"模型沒有回傳合法 JSON，無法建立簡報。\n\n模型原始回覆：\n{raw[:800]}",
            events,
            usage,
        )

    # ── Phase 2: call create_presentation tool ────────────────────────────────
    args = {
        "title":    outline.get("title", "Presentation"),
        "subtitle": outline.get("subtitle", ""),
        "filename": outline.get("filename", "presentation"),
        "slides":   outline.get("slides", []),
    }

    events.append({"type": "tool_start", "name": "create_presentation", "args": {
        "title": args["title"], "slide_count": len(args["slides"])
    }})
    result = tool_registry.execute("create_presentation", args)
    latency = result.pop("_latency_ms", 0)
    events.append({"type": "tool_end", "name": "create_presentation",
                   "result": result, "latency_ms": latency})

    if req.session_id:
        mem.log_tool_call(req.session_id, "create_presentation", args, result,
                          "ok" if "error" not in result else "error", latency)

    if "error" in result:
        text = f"簡報生成工具發生錯誤：{result.get('error')}"
    else:
        dl_url  = result.get("download_url", "")
        n_slides = result.get("slide_count", len(args["slides"]))
        title    = result.get("title", args["title"])
        text = (
            f"已自動路由到簡報生成工具 `create_presentation`。\n\n"
            f"**{title}** — 共 {n_slides} 張投影片\n\n"
            f"📥 [點擊下載 .pptx 檔案]({dl_url})\n\n"
            "---\n**投影片大綱：**\n"
        )
        for i, s in enumerate(args["slides"], start=1):
            bullets = s.get("bullets") or []
            text += f"\n**Slide {i}｜{s.get('title', '')}**\n"
            for b in bullets:
                text += f"- {b}\n"

    return text, events, usage


def _run_tools(api_messages: list, model: str, req: ChatRequest) -> tuple[str, list, dict]:
    """
    MCP-style tool loop using tools.py ToolRegistry.
    Returns (final_text, frontend_events, usage_dict).
    """
    # Tool-calling loop is implemented via OpenAI-compatible chat completions.
    # If the routed / selected model is Gemini, fall back to the default NVIDIA text model for tool orchestration.
    if is_gemini_model(model):
        model = DEFAULT_MODEL
    tool_defs = tool_registry.openai_tool_defs()
    events = []
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "estimated": True}
    max_iter = int(clamp(req.tool_max_iterations, 1, 20, 20))

    for iteration in range(max_iter):
        response = client.chat.completions.create(
            model=model,
            messages=api_messages,
            temperature=clamp(req.temperature, 0.0, 2.0, 0.7),
            top_p=clamp(req.top_p, 0.1, 1.0, 1.0),
            max_tokens=int(clamp(req.max_tokens, 64, 4096, 512)),
            tools=tool_defs,
            tool_choice="auto",
        )
        if getattr(response, "usage", None):
            usage = {
                "prompt_tokens": getattr(response.usage, "prompt_tokens", 0),
                "completion_tokens": getattr(response.usage, "completion_tokens", 0),
                "total_tokens": getattr(response.usage, "total_tokens", 0),
            }

        msg = response.choices[0].message
        tool_calls = getattr(msg, "tool_calls", None)

        if not tool_calls:
            final_text = msg.content or ""
            api_messages.append({"role": "assistant", "content": final_text})
            return final_text, events, usage

        # Emit progress so the frontend can show "第 N 次工具呼叫"
        events.append({"type": "tool_iteration", "iteration": iteration + 1, "max": max_iter})

        api_messages.append(msg.model_dump(exclude_none=True))

        for tc in tool_calls:
            fn_name = tc.function.name
            try:
                fn_args = json.loads(tc.function.arguments or "{}")
            except Exception:
                fn_args = {}

            events.append({"type": "tool_start", "name": fn_name, "args": fn_args})
 
            # ★ 快取檢查
            result = tool_cache.get(fn_name, fn_args)
            latency = 0
            if result is None:
                result = tool_registry.execute(fn_name, fn_args)
                latency = result.pop("_latency_ms", 0)
                # ★ 寫入快取（只快取成功結果）
                if "error" not in result:
                    tool_cache.set(fn_name, fn_args, result)
 
            events.append({
                "type": "tool_end",
                "name": fn_name,
                "result": result,
                "latency_ms": latency,
            })

            if req.session_id:
                mem.log_tool_call(
                    req.session_id,
                    fn_name,
                    fn_args,
                    result,
                    "ok" if "error" not in result else "error",
                    latency,
                )

            api_messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result, ensure_ascii=False),
            })

    return f"工具呼叫達到上限（{max_iter} 次），請重新嘗試或增加 tool_max_iterations。", events, usage


@app.post("/api/chat")
async def chat(req: ChatRequest):
    try:
        user_text = req.messages[-1].content if req.messages else ""
        route_decision, model = _decide_route(req, user_text)
        memory_context, memories = _get_memory_context(req, route_decision, user_text)
        agent_steps = _agent_steps_for(req, route_decision, user_text, bool(memory_context)) if req.agent_mode else []
        api_messages = _build_messages(req, user_text, model=model, memory_context=memory_context)

        if req.tools_enabled and route_decision.task_type == "image_gen":
            text, tool_events, usage = _run_image_generation_tool(user_text, req)
        elif req.tools_enabled and route_decision.task_type == "weather":
            text, tool_events, usage = _run_weather_tool(user_text, req)
        elif req.tools_enabled and route_decision.task_type == "presentation":
            text, tool_events, usage = _run_presentation_tool(user_text, req)
        elif req.tools_enabled and route_decision.task_type == "datetime":
            text, tool_events, usage = _run_datetime_tool(user_text, req)
        elif req.tools_enabled and route_decision.use_tools:
            text, tool_events, usage = _run_tools(api_messages, model, req)
        else:
            if is_gemini_model(model):
                text, usage = _gemini_generate_once(model, api_messages, req)
            else:
                response = client.chat.completions.create(
                    model=model,
                    messages=api_messages,
                    temperature=clamp(req.temperature, 0.0, 2.0, 0.7),
                    top_p=clamp(req.top_p, 0.1, 1.0, 1.0),
                    max_tokens=int(clamp(req.max_tokens, 64, 4096, 512)),
                )
                text = response.choices[0].message.content if response.choices else ""
                usage = {
                    "prompt_tokens":     getattr(response.usage, "prompt_tokens", 0),
                    "completion_tokens": getattr(response.usage, "completion_tokens", 0),
                    "total_tokens":      getattr(response.usage, "total_tokens", 0),
                }
            tool_events = []

        if req.session_id:
            db_add_message(req.session_id, "user", user_text)
            db_add_message(req.session_id, "assistant", text or "")
        stored_memory_id = _maybe_store_memory(req, user_text)

        return {
            "output": text or "",
            "usage": usage,
            "routing": {
                "model": _display_route_model(route_decision, model),
                "reason": route_decision.reason,
                "task_type": route_decision.task_type,
                "use_tools": bool(route_decision.use_tools and req.tools_enabled),
            },
            "memories_used": memories,
            "stored_memory_id": stored_memory_id,
            "tool_events": tool_events,
            "agent_steps": agent_steps,
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Chat request failed: {str(e)}"})


@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest):
    try:
        user_text = req.messages[-1].content if req.messages else ""
        route_decision, model = _decide_route(req, user_text)
        memory_context, memories = _get_memory_context(req, route_decision, user_text)
        agent_steps = _agent_steps_for(req, route_decision, user_text, bool(memory_context)) if req.agent_mode else []
        api_messages = _build_messages(req, user_text, model=model, memory_context=memory_context)

        def generate_tool_response():
            try:
                yield _json_line({"type": "routing", "model": _display_route_model(route_decision, model),
                                  "reason": route_decision.reason,
                                  "task_type": route_decision.task_type,
                                  "use_tools": bool(route_decision.use_tools and req.tools_enabled)})
                if memories:
                    yield _json_line({"type": "memory", "memories": memories})
                for i, step in enumerate(agent_steps):
                    yield _json_line({"type": "agent_step", "index": i, "total": len(agent_steps), **step})

                if route_decision.task_type == "image_gen":
                    text, tool_events, usage = _run_image_generation_tool(user_text, req)
                elif route_decision.task_type == "weather":
                    text, tool_events, usage = _run_weather_tool(user_text, req)
                elif route_decision.task_type == "presentation":
                    text, tool_events, usage = _run_presentation_tool(user_text, req)
                elif route_decision.task_type == "datetime":
                    text, tool_events, usage = _run_datetime_tool(user_text, req)
                else:
                    text, tool_events, usage = _run_tools(api_messages, model, req)
                for ev in tool_events:
                    yield _json_line(ev)
                if text:
                    yield _json_line({"type": "delta", "content": text})

                if req.session_id:
                    db_add_message(req.session_id, "user", user_text)
                    db_add_message(req.session_id, "assistant", text or "")
                stored_memory_id = _maybe_store_memory(req, user_text)
                yield _json_line({"type": "done", "usage": usage,
                                  "routing": {"model": _display_route_model(route_decision, model), "reason": route_decision.reason},
                                  "agent_steps": agent_steps,
                                  "stored_memory_id": stored_memory_id})
            except Exception as e:
                yield _json_line({"type": "error", "content": str(e)})

        # Tool-calling APIs usually return after tool calls rather than token-streaming.
        # Keep the frontend protocol the same by sending tool events + one final delta.
        if req.tools_enabled and route_decision.use_tools:
            return StreamingResponse(generate_tool_response(), media_type="application/x-ndjson")

        if is_gemini_model(model):
            text, usage = _gemini_generate_once(model, api_messages, req)

            def generate():
                try:
                    yield _json_line({"type": "routing", "model": _display_route_model(route_decision, model),
                                      "reason": route_decision.reason,
                                      "task_type": route_decision.task_type,
                                      "use_tools": False})
                    if memories:
                        yield _json_line({"type": "memory", "memories": memories})
                    for i, step in enumerate(agent_steps):
                        yield _json_line({"type": "agent_step", "index": i, "total": len(agent_steps), **step})

                    chunk_size = 80
                    for i in range(0, len(text), chunk_size):
                        yield _json_line({"type": "delta", "content": text[i:i+chunk_size]})

                    if req.session_id:
                        db_add_message(req.session_id, "user", user_text)
                        db_add_message(req.session_id, "assistant", text)
                    stored_memory_id = _maybe_store_memory(req, user_text)
                    yield _json_line({"type": "done", "usage": usage,
                                      "routing": {"model": _display_route_model(route_decision, model), "reason": route_decision.reason},
                                      "agent_steps": agent_steps,
                                      "stored_memory_id": stored_memory_id})
                except Exception as e:
                    yield _json_line({"type": "error", "content": str(e)})

            return StreamingResponse(generate(), media_type="application/x-ndjson")

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
                yield _json_line({"type": "routing", "model": _display_route_model(route_decision, model),
                                  "reason": route_decision.reason,
                                  "task_type": route_decision.task_type,
                                  "use_tools": False})
                if memories:
                    yield _json_line({"type": "memory", "memories": memories})
                for i, step in enumerate(agent_steps):
                    yield _json_line({"type": "agent_step", "index": i, "total": len(agent_steps), **step})

                for chunk in stream:
                    delta = None
                    if chunk.choices and chunk.choices[0].delta:
                        delta = chunk.choices[0].delta.content
                    if delta:
                        full_text.append(delta)
                        yield _json_line({"type": "delta", "content": delta})

                if req.session_id:
                    db_add_message(req.session_id, "user", user_text)
                    db_add_message(req.session_id, "assistant", "".join(full_text))
                stored_memory_id = _maybe_store_memory(req, user_text)

                total_chars = sum(len(m.content) for m in req.messages)
                est_prompt  = total_chars // 4
                est_comp    = len("".join(full_text)) // 4
                yield _json_line({"type": "done", "usage": {
                    "prompt_tokens":     est_prompt,
                    "completion_tokens": est_comp,
                    "total_tokens":      est_prompt + est_comp,
                    "estimated":         True,
                }, "routing": {"model": _display_route_model(route_decision, model), "reason": route_decision.reason},
                   "agent_steps": agent_steps,
                   "stored_memory_id": stored_memory_id})
            except Exception as e:
                yield _json_line({"type": "error", "content": str(e)})

        return StreamingResponse(generate(), media_type="application/x-ndjson")
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Streaming request failed: {str(e)}"})


# ── HW2 Long-term memory / tools endpoints ───────────────────────────────────

@app.get("/api/memories")
async def list_memories(user_id: str = "default"):
    return {"memories": mem.list_memories(user_id)}


class MemoryCreateRequest(BaseModel):
    content: str
    memory_type: str = "profile"   # profile | episodic | semantic
    user_id: str = "default"


@app.post("/api/memories")
async def create_memory(body: MemoryCreateRequest):
    """Manually add a long-term memory entry."""
    content = (body.content or "").strip()
    if not content:
        raise HTTPException(status_code=400, detail="content must not be empty")
    valid_types = {"profile", "episodic", "semantic"}
    mtype = body.memory_type if body.memory_type in valid_types else "profile"
    ttl = 365 if mtype == "profile" else (60 if mtype == "episodic" else 180)
    mid = mem.upsert_memory(
        user_id=body.user_id,
        session_id="manual",
        memory_type=mtype,
        content=content,
        confidence=1.0,
        ttl_days=ttl,
    )
    return {"ok": True, "id": mid, "type": mtype, "content": content}


@app.get("/api/memories/search")
async def search_memories(q: str, user_id: str = "default"):
    return {"results": mem.search_memory(q, user_id)}

from fastapi.responses import FileResponse

@app.get("/api/download/{filename}")
async def download_file(filename: str):
    file_path = Path("static/generated") / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(
        path=str(file_path),
        filename=filename,
        media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation"
    )

@app.delete("/api/memories/{memory_id}")
async def delete_memory(memory_id: str):
    mem.delete_memory(memory_id)
    return {"ok": True}


@app.get("/api/tools")
async def list_tools():
    return {"tools": tool_registry.list_tools()}

# ── RAG endpoints ─────────────────────────────────────────────────────────────

@app.get("/api/rag/docs")
async def list_rag_docs(session_id: str):
    return {"docs": rag_manager.list_docs(session_id)}


@app.post("/api/rag/ingest")
async def ingest_rag_doc(
    session_id: str,
    url: Optional[str] = None,           # ★ 新增：URL query param
    file: Optional[UploadFile] = File(None),
):
    """Ingest a file upload OR a URL into the RAG index."""
    if url and url.startswith("http"):
        # URL 模式：data 為空，filename 傳 URL
        result = rag_manager.ingest(b"", url, session_id, "text/html")
    elif file:
        data = await file.read()
        mime = file.content_type or "application/octet-stream"
        result = rag_manager.ingest(data, file.filename, session_id, mime)
    else:
        return JSONResponse(status_code=400, content={"error": "需要提供 file 或 url"})
 
    if "error" in result:
        return JSONResponse(status_code=400, content=result)
    return result


@app.delete("/api/rag/docs/{doc_id}")
async def delete_rag_doc(doc_id: str):
    rag_manager.delete_doc(doc_id)
    return {"ok": True}

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


# ── MCP Management UI ─────────────────────────────────────────────────────────
from pathlib import Path as _MCPPath

@app.get("/mcp-ui", response_class=HTMLResponse)
async def mcp_ui_page(request: Request):
    """Serve the standalone MCP Tools Manager UI."""
    ui_path = _MCPPath(__file__).parent / "mcp_ui.html"
    if not ui_path.exists():
        return HTMLResponse("<h1>mcp_ui.html not found</h1>", status_code=404)
    return HTMLResponse(ui_path.read_text(encoding="utf-8"))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=PORT, reload=True)