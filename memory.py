"""
memory.py — Long-term memory manager
Stores user preferences, episodic facts, and semantic knowledge in SQLite.
Uses keyword-overlap similarity for retrieval (no GPU / vector DB required).
"""

import json
import math
import re
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

DB_PATH = Path("chat_history.db")

# ── Similarity ────────────────────────────────────────────────────────────────

_STOP_WORDS = {
    # English
    "the","a","an","is","are","was","were","be","been","have","has","had",
    "do","does","did","will","would","could","should","may","might","can",
    "shall","to","of","in","on","at","for","with","about","by","from",
    "i","my","me","you","your","he","she","it","we","they","this","that",
    "and","or","but","not","so","if","then","as","what","who","when","where",
    # Chinese
    "我","你","他","她","是","的","了","在","和","有","不","也","就",
    "都","而","及","與","跟","但","所","如","被","從","由","對","到",
}

def _tokens(text: str) -> set:
    words = re.findall(r"[\u4e00-\u9fff]|[a-zA-Z0-9]+", text.lower())
    return {w for w in words if w not in _STOP_WORDS and len(w) > 1}

def _jaccard(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / math.sqrt(len(ta) * len(tb))

# ── DB init ───────────────────────────────────────────────────────────────────

def init_memory_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS long_term_memory (
            id           TEXT PRIMARY KEY,
            user_id      TEXT    NOT NULL DEFAULT 'default',
            memory_type  TEXT    NOT NULL,          -- profile | episodic | semantic
            content      TEXT    NOT NULL,
            summary      TEXT,
            keywords     TEXT,                       -- JSON array
            confidence   REAL    DEFAULT 1.0,
            ttl_days     INTEGER DEFAULT 90,
            source_session TEXT,
            created_at   TEXT,
            updated_at   TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS tool_logs (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id  TEXT,
            tool_name   TEXT,
            input_data  TEXT,
            output_data TEXT,
            status      TEXT,
            latency_ms  REAL,
            created_at  TEXT
        )
    """)
    conn.commit()
    conn.close()

# ── Keyword extraction ────────────────────────────────────────────────────────

def _extract_keywords(text: str) -> List[str]:
    return list(_tokens(text))[:25]

# ── Classifier ────────────────────────────────────────────────────────────────

_PROFILE_RE = re.compile(
    r"my name is|i am|i'm|i prefer|i like|i love|i hate|i always|i usually|"
    r"call me|i work|i study|i'm from|my job|我叫|我是|我喜歡|我不喜歡|我偏好|"
    r"我習慣|請叫我|我的名字|我來自|我在",
    re.IGNORECASE,
)
_EPISODIC_RE = re.compile(
    r"we discussed|i mentioned|last time|working on|project|deadline|task|"
    r"我們討論|我提到|上次|目前在做|任務|專案|這次",
    re.IGNORECASE,
)
_SEMANTIC_RE = re.compile(
    r"means|is defined as|refers to|fact that|definition of|"
    r"意思是|定義|代表|指的是|事實",
    re.IGNORECASE,
)

def classify_memory(text: str) -> Optional[str]:
    if _PROFILE_RE.search(text):
        return "profile"
    if _EPISODIC_RE.search(text):
        return "episodic"
    if _SEMANTIC_RE.search(text):
        return "semantic"
    return None

def should_store(user_msg: str) -> Dict:
    mtype = classify_memory(user_msg)
    if mtype:
        ttl = 365 if mtype == "profile" else (60 if mtype == "episodic" else 180)
        return {"store": True, "type": mtype, "ttl_days": ttl, "confidence": 0.85}
    return {"store": False}

# ── CRUD ──────────────────────────────────────────────────────────────────────

def upsert_memory(
    user_id: str,
    session_id: str,
    memory_type: str,
    content: str,
    confidence: float = 0.85,
    ttl_days: int = 90,
) -> str:
    keywords = json.dumps(_extract_keywords(content), ensure_ascii=False)
    now = datetime.utcnow().isoformat()

    conn = sqlite3.connect(DB_PATH)
    # Dedup: update if very similar memory exists
    rows = conn.execute(
        "SELECT id, content FROM long_term_memory WHERE user_id=? AND memory_type=?",
        (user_id, memory_type),
    ).fetchall()
    for rid, existing in rows:
        if _jaccard(content, existing) > 0.65:
            conn.execute(
                "UPDATE long_term_memory SET content=?, keywords=?, confidence=?, updated_at=? WHERE id=?",
                (content, keywords, confidence, now, rid),
            )
            conn.commit()
            conn.close()
            return rid

    mid = str(uuid.uuid4())
    conn.execute(
        """INSERT INTO long_term_memory
           (id,user_id,memory_type,content,summary,keywords,confidence,ttl_days,source_session,created_at,updated_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
        (mid, user_id, memory_type, content, content[:120], keywords,
         confidence, ttl_days, session_id, now, now),
    )
    conn.commit()
    conn.close()
    return mid


def search_memory(query: str, user_id: str = "default", top_k: int = 4) -> List[Dict]:
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        """SELECT id, memory_type, content, summary, confidence, created_at
           FROM long_term_memory WHERE user_id=? ORDER BY updated_at DESC""",
        (user_id,),
    ).fetchall()
    conn.close()
    if not rows:
        return []
    scored = []
    for rid, mtype, content, summary, conf, created_at in rows:
        score = _jaccard(query, content) * float(conf)
        if score > 0.04:
            scored.append({"id": rid, "type": mtype, "content": content,
                           "summary": summary, "score": round(score, 3),
                           "created_at": created_at})
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


def list_memories(user_id: str = "default") -> List[Dict]:
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        """SELECT id, memory_type, content, summary, confidence, created_at, updated_at
           FROM long_term_memory WHERE user_id=? ORDER BY updated_at DESC""",
        (user_id,),
    ).fetchall()
    conn.close()
    return [
        {"id": r[0], "type": r[1], "content": r[2], "summary": r[3],
         "confidence": r[4], "created_at": r[5], "updated_at": r[6]}
        for r in rows
    ]


def delete_memory(memory_id: str):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM long_term_memory WHERE id=?", (memory_id,))
    conn.commit()
    conn.close()


def format_memory_context(memories: List[Dict]) -> str:
    if not memories:
        return ""
    lines = ["\n\n[使用者長期記憶 — 關於這個使用者你已知道的資訊:]"]
    for m in memories:
        emoji = {"profile": "👤", "episodic": "📅", "semantic": "📚"}.get(m["type"], "🧠")
        lines.append(f"  {emoji} [{m['type']}] {m['content']}")
    lines.append("[以上記憶僅供參考，如有衝突以使用者最新說法為準]")
    return "\n".join(lines)


def log_tool_call(session_id: str, tool_name: str,
                  input_data: dict, output_data: dict,
                  status: str, latency_ms: float):
    now = datetime.utcnow().isoformat()
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """INSERT INTO tool_logs (session_id,tool_name,input_data,output_data,status,latency_ms,created_at)
           VALUES (?,?,?,?,?,?,?)""",
        (session_id, tool_name,
         json.dumps(input_data, ensure_ascii=False),
         json.dumps(output_data, ensure_ascii=False),
         status, latency_ms, now),
    )
    conn.commit()
    conn.close()
