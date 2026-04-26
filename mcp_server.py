"""
mcp_server.py — Standard MCP (Model Context Protocol) Server
Implements JSON-RPC 2.0 over HTTP with SSE streaming support.
Endpoints:
  POST /mcp          — JSON-RPC 2.0 request/response
  GET  /mcp/sse      — SSE stream for server-push events
  GET  /mcp/tools    — list all registered tools
  POST /mcp/tools/{name}/invoke  — direct tool invocation
  POST /mcp/tools/{name}/toggle  — enable / disable a tool
  GET  /mcp/health   — server health & stats
"""

import asyncio
import json
import time
import uuid
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from tools import registry as tool_registry

router = APIRouter(prefix="/mcp", tags=["MCP"])

# ── SSE subscriber store ──────────────────────────────────────────────────────
_sse_subscribers: Dict[str, asyncio.Queue] = {}

# ── Disabled tool store (in-memory; could be persisted to DB) ─────────────────
_disabled_tools: set = set()

# ── Call log (last 200 entries) ───────────────────────────────────────────────
_call_log: List[Dict] = []
MAX_LOG = 200


def _log_call(tool_name: str, args: dict, result: dict, status: str, duration_ms: float):
    entry = {
        "id": uuid.uuid4().hex[:12],
        "tool": tool_name,
        "args": args,
        "result": result,
        "status": status,
        "duration_ms": round(duration_ms, 1),
        "ts": datetime.utcnow().isoformat(),
    }
    _call_log.append(entry)
    if len(_call_log) > MAX_LOG:
        _call_log.pop(0)
    # Broadcast to SSE subscribers
    _broadcast_sse({"type": "tool_call", "data": entry})
    return entry


def _broadcast_sse(event: dict):
    dead = []
    for sid, q in _sse_subscribers.items():
        try:
            q.put_nowait(event)
        except asyncio.QueueFull:
            dead.append(sid)
    for sid in dead:
        _sse_subscribers.pop(sid, None)


# ── Pydantic models ───────────────────────────────────────────────────────────

class JsonRpcRequest(BaseModel):
    jsonrpc: str = "2.0"
    id: Optional[Any] = None
    method: str
    params: Optional[Dict[str, Any]] = None


class ToolToggleRequest(BaseModel):
    enabled: bool


class ToolInvokeRequest(BaseModel):
    args: Dict[str, Any] = {}


# ── JSON-RPC helpers ──────────────────────────────────────────────────────────

def _ok(req_id, result):
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def _err(req_id, code: int, message: str, data=None):
    err = {"code": code, "message": message}
    if data:
        err["data"] = data
    return {"jsonrpc": "2.0", "id": req_id, "error": err}


# ── MCP method dispatcher ─────────────────────────────────────────────────────

def _dispatch(method: str, params: dict, req_id):
    if method == "tools/list":
        tools = []
        for t in tool_registry.list_tools():
            tools.append({
                **t,
                "enabled": t["name"] not in _disabled_tools,
                "schema": _get_tool_schema(t["name"]),
            })
        return _ok(req_id, {"tools": tools, "count": len(tools)})

    if method == "tools/call":
        name = params.get("name", "")
        args = params.get("arguments", {}) or {}

        if not name:
            return _err(req_id, -32602, "Missing tool name")
        if name not in {t["name"] for t in tool_registry.list_tools()}:
            return _err(req_id, -32601, f"Unknown tool: {name}")
        if name in _disabled_tools:
            return _err(req_id, -32000, f"Tool '{name}' is disabled")

        t0 = time.time()
        result = tool_registry.execute(name, args)
        latency = result.pop("_latency_ms", round((time.time() - t0) * 1000, 1))
        status = "error" if "error" in result else "ok"
        _log_call(name, args, result, status, latency)

        return _ok(req_id, {
            "content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False)}],
            "isError": status == "error",
            "_latency_ms": latency,
        })

    if method == "tools/toggle":
        name = params.get("name", "")
        enabled = params.get("enabled", True)
        if not name:
            return _err(req_id, -32602, "Missing tool name")
        if enabled:
            _disabled_tools.discard(name)
        else:
            _disabled_tools.add(name)
        _broadcast_sse({"type": "tool_toggled", "tool": name, "enabled": enabled})
        return _ok(req_id, {"name": name, "enabled": enabled})

    if method == "logs/list":
        limit = int(params.get("limit", 50))
        return _ok(req_id, {"logs": _call_log[-limit:][::-1], "total": len(_call_log)})

    if method == "logs/clear":
        _call_log.clear()
        return _ok(req_id, {"cleared": True})

    if method == "server/info":
        return _ok(req_id, {
            "name": "hw2-mcp-server",
            "version": "1.0.0",
            "protocol": "2024-11-05",
            "capabilities": {
                "tools": {"listChanged": True},
                "logging": {},
                "resources": {},
            },
            "tool_count": len(tool_registry.list_tools()),
            "disabled_count": len(_disabled_tools),
        })

    if method == "ping":
        return _ok(req_id, {"pong": True, "ts": datetime.utcnow().isoformat()})

    return _err(req_id, -32601, f"Method not found: {method}")


def _get_tool_schema(name: str) -> Optional[dict]:
    spec = tool_registry.get(name)
    if spec:
        return spec.parameters
    return None


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post("")
async def mcp_jsonrpc(req: JsonRpcRequest):
    """Standard MCP JSON-RPC 2.0 endpoint."""
    params = req.params or {}
    result = _dispatch(req.method, params, req.id)
    return JSONResponse(content=result)


@router.post("/batch")
async def mcp_batch(requests: List[JsonRpcRequest]):
    """Batch JSON-RPC 2.0 endpoint."""
    results = []
    for req in requests:
        params = req.params or {}
        results.append(_dispatch(req.method, params, req.id))
    return JSONResponse(content=results)


@router.get("/sse")
async def mcp_sse(request: Request):
    """SSE stream — clients subscribe here for real-time tool call events."""
    sid = uuid.uuid4().hex
    q: asyncio.Queue = asyncio.Queue(maxsize=100)
    _sse_subscribers[sid] = q

    async def event_stream() -> AsyncGenerator[str, None]:
        # Send connection confirmation
        yield f"data: {json.dumps({'type': 'connected', 'sid': sid, 'ts': datetime.utcnow().isoformat()})}\n\n"
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    event = await asyncio.wait_for(q.get(), timeout=25.0)
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                except asyncio.TimeoutError:
                    # Heartbeat
                    yield f": heartbeat {int(time.time())}\n\n"
        finally:
            _sse_subscribers.pop(sid, None)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@router.get("/tools")
async def list_tools():
    """List all tools with enabled/disabled state."""
    tools = []
    for t in tool_registry.list_tools():
        spec = tool_registry.get(t["name"])
        tools.append({
            **t,
            "enabled": t["name"] not in _disabled_tools,
            "parameters": spec.parameters if spec else {},
            "timeout_s": spec.timeout_s if spec else 15,
            "call_count": sum(1 for l in _call_log if l["tool"] == t["name"]),
            "error_count": sum(1 for l in _call_log if l["tool"] == t["name"] and l["status"] == "error"),
            "avg_latency_ms": _avg_latency(t["name"]),
        })
    return {"tools": tools, "total": len(tools), "disabled": len(_disabled_tools)}


def _avg_latency(name: str) -> float:
    calls = [l["duration_ms"] for l in _call_log if l["tool"] == name]
    return round(sum(calls) / len(calls), 1) if calls else 0.0


@router.post("/tools/{name}/invoke")
async def invoke_tool(name: str, body: ToolInvokeRequest):
    """Direct tool invocation endpoint."""
    spec = tool_registry.get(name)
    if not spec:
        raise HTTPException(status_code=404, detail=f"Tool '{name}' not found")
    if name in _disabled_tools:
        raise HTTPException(status_code=403, detail=f"Tool '{name}' is disabled")

    t0 = time.time()
    result = tool_registry.execute(name, body.args)
    latency = result.pop("_latency_ms", round((time.time() - t0) * 1000, 1))
    status = "error" if "error" in result else "ok"
    log_entry = _log_call(name, body.args, result, status, latency)

    return {
        "tool": name,
        "result": result,
        "status": status,
        "duration_ms": latency,
        "log_id": log_entry["id"],
    }


@router.post("/tools/{name}/toggle")
async def toggle_tool(name: str, body: ToolToggleRequest):
    """Enable or disable a tool."""
    spec = tool_registry.get(name)
    if not spec:
        raise HTTPException(status_code=404, detail=f"Tool '{name}' not found")
    if body.enabled:
        _disabled_tools.discard(name)
    else:
        _disabled_tools.add(name)
    _broadcast_sse({"type": "tool_toggled", "tool": name, "enabled": body.enabled})
    return {"name": name, "enabled": body.enabled}


@router.get("/logs")
async def get_logs(limit: int = 50, tool: Optional[str] = None):
    """Get recent tool call logs."""
    logs = _call_log[-200:][::-1]
    if tool:
        logs = [l for l in logs if l["tool"] == tool]
    return {"logs": logs[:limit], "total": len(_call_log)}


@router.delete("/logs")
async def clear_logs():
    """Clear all tool call logs."""
    _call_log.clear()
    return {"cleared": True}


@router.get("/health")
async def health():
    """MCP server health check."""
    return {
        "status": "ok",
        "ts": datetime.utcnow().isoformat(),
        "tools": len(tool_registry.list_tools()),
        "disabled": len(_disabled_tools),
        "log_entries": len(_call_log),
        "sse_subscribers": len(_sse_subscribers),
        "protocol": "MCP/2024-11-05",
    }