"""
tool_cache.py — Simple TTL cache for MCP tool results
Avoids redundant network calls for weather, web_search, etc.

Usage in tools.py / app.py:

    from tool_cache import tool_cache

    # Before calling the real tool:
    cached = tool_cache.get("get_weather", {"location": "Taipei"})
    if cached:
        return cached

    result = _real_weather_call(...)
    tool_cache.set("get_weather", {"location": "Taipei"}, result, ttl=600)
    return result
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Optional


# ── Default TTLs per tool (seconds) ──────────────────────────────────────────
DEFAULT_TTLS: dict[str, int] = {
    "get_weather":   600,    # 10 min  — weather changes slowly
    "web_search":    300,    # 5 min   — search results
    "get_datetime":  0,      # never cache — always real-time
    "calculator":    86400,  # 1 day   — deterministic
    "generate_image": 0,     # never cache — generative
    "create_presentation": 0,  # never cache — generative
}

GLOBAL_DEFAULT_TTL = 120  # 2 min fallback for unknown tools


@dataclass
class _CacheEntry:
    value: Any
    expires_at: float  # unix timestamp; 0 = never expires (disabled)


class ToolCache:
    """Thread-safe in-memory TTL cache for tool results."""

    def __init__(self) -> None:
        self._store: dict[str, _CacheEntry] = {}
        self._hits: int = 0
        self._misses: int = 0

    # ── Cache key ─────────────────────────────────────────────────────────────

    @staticmethod
    def _make_key(tool_name: str, args: dict) -> str:
        """
        Deterministic cache key.
        Args are sorted so {"a":1,"b":2} == {"b":2,"a":1}.
        """
        canonical = json.dumps(args, sort_keys=True, ensure_ascii=False)
        digest = hashlib.sha256(f"{tool_name}:{canonical}".encode()).hexdigest()[:16]
        return f"{tool_name}:{digest}"

    # ── Public API ────────────────────────────────────────────────────────────

    def get(self, tool_name: str, args: dict) -> Optional[Any]:
        """Return cached value or None if missing / expired."""
        ttl = DEFAULT_TTLS.get(tool_name, GLOBAL_DEFAULT_TTL)
        if ttl == 0:
            # This tool is deliberately never cached
            self._misses += 1
            return None

        key = self._make_key(tool_name, args)
        entry = self._store.get(key)
        if entry is None:
            self._misses += 1
            return None

        if entry.expires_at and time.time() > entry.expires_at:
            del self._store[key]
            self._misses += 1
            return None

        self._hits += 1
        # Attach cache metadata so callers can see it was a cache hit
        result = dict(entry.value)
        result["_cache_hit"] = True
        return result

    def set(
        self,
        tool_name: str,
        args: dict,
        value: Any,
        ttl: Optional[int] = None,
    ) -> None:
        """
        Store a result.
        - ttl=None  → use DEFAULT_TTLS or GLOBAL_DEFAULT_TTL
        - ttl=0     → skip storing (non-cacheable tool)
        """
        effective_ttl = ttl if ttl is not None else DEFAULT_TTLS.get(tool_name, GLOBAL_DEFAULT_TTL)
        if effective_ttl == 0:
            return

        key = self._make_key(tool_name, args)
        expires_at = time.time() + effective_ttl
        # Store a copy without internal metadata keys
        clean_value = {k: v for k, v in value.items() if not k.startswith("_")}
        self._store[key] = _CacheEntry(value=clean_value, expires_at=expires_at)

    def invalidate(self, tool_name: Optional[str] = None) -> int:
        """
        Remove cached entries.
        - tool_name=None  → flush everything
        - tool_name=str   → flush only that tool's entries
        Returns number of entries removed.
        """
        if tool_name is None:
            count = len(self._store)
            self._store.clear()
            return count

        keys_to_del = [k for k in self._store if k.startswith(f"{tool_name}:")]
        for k in keys_to_del:
            del self._store[k]
        return len(keys_to_del)

    def evict_expired(self) -> int:
        """Remove all expired entries. Call periodically if needed."""
        now = time.time()
        expired = [k for k, e in self._store.items() if e.expires_at and now > e.expires_at]
        for k in expired:
            del self._store[k]
        return len(expired)

    @property
    def stats(self) -> dict:
        self.evict_expired()
        return {
            "entries": len(self._store),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / max(1, self._hits + self._misses), 3),
            "tools": list({k.split(":")[0] for k in self._store}),
        }


# Singleton — import this everywhere
tool_cache = ToolCache()