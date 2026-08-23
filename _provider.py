"""RecallMemoryProvider — the MemoryProvider ABC implementation.

Every public method fails open: it wraps its body and returns the neutral
value rather than letting an exception reach the agent loop.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

from agent.memory_provider import MemoryProvider

from ._client import RecallAuthError, RecallClient
from ._filters import is_trivial_prompt, truncate

try:  # Present from Hermes 0.20; absent on 0.19.1.
    from agent.memory_provider import RecallStatus
except Exception:  # pragma: no cover - 0.19.1 fallback
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class RecallStatus:  # type: ignore[no-redef]
        provider_label: str
        count: int
        glyph: str = "🧠"


try:
    from tools.registry import tool_error  # noqa: F401  (used from Task 8 on)
except Exception:  # pragma: no cover - bare test environment

    def tool_error(message: Any, **extra: Any) -> str:  # type: ignore[misc]
        result: dict[str, Any] = {"error": str(message)}
        if extra:
            result.update(extra)
        return json.dumps(result, ensure_ascii=False)


logger = logging.getLogger(__name__)

CONFIG_FILENAME = "recall.json"
SNIPPET_CHARS = 300
PROVIDER_LABEL = "Recall"
GLYPH = "🧠"

DEFAULTS: dict[str, Any] = {
    "base_url": "https://recall.carnival-devops.com",
    "limit": 5,
    "rerank": True,
    "graph_boost": False,
    "sync_turns": True,
    "session_summary": True,
    "max_chars": 4000,
    "min_chars": 40,
}


def _format_memories(items: list[dict[str, Any]]) -> tuple[str, int]:
    """Render search items as the injected block. Returns ``(text, count)``."""
    lines = []
    for item in items:
        snippet = str(item.get("snippet") or "").strip().replace("\n", " ")
        if not snippet:
            continue
        mem_type = str(item.get("type") or "memory").strip() or "memory"
        stamp = str(item.get("timestamp") or "")[:10]
        prefix = f"[{mem_type}, {stamp}]" if stamp else f"[{mem_type}]"
        lines.append(f"- {prefix} {truncate(snippet, SNIPPET_CHARS)}")
    if not lines:
        return "", 0
    return "Relevant memories (Recall):\n" + "\n".join(lines), len(lines)


def load_recall_config(hermes_home: str) -> dict[str, Any]:
    """Merge ``$HERMES_HOME/recall.json`` and the env over the defaults."""
    config = dict(DEFAULTS)
    try:
        path = Path(hermes_home) / CONFIG_FILENAME
        if path.is_file():
            raw = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                for key, value in raw.items():
                    if key in DEFAULTS:
                        config[key] = value
    except Exception as exc:
        logger.warning("Recall config unreadable (%s) — using defaults", type(exc).__name__)

    env_base_url = os.environ.get("RECALL_BASE_URL", "").strip()
    if env_base_url:
        config["base_url"] = env_base_url
    config["base_url"] = str(config["base_url"]).rstrip("/")
    return config


class RecallMemoryProvider(MemoryProvider):
    """Recall as Hermes' persistent cross-session memory."""

    def __init__(self, client: RecallClient | None = None) -> None:
        self._client = client or RecallClient()
        self._config: dict[str, Any] = dict(DEFAULTS)
        self._session_id = ""
        self._hermes_home = ""
        self._platform = ""
        self._agent_identity = ""
        self._agent_context = "primary"
        self._writes_enabled = True
        self._auth_warned = False
        self._prefetch_cache: dict[str, str] = {}
        self._prefetch_counts: dict[str, int] = {}
        self._last_count: int = 0
        self._threads: dict[str, threading.Thread] = {}

    # -- identity / availability ------------------------------------------

    @property
    def name(self) -> str:
        return "recall"

    def is_available(self) -> bool:
        try:
            return bool(self._client.api_key)
        except Exception:
            return False

    def unavailable_reason(self) -> str:
        return "Set RECALL_API_KEY (Recall → Settings → API keys)"

    def backup_paths(self) -> list[str]:
        return []

    def system_prompt_block(self) -> str:
        return (
            "# Recall Memory\n"
            "Recall is active as persistent cross-session memory. Relevant memories are "
            "injected automatically before each turn under 'Relevant memories (Recall)'. "
            "Use recall_search for anything about the past; use recall_store to pin a "
            "decision, preference or resolved bug worth keeping."
        )

    # -- lifecycle ---------------------------------------------------------

    def initialize(self, session_id: str, **kwargs: Any) -> None:
        try:
            self._session_id = session_id or ""
            self._hermes_home = str(kwargs.get("hermes_home") or "")
            self._platform = str(kwargs.get("platform") or "")
            self._agent_identity = str(kwargs.get("agent_identity") or "")
            self._agent_context = str(kwargs.get("agent_context") or "primary")
            self._writes_enabled = self._agent_context == "primary"
            self._auth_warned = False
            self._prefetch_cache.clear()
            self._prefetch_counts.clear()
            self._last_count = 0

            self._config = load_recall_config(self._hermes_home)
            self._client.base_url = str(self._config["base_url"]).rstrip("/")
        except Exception as exc:
            logger.warning("Recall initialize failed: %s", exc)

    # -- tools -------------------------------------------------------------

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        """Placeholder so the ABC is satisfied; Task 8 declares the real tools."""
        return []

    # -- config ------------------------------------------------------------

    def get_config_schema(self) -> list[dict[str, Any]]:
        return [
            {
                "key": "api_key",
                "description": "Recall API key",
                "secret": True,
                "required": True,
                "env_var": "RECALL_API_KEY",
                "url": "https://recall.carnival-devops.com",
            }
        ]

    def save_config(self, values: dict[str, Any], hermes_home: str) -> None:
        try:
            payload = {k: v for k, v in (values or {}).items() if k in DEFAULTS}
            path = Path(hermes_home) / CONFIG_FILENAME
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        except Exception as exc:
            logger.warning("Recall save_config failed: %s", exc)

    # -- internals ---------------------------------------------------------

    def _tags(self, *extra: str) -> list[str]:
        tags = ["hermes"]
        if self._session_id:
            tags.append(f"session:{self._session_id}")
        if self._platform:
            tags.append(f"platform:{self._platform}")
        if self._agent_identity:
            tags.append(f"agent:{self._agent_identity}")
        tags.extend(t for t in extra if t)
        return tags

    def _log_failure(self, operation: str, exc: BaseException) -> None:
        if isinstance(exc, RecallAuthError):
            if self._auth_warned:
                return
            self._auth_warned = True
            logger.warning("Recall API key rejected during %s — update RECALL_API_KEY", operation)
            return
        logger.warning("Recall %s failed: %s", operation, exc)

    # -- threading ---------------------------------------------------------

    def _spawn(self, name: str, target) -> None:
        """Start a daemon thread, joining the previous one of the same name."""
        previous = self._threads.get(name)
        if previous is not None and previous.is_alive():
            previous.join(timeout=5.0)
        thread = threading.Thread(target=target, daemon=True, name=f"recall-{name}")
        self._threads[name] = thread
        thread.start()

    # -- recall ------------------------------------------------------------

    def _search_and_cache(self, query: str, session_id: str) -> str:
        """Run one search and cache the rendered block. Returns "" on failure."""
        try:
            items = self._client.search(
                query,
                limit=int(self._config.get("limit", 5)),
                rerank=bool(self._config.get("rerank", True)),
                graph_boost=bool(self._config.get("graph_boost", False)),
            )
        except Exception as exc:
            self._log_failure("search", exc)
            return ""
        block, count = _format_memories(items)
        key = session_id or self._session_id
        if block:
            self._prefetch_cache[key] = block
            self._prefetch_counts[key] = count
        else:
            self._prefetch_cache.pop(key, None)
            self._prefetch_counts.pop(key, None)
        return block

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Return the memory block to inject. Never raises, never blocks >3 s."""
        try:
            if is_trivial_prompt(query):
                self._last_count = 0
                return ""
            key = session_id or self._session_id
            cached = self._prefetch_cache.pop(key, None)
            if cached:
                self._last_count = self._prefetch_counts.pop(key, 0)
                return cached
            block = self._search_and_cache(query, key)
            # A cold synchronous hit is consumed immediately, not left cached.
            self._last_count = self._prefetch_counts.pop(key, 0)
            self._prefetch_cache.pop(key, None)
            return block
        except Exception as exc:
            self._log_failure("prefetch", exc)
            self._last_count = 0
            return ""

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        """Warm the cache in the background for the next turn."""
        try:
            if is_trivial_prompt(query):
                return
            key = session_id or self._session_id
            self._spawn("prefetch", lambda: self._search_and_cache(query, key))
        except Exception as exc:
            self._log_failure("queue_prefetch", exc)

    def recall_status(self) -> RecallStatus | None:
        """Describe ONLY the most recent prefetch. Never a stale count."""
        try:
            if not self._last_count:
                return None
            return RecallStatus(
                provider_label=PROVIDER_LABEL, count=self._last_count, glyph=GLYPH
            )
        except Exception:
            return None

    # -- shutdown ----------------------------------------------------------

    def shutdown(self) -> None:
        """Join every live background thread within a 5 s total budget."""
        deadline = time.monotonic() + 5.0
        for thread in list(self._threads.values()):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            if thread.is_alive():
                thread.join(timeout=remaining)
