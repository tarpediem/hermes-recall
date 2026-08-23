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

from ._client import (
    MEMORY_TYPES,
    READ_TIMEOUT,
    SLOW_READ_TIMEOUT,
    RecallAuthError,
    RecallClient,
)
from ._filters import (
    condense_turn,
    extract_insights,
    is_trivial_prompt,
    is_worth_storing,
    summarize_session,
    truncate,
)

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

CONFIG_DIRNAME = "recall"
CONFIG_FILENAME = "config.json"
SNIPPET_CHARS = 300
# Hard cap on concurrently live background threads. Reached only when Recall
# is wedged (each write can hang for WRITE_TIMEOUT); beyond it, writes are
# dropped with a warning rather than queued or awaited.
MAX_LIVE_THREADS = 16
SHUTDOWN_BUDGET_SECONDS = 5.0
SESSION_MIN_TURNS = 2
PRE_COMPRESS_MAX_INSIGHTS = 8
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


SEARCH_TOOL_SCHEMA: dict[str, Any] = {
    "name": "recall_search",
    "description": (
        "Search Recall, the persistent cross-session memory, for anything from the "
        "past: prior decisions and their rationale, resolved bugs, people, machines, "
        "configs, procedures. Use it before asking the user something that may "
        "already be known."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to look for."},
            "limit": {
                "type": "integer",
                "description": "How many memories to return (1-100).",
                "default": 5,
            },
        },
        "required": ["query"],
    },
}

STORE_TOOL_SCHEMA: dict[str, Any] = {
    "name": "recall_store",
    "description": (
        "Pin something durable in Recall: a decision and why it was taken, a lasting "
        "preference, a resolved bug (symptom + cause + fix), a procedure. Turns and "
        "session summaries are already written automatically — do not restate the "
        "current turn or re-store memories that were injected into this context."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "The memory to store."},
            "memory_type": {
                "type": "string",
                "description": "Kind of memory.",
                "enum": ["context", "decision", "bugfix", "architecture", "preference", "snippet"],
                "default": "context",
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Extra tags to attach.",
                "default": [],
            },
        },
        "required": ["content"],
    },
}


def recall_config_path(hermes_home: str) -> Path:
    """``$HERMES_HOME/recall/config.json`` — where the dashboard panel writes.

    ``config_schema.py`` declares the default ``STORAGE_FLAT_JSON`` backend,
    whose path contract is ``get_hermes_home() / <provider name> /
    "config.json"`` (``hermes_cli/web_server.py::_flat_json_path``). The
    provider must read exactly that file: any other location and every
    non-secret field saved from the panel would be written somewhere nothing
    ever loads. Same convention as the shipped hindsight provider.
    """
    return Path(hermes_home) / CONFIG_DIRNAME / CONFIG_FILENAME


def load_recall_config(hermes_home: str) -> dict[str, Any]:
    """Merge ``$HERMES_HOME/recall/config.json`` and the env over the defaults."""
    config = dict(DEFAULTS)
    try:
        path = recall_config_path(hermes_home)
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
        # One atomic entry per session: (block, count). Published as a single
        # dict assignment so a concurrent prefetch() can never observe a block
        # without its count (see _search_and_cache).
        self._prefetch_cache: dict[str, tuple[str, int]] = {}
        # Declared by the provider skeleton; superseded by the tuple above and
        # kept only so nothing that touches the attribute breaks.
        self._prefetch_counts: dict[str, int] = {}
        self._last_count: int = 0
        # Live background threads, newest last. A list — not a dict keyed by
        # name — because a per-name registry forced _spawn to join the previous
        # thread of the same name, which put a join (up to WRITE_TIMEOUT) on the
        # turn path. Joins now happen only at a session boundary or shutdown.
        self._threads: list[threading.Thread] = []
        self._threads_lock = threading.Lock()
        # Bumped whenever the prefetch cache is flushed wholesale. A background
        # search carries the generation it was spawned under and drops its
        # result if the cache has been flushed since — otherwise an in-flight
        # prefetch writes its entry back after the flush, under a session key
        # that will never be read or evicted again.
        self._cache_generation = 0

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
            self._cache_generation += 1
            self._last_count = 0
            # Forget finished threads; deliberately do NOT join live ones. A
            # store from the previous session is a daemon thread writing a
            # valid memory — it is left to finish on its own, and re-init stays
            # fast even mid-flight.
            self._prune_threads()

            self._config = load_recall_config(self._hermes_home)
            self._client.base_url = str(self._config["base_url"]).rstrip("/")
        except Exception as exc:
            logger.warning("Recall initialize failed: %s", exc)

    # -- tools -------------------------------------------------------------

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        return [SEARCH_TOOL_SCHEMA, STORE_TOOL_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: dict[str, Any], **kwargs: Any) -> str:
        try:
            if tool_name == "recall_search":
                return self._tool_search(args or {})
            if tool_name == "recall_store":
                return self._tool_store(args or {})
            return tool_error(f"Unknown tool: {tool_name}")
        except Exception as exc:
            self._log_failure(f"tool {tool_name}", exc)
            return tool_error("Recall tool call failed")

    def _tool_search(self, args: dict[str, Any]) -> str:
        query = str(args.get("query") or "").strip()
        if not query:
            return tool_error("query is required")
        try:
            limit = int(args.get("limit") or self._config.get("limit", 5))
        except (TypeError, ValueError):
            limit = int(self._config.get("limit", 5))
        try:
            items = self._client.search(
                query,
                limit=limit,
                rerank=bool(self._config.get("rerank", True)),
                graph_boost=bool(self._config.get("graph_boost", False)),
                # An explicit tool call is not the turn-path injection: the
                # model asked for this one and a reranked query really takes
                # ~4.5 s, so the 3 s turn budget would make the tool useless.
                timeout=SLOW_READ_TIMEOUT,
            )
        except Exception as exc:
            self._log_failure("search", exc)
            return tool_error("Recall search failed")

        results = []
        for item in items:
            # Same rule as the injected block: an item with no snippet is not a
            # result. Without this, a degenerate payload makes the tool report
            # hits the model cannot read, and `count` overstates what was found.
            snippet = truncate(str(item.get("snippet") or "").strip(), SNIPPET_CHARS)
            if not snippet:
                continue
            results.append(
                {
                    "id": item.get("id", ""),
                    "type": item.get("type", "memory"),
                    "timestamp": item.get("timestamp", ""),
                    "score": item.get("score"),
                    "snippet": snippet,
                }
            )
        return json.dumps({"count": len(results), "results": results}, ensure_ascii=False)

    def _tool_store(self, args: dict[str, Any]) -> str:
        content = str(args.get("content") or "").strip()
        if not content:
            return tool_error("content is required")
        memory_type = str(args.get("memory_type") or "context")
        if memory_type not in MEMORY_TYPES:
            return tool_error(f"memory_type must be one of {sorted(MEMORY_TYPES)}")
        if not self._writes_enabled:
            return tool_error(
                f"Recall writes are disabled in the '{self._agent_context}' agent context"
            )

        extra = args.get("tags") or []
        extra_tags = [str(t) for t in extra if str(t).strip()] if isinstance(extra, list) else []
        tags = self._tags("model-tool", *extra_tags)
        body = truncate(content, int(self._config.get("max_chars", 4000)))
        try:
            memory_id = self._client.store(
                body, memory_type=memory_type, scope="project", tags=tags
            )
        except Exception as exc:
            self._log_failure("store", exc)
            return tool_error("Recall store failed")
        return json.dumps({"stored": True, "memory_id": memory_id}, ensure_ascii=False)

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
            path = recall_config_path(hermes_home)
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

    def _prune_threads(self) -> None:
        """Forget finished threads. Never joins — safe on the turn path."""
        with self._threads_lock:
            self._threads = [t for t in self._threads if t.is_alive()]

    def _join_threads(self, budget: float) -> None:
        """Join live threads within a total ``budget`` in seconds, then prune.

        Only ever called at a boundary (``shutdown``, ``on_session_switch``
        with ``reset=True``) — never from ``prefetch``/``sync_turn``/mirrors.
        """
        deadline = time.monotonic() + budget
        with self._threads_lock:
            snapshot = list(self._threads)
        # Joined OUTSIDE the lock: holding it here would block _spawn (and so
        # the turn path) for the whole budget.
        for thread in snapshot:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            if thread.is_alive():
                thread.join(timeout=remaining)
        self._prune_threads()

    def _spawn(self, name: str, target) -> None:
        """Start a daemon thread. O(1): it never joins, so callers never block.

        The target runs wholly inside a failure handler, so no background
        exception — from any target — can reach ``threading.excepthook`` and
        print a traceback into the agent's terminal.
        """

        def _run() -> None:
            try:
                target()
            except Exception as exc:  # noqa: BLE001 - the point is to catch everything
                self._log_failure("background", exc)

        thread = threading.Thread(target=_run, daemon=True, name=f"recall-{name}")
        with self._threads_lock:
            self._threads = [t for t in self._threads if t.is_alive()]
            if len(self._threads) >= MAX_LIVE_THREADS:
                # A wedged Recall must not grow the pool without bound. Drop the
                # work rather than block: memory is best-effort by design.
                logger.warning(
                    "Recall %s dropped: %d background writes already in flight",
                    name,
                    len(self._threads),
                )
                return
            self._threads.append(thread)
            # Started under the lock: a not-yet-started thread reads as
            # is_alive() == False, so a concurrent prune would otherwise drop
            # it from the registry and shutdown would never join it.
            thread.start()

    # -- recall ------------------------------------------------------------

    def _search_and_cache(
        self,
        query: str,
        session_id: str,
        generation: int | None = None,
        timeout: float | None = None,
        rerank: bool | None = None,
    ) -> str:
        """Run one search and cache the rendered block. Returns "" on failure.

        ``timeout`` defaults to the client's turn-path budget and ``rerank`` to
        the configured value; the two callers off the turn path and on it
        override them in opposite directions (see ``prefetch`` /
        ``queue_prefetch``).
        """
        try:
            items = self._client.search(
                query,
                limit=int(self._config.get("limit", 5)),
                rerank=bool(self._config.get("rerank", True)) if rerank is None else bool(rerank),
                graph_boost=bool(self._config.get("graph_boost", False)),
                timeout=timeout,
            )
        except Exception as exc:
            self._log_failure("search", exc)
            return ""
        block, count = _format_memories(items)
        if generation is not None and generation != self._cache_generation:
            # The cache was flushed while this search was in flight (re-init or
            # a session reset). Return the block to a synchronous caller but do
            # not resurrect an entry under the dead session's key.
            return block
        key = session_id or self._session_id
        if block:
            # Single assignment: block and count are published together.
            self._prefetch_cache[key] = (block, count)
        else:
            self._prefetch_cache.pop(key, None)
        return block

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Return the memory block to inject. Never raises, never blocks >3 s.

        A cache miss — the first turn of a session, or a warm-up that has not
        landed yet — is served by a synchronous search that DROPS rerank. A
        reranked query costs ~4.5 s and could never fit the 3 s turn budget, so
        the alternative to an unreranked hit here is no memory at all on the
        one turn where cross-session recall matters most. Every warm turn after
        it is served from the reranked warm-up.
        """
        try:
            if is_trivial_prompt(query):
                self._last_count = 0
                return ""
            key = session_id or self._session_id
            entry = self._prefetch_cache.pop(key, None)
            if entry is None:
                self._search_and_cache(query, key, timeout=READ_TIMEOUT, rerank=False)
                # A cold synchronous hit is consumed immediately, not left cached.
                entry = self._prefetch_cache.pop(key, None)
            block, count = entry if entry else ("", 0)
            self._last_count = count
            return block
        except Exception as exc:
            self._log_failure("prefetch", exc)
            self._last_count = 0
            return ""

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        """Warm the cache in the background for the next turn.

        Off the turn path, so it keeps the configured ``rerank`` and gets the
        longer budget: a reranked search against a real instance takes ~4.5 s,
        and warming with the 3 s turn-path budget meant the cache was never
        filled at all.
        """
        try:
            if is_trivial_prompt(query):
                return
            key = session_id or self._session_id
            generation = self._cache_generation
            self._spawn(
                "prefetch",
                lambda: self._search_and_cache(query, key, generation, SLOW_READ_TIMEOUT),
            )
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

    # -- writes --------------------------------------------------------------

    def _store_async(
        self,
        thread_name: str,
        content: str,
        *,
        memory_type: str,
        tags: list[str],
    ) -> None:
        """Single write path: gated on agent_context, off-thread, fail-open."""
        if not self._writes_enabled or not content:
            return

        def _run() -> None:
            try:
                self._client.store(
                    content, memory_type=memory_type, scope="project", tags=tags
                )
            except Exception as exc:
                self._log_failure("store", exc)

        self._spawn(thread_name, _run)

    def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
        messages: list[dict[str, Any]] | None = None,
    ) -> None:
        """Persist a completed turn.

        ``messages`` is accepted and deliberately IGNORED: only the
        user/assistant pair may leave the device, so tool calls and tool
        results cannot leak workspace paths or command output into Recall.
        """
        try:
            if not self._writes_enabled:
                return
            if not bool(self._config.get("sync_turns", True)):
                return
            min_chars = int(self._config.get("min_chars", 40))
            if not is_worth_storing(user_content, assistant_content, min_chars=min_chars):
                return
            content = condense_turn(
                user_content,
                assistant_content,
                max_chars=int(self._config.get("max_chars", 4000)),
            )
            session_tag = f"session:{session_id}" if session_id else ""
            tags = ["hermes"]
            tags.append(session_tag or (f"session:{self._session_id}" if self._session_id else ""))
            if self._platform:
                tags.append(f"platform:{self._platform}")
            if self._agent_identity:
                tags.append(f"agent:{self._agent_identity}")
            tags = [t for t in tags if t]
            self._store_async("sync", content, memory_type="context", tags=tags)
        except Exception as exc:
            self._log_failure("sync_turn", exc)

    def on_memory_write(
        self,
        action: str,
        target: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Mirror a built-in memory tool write into Recall.

        ``metadata`` is a defaulted keyword, so Hermes 0.19.1's
        three-positional-argument call site works unchanged.
        """
        try:
            if action not in {"add", "replace"}:
                return  # 'remove' is a documented no-op: nothing is deleted
            text = (content or "").strip()
            if not text:
                return
            memory_type = "preference" if target == "user" else "context"
            label = "User profile" if target == "user" else "Agent memory"
            body = truncate(
                f"[{label}] {text}", int(self._config.get("max_chars", 4000))
            )
            self._store_async(
                "memwrite",
                body,
                memory_type=memory_type,
                tags=self._tags("builtin-mirror"),
            )
        except Exception as exc:
            self._log_failure("on_memory_write", exc)

    def on_delegation(
        self,
        task: str,
        result: str,
        *,
        child_session_id: str = "",
        **kwargs: Any,
    ) -> None:
        """Store what was delegated and what came back, as one memory."""
        try:
            task_text = (task or "").strip()
            result_text = (result or "").strip()
            if not task_text or not result_text:
                return
            body = truncate(
                f"Delegated task: {task_text}\nResult: {result_text}",
                int(self._config.get("max_chars", 4000)),
            )
            self._store_async(
                "delegation", body, memory_type="context", tags=self._tags("delegation")
            )
        except Exception as exc:
            self._log_failure("on_delegation", exc)

    # -- session lifecycle -------------------------------------------------

    def on_session_end(self, messages: list[dict[str, Any]]) -> None:
        """Write ONE synthesis of the session — not a dump of every turn."""
        try:
            if not self._writes_enabled:
                return
            if not bool(self._config.get("session_summary", True)):
                return
            summary = summarize_session(
                messages,
                max_chars=int(self._config.get("max_chars", 4000)),
                min_turns=SESSION_MIN_TURNS,
            )
            if summary is None:
                return
            content, memory_type = summary
            self._store_async(
                "summary", content, memory_type=memory_type, tags=self._tags("session-summary")
            )
        except Exception as exc:
            self._log_failure("on_session_end", exc)

    def on_session_switch(
        self,
        new_session_id: str,
        *,
        parent_session_id: str = "",
        reset: bool = False,
        rewound: bool = False,
        **kwargs: Any,
    ) -> None:
        """Re-scope cached state so writes land in the right session's record.

        A plain switch (a subagent handoff) stays O(1). ``reset`` is a real
        session boundary, so it is the one place besides ``shutdown`` where
        joining in-flight writes is allowed — the old session's memories are
        flushed before its state is dropped.
        """
        try:
            if rewound:
                key = new_session_id or self._session_id
                self._prefetch_cache.pop(key, None)
                self._prefetch_counts.pop(key, None)
            if reset:
                self._join_threads(SHUTDOWN_BUDGET_SECONDS)
                self._prefetch_cache.clear()
                self._prefetch_counts.clear()
                self._cache_generation += 1
                self._last_count = 0
                self._auth_warned = False
            if new_session_id:
                self._session_id = new_session_id
        except Exception as exc:
            self._log_failure("on_session_switch", exc)

    def on_pre_compress(self, messages: list[dict[str, Any]]) -> str:
        """Archive what is about to be discarded, and tell the compressor
        what Recall judged worth keeping.

        (a) background: condense the discarded messages and store them with
            the ``pre-compress`` tag.
        (b) synchronous: return an insight block that Hermes feeds into the
            compression summary prompt.
        """
        try:
            archive = summarize_session(
                messages,
                max_chars=int(self._config.get("max_chars", 4000)),
                min_turns=SESSION_MIN_TURNS,
            )
            if archive is not None:
                content, memory_type = archive
                self._store_async(
                    "precompress",
                    content,
                    memory_type=memory_type,
                    tags=self._tags("pre-compress"),
                )

            insights = extract_insights(messages, max_items=PRE_COMPRESS_MAX_INSIGHTS)
            if not insights:
                return ""
            lines = ["Insights to preserve from the discarded context:"]
            lines += [f"- {item}" for item in insights]
            return "\n".join(lines)
        except Exception as exc:
            self._log_failure("on_pre_compress", exc)
            return ""

    # -- shutdown ----------------------------------------------------------

    def shutdown(self) -> None:
        """Join every live background thread within a 5 s total budget.

        Whatever did not finish inside the budget is abandoned, so the cache
        generation is bumped: a warm-up that outlives the join must discard its
        result rather than write it back under a key nothing will read again.
        Same rule as ``initialize`` and a session ``reset``.
        """
        try:
            self._join_threads(SHUTDOWN_BUDGET_SECONDS)
            self._cache_generation += 1
        except Exception as exc:
            self._log_failure("shutdown", exc)
