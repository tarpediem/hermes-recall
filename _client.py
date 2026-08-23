"""Thin REST client for the Recall API.

No Hermes imports, no threading, no logging of secrets. Every failure is
raised as a typed exception; the provider decides how to fail open.
"""

from __future__ import annotations

import os
from typing import Any

import requests

DEFAULT_BASE_URL = "https://recall.carnival-devops.com"

# Explicit browser-style UA: Cloudflare fronts the instance and has answered
# 403 (code 1010) to default Python user agents in the past.
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/128.0.0.0 Safari/537.36 hermes-recall/1.0"
)

# Budget for a search that sits on the turn path: the agent waits on it, so it
# must give up fast and let the provider inject nothing.
READ_TIMEOUT = 3.0
# Budget for a search that does NOT sit on the turn path — the background
# prefetch warm-up and the explicit `recall_search` tool. A reranked query
# costs a cross-encoder round-trip on the ML API GPU and measures 4.3-4.6 s
# against the public instance, so capping those at READ_TIMEOUT made every
# reranked search time out and the plugin injected nothing at all.
SLOW_READ_TIMEOUT = 10.0
WRITE_TIMEOUT = 10.0
# Budget for establishing the TCP/TLS connection, spent BEFORE any of the
# budgets above. ``requests`` applies a scalar timeout to the connect phase
# and the read phase separately, so a scalar 3.0 is really "up to 3 s to
# connect, then up to 3 s to read" — every call below therefore passes the
# explicit ``(connect, read)`` tuple, which is what the documented budget
# actually costs. 1.5 s is generous for a TLS handshake and short enough that
# an unreachable host fails fast.
CONNECT_TIMEOUT = 1.5

MAX_QUERY_CHARS = 2000
MIN_LIMIT = 1
MAX_LIMIT = 100
# Ranges the Recall graph endpoints validate their query string against: a
# value outside them is a 422, so they are clamped here rather than sent.
MIN_DEPTH = 1
MAX_DEPTH = 5
MAX_GRAPH_LIMIT = 50
MAX_WHO_KNOWS_RESULTS = 20

SEARCH_PATH = "/api/v1/memory/search"
STORE_PATH = "/api/v1/memories"
GRAPH_RECALL_PATH = "/api/v1/graph/recall"
GRAPH_STATS_PATH = "/api/v1/graph/stats"
WHO_KNOWS_PATH = "/api/v1/graph/who-knows"
SEARCH_STATS_PATH = "/api/v1/search/stats"

# Canonical values accepted by the Recall StoreRequest model.
MEMORY_TYPES = frozenset(
    {"context", "decision", "bugfix", "architecture", "preference", "snippet"}
)
SCOPES = frozenset({"project", "global"})


class RecallError(Exception):
    """Any Recall API failure. Never carries the API key."""


class RecallAuthError(RecallError):
    """The API key was rejected (HTTP 401/403)."""


class RecallClient:
    """Minimal REST client for Recall.

    ``api_key`` and ``base_url`` are public attributes so the provider can
    apply a ``recall/config.json`` override after construction.
    """

    def __init__(self, api_key: str = "", base_url: str = "") -> None:
        self.api_key = api_key or os.environ.get("RECALL_API_KEY", "")
        self.base_url = (
            base_url or os.environ.get("RECALL_BASE_URL", "") or DEFAULT_BASE_URL
        ).rstrip("/")

    def __repr__(self) -> str:  # never leak the key
        return f"<RecallClient base_url={self.base_url!r} configured={bool(self.api_key)}>"

    # -- internals ---------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"ApiKey {self.api_key}",
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    @staticmethod
    def _check(response: Any, operation: str) -> None:
        status = getattr(response, "status_code", 0)
        if status in (401, 403):
            raise RecallAuthError(f"Recall {operation} rejected the API key (HTTP {status})")
        if status < 200 or status >= 300:
            raise RecallError(f"Recall {operation} returned HTTP {status}")

    @staticmethod
    def _json(response: Any, operation: str) -> dict[str, Any]:
        try:
            payload = response.json()
        except Exception as exc:  # malformed body
            raise RecallError(f"Recall {operation} returned a malformed body") from exc
        if not isinstance(payload, dict):
            raise RecallError(f"Recall {operation} returned an unexpected payload type")
        return payload

    @classmethod
    def _json_any(cls, response: Any, operation: str) -> dict[str, Any] | list[Any]:
        """Like ``_json`` but a top-level list is legitimate too.

        ``/graph/recall`` may answer either an object or a bare array of
        entities depending on the query; anything else (a scalar, a string) is
        still a malformed body.
        """
        try:
            payload = response.json()
        except Exception as exc:
            raise RecallError(f"Recall {operation} returned a malformed body") from exc
        if not isinstance(payload, (dict, list)):
            raise RecallError(f"Recall {operation} returned an unexpected payload type")
        return payload

    @staticmethod
    def _clamp(value: Any, low: int, high: int, fallback: int) -> int:
        try:
            number = int(value)
        except (TypeError, ValueError):
            return fallback
        return max(low, min(number, high))

    def _get(self, path: str, params: dict[str, Any], operation: str, timeout: float | None):
        """One GET with the shared headers, budget and error mapping."""
        try:
            return requests.get(
                f"{self.base_url}{path}",
                headers=self._headers(),
                params=params,
                timeout=(CONNECT_TIMEOUT, float(timeout) if timeout else READ_TIMEOUT),
            )
        except Exception as exc:
            raise RecallError(
                f"Recall {operation} transport failure: {type(exc).__name__}"
            ) from exc

    # -- read --------------------------------------------------------------

    def search(
        self,
        query: str,
        *,
        limit: int = 5,
        rerank: bool = True,
        graph_boost: bool = False,
        tags: list[str] | None = None,
        timeout: float | None = None,
    ) -> list[dict[str, Any]]:
        """Progressive-retrieval search. Returns the raw ``items`` list.

        Returns ``[]`` without any HTTP call when unconfigured or the query is
        blank. Raises ``RecallAuthError`` / ``RecallError`` on failure — no
        retry: this call sits on the turn path with a 3 s budget by default.
        Callers that are NOT on the turn path pass ``SLOW_READ_TIMEOUT``.
        """
        if not self.api_key:
            return []
        text = (query or "").strip()
        if not text:
            return []

        params: dict[str, Any] = {
            "query": text[:MAX_QUERY_CHARS],
            "limit": max(MIN_LIMIT, min(int(limit), MAX_LIMIT)),
            "rerank": "true" if rerank else "false",
            "graph_boost": "true" if graph_boost else "false",
            "scope": "all",
        }
        if tags:
            params["tags"] = ",".join(str(t) for t in tags)

        try:
            response = requests.get(
                f"{self.base_url}{SEARCH_PATH}",
                headers=self._headers(),
                params=params,
                timeout=(CONNECT_TIMEOUT, float(timeout) if timeout else READ_TIMEOUT),
            )
        except Exception as exc:
            raise RecallError(f"Recall search transport failure: {type(exc).__name__}") from exc

        self._check(response, "search")
        payload = self._json(response, "search")
        items = payload.get("items") or []
        if not isinstance(items, list):
            return []
        return [item for item in items if isinstance(item, dict)]

    # -- read: the opt-in extras -------------------------------------------
    #
    # All three are off the turn path (they only ever run behind an explicitly
    # enabled tool), read-only, and — like ``search`` — never retried: a read
    # that failed is a read the model can simply ask for again.

    def graph_recall(
        self,
        query: str,
        *,
        depth: int = 2,
        limit: int = 10,
        timeout: float | None = None,
    ) -> dict[str, Any] | list[Any]:
        """Entity-centric recall with relations. Returns the raw payload.

        Returns ``{}`` without any HTTP call when unconfigured or the query is
        blank. Raises ``RecallAuthError`` / ``RecallError`` on failure.
        """
        if not self.api_key:
            return {}
        text = (query or "").strip()
        if not text:
            return {}

        params: dict[str, Any] = {
            "q": text[:MAX_QUERY_CHARS],
            "depth": self._clamp(depth, MIN_DEPTH, MAX_DEPTH, 2),
            "include_relations": "true",
            "limit": self._clamp(limit, MIN_LIMIT, MAX_GRAPH_LIMIT, 10),
        }
        response = self._get(GRAPH_RECALL_PATH, params, "graph recall", timeout)
        self._check(response, "graph recall")
        return self._json_any(response, "graph recall")

    def who_knows(
        self, topic: str, *, n_results: int = 5, timeout: float | None = None
    ) -> dict[str, Any]:
        """The people/entities most connected to ``topic``.

        Returns ``{}`` without any HTTP call when unconfigured or the topic is
        blank. Raises ``RecallAuthError`` / ``RecallError`` on failure.
        """
        if not self.api_key:
            return {}
        text = (topic or "").strip()
        if not text:
            return {}

        params: dict[str, Any] = {
            "topic": text[:MAX_QUERY_CHARS],
            "n_results": self._clamp(n_results, MIN_LIMIT, MAX_WHO_KNOWS_RESULTS, 5),
        }
        response = self._get(WHO_KNOWS_PATH, params, "who knows", timeout)
        self._check(response, "who knows")
        return self._json(response, "who knows")

    def stats(self, timeout: float | None = None) -> dict[str, Any]:
        """Memory-store statistics: the ``/search/stats`` payload, plus the
        graph numbers under ``"graph"`` when that second call succeeds.

        The graph is a separate service (Neo4j) and is allowed to be down:
        stats must degrade to the memory numbers alone, never fail because of
        it. Returns ``{}`` without any HTTP call when unconfigured. Raises
        ``RecallAuthError`` / ``RecallError`` only if the memory stats fail.
        """
        if not self.api_key:
            return {}

        response = self._get(SEARCH_STATS_PATH, {"scope": "all"}, "stats", timeout)
        self._check(response, "stats")
        payload = dict(self._json(response, "stats"))

        try:
            graph_response = self._get(GRAPH_STATS_PATH, {}, "graph stats", timeout)
            self._check(graph_response, "graph stats")
            payload["graph"] = self._json(graph_response, "graph stats")
        except RecallError:
            pass  # documented degradation: memory numbers without the graph
        return payload

    # -- write -------------------------------------------------------------

    def store(
        self,
        content: str,
        *,
        memory_type: str = "context",
        scope: str = "project",
        tags: list[str] | None = None,
    ) -> str:
        """Persist one memory. Returns the created ``memory_id`` (may be "").

        Always off the turn path, so it retries **once** on a transport error
        or a 5xx. Auth failures and 4xx are terminal — retrying them only
        doubles the load. Raises ``RecallAuthError`` / ``RecallError``.
        """
        if memory_type not in MEMORY_TYPES:
            raise ValueError(f"memory_type must be one of {sorted(MEMORY_TYPES)}")
        if scope not in SCOPES:
            raise ValueError(f"scope must be one of {sorted(SCOPES)}")
        if not self.api_key:
            return ""
        text = (content or "").strip()
        if not text:
            return ""

        body = {
            "content": text,
            "memory_type": memory_type,
            "scope": scope,
            "tags": list(tags or []),
        }
        url = f"{self.base_url}{STORE_PATH}"

        last_error: RecallError | None = None
        for attempt in (0, 1):
            try:
                response = requests.post(
                    url,
                    headers=self._headers(),
                    json=body,
                    timeout=(CONNECT_TIMEOUT, WRITE_TIMEOUT),
                )
            except Exception as exc:
                last_error = RecallError(
                    f"Recall store transport failure: {type(exc).__name__}"
                )
                if attempt == 0:
                    continue
                raise last_error from exc

            status = getattr(response, "status_code", 0)
            if status in (401, 403):
                raise RecallAuthError(f"Recall store rejected the API key (HTTP {status})")
            if 500 <= status < 600:
                last_error = RecallError(f"Recall store returned HTTP {status}")
                if attempt == 0:
                    continue
                raise last_error
            self._check(response, "store")

            payload = self._json(response, "store")
            memory_id = payload.get("memory_id") or payload.get("id") or ""
            return str(memory_id) if memory_id else ""

        # No trailing raise: the loop cannot fall through. Attempt 1 either
        # returns, or raises — `continue` is only ever taken on attempt 0.
