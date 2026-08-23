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

MAX_QUERY_CHARS = 2000
MIN_LIMIT = 1
MAX_LIMIT = 100

SEARCH_PATH = "/api/v1/memory/search"
STORE_PATH = "/api/v1/memories"

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
                timeout=float(timeout) if timeout else READ_TIMEOUT,
            )
        except RecallError:
            raise
        except Exception as exc:
            raise RecallError(f"Recall search transport failure: {type(exc).__name__}") from exc

        self._check(response, "search")
        payload = self._json(response, "search")
        items = payload.get("items") or []
        if not isinstance(items, list):
            return []
        return [item for item in items if isinstance(item, dict)]

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
                    url, headers=self._headers(), json=body, timeout=WRITE_TIMEOUT
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

        raise last_error or RecallError("Recall store failed")
