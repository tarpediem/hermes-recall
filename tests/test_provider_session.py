"""Session end synthesis, session switching, and bounded shutdown."""

from __future__ import annotations

import threading
import time

from recall._client import RecallClient, RecallError
from recall._provider import RecallMemoryProvider

MESSAGES = [
    {"role": "user", "content": "Should we move marker extraction back to the GPU?"},
    {"role": "assistant", "content": "We decided to move it back with page-by-page chunking."},
    {"role": "user", "content": "And what timeout should the extractor use from now on?"},
    {"role": "assistant", "content": "EXTRACT_TIMEOUT_SECONDS is now 1800 instead of 300."},
]


class RecordingClient(RecallClient):
    def __init__(self, raiser=None, delay=0.0):
        super().__init__(api_key="rag_k", base_url="https://recall.example")
        self.stored = []
        self.raiser = raiser
        self.delay = delay

    def store(self, content, *, memory_type="context", scope="project", tags=None):
        if self.delay:
            time.sleep(self.delay)
        self.stored.append(
            {"content": content, "memory_type": memory_type, "tags": list(tags or [])}
        )
        if self.raiser is not None:
            raise self.raiser
        return "mem-1"

    def search(self, query, *, limit=5, rerank=True, graph_boost=False, tags=None, timeout=None):
        return [
            {
                "id": "m",
                "type": "context",
                "timestamp": "2026-01-01T00:00:00Z",
                "snippet": "cached snippet",
            }
        ]


class GatedRecordingClient(RecordingClient):
    """A store that blocks until the test releases it — no sleeps anywhere."""

    def __init__(self):
        super().__init__()
        self.gate = threading.Event()
        self.entered = threading.Semaphore(0)

    def store(self, content, *, memory_type="context", scope="project", tags=None):
        self.entered.release()
        self.gate.wait(timeout=10.0)
        return super().store(
            content, memory_type=memory_type, scope=scope, tags=tags
        )


def _provider(client, tmp_path, **init):
    provider = RecallMemoryProvider(client)
    provider.initialize(
        "s1", hermes_home=str(tmp_path), platform="cli", agent_identity="charlotte", **init
    )
    return provider


def test_on_session_end_stores_one_synthesis(tmp_path):
    client = RecordingClient()
    provider = _provider(client, tmp_path)

    provider.on_session_end(MESSAGES)
    provider.shutdown()

    assert len(client.stored) == 1
    record = client.stored[0]
    assert "Session summary" in record["content"]
    assert record["memory_type"] == "decision"
    assert "session-summary" in record["tags"]


def test_on_session_end_skips_below_two_useful_turns(tmp_path):
    client = RecordingClient()
    provider = _provider(client, tmp_path)

    provider.on_session_end(MESSAGES[:2])
    provider.shutdown()

    assert client.stored == []


def test_on_session_end_skips_when_session_summary_disabled(tmp_path, write_recall_config):
    write_recall_config(tmp_path, {"session_summary": False})
    client = RecordingClient()
    provider = _provider(client, tmp_path)

    provider.on_session_end(MESSAGES)
    provider.shutdown()

    assert client.stored == []


def test_on_session_end_does_not_write_for_non_primary_context(tmp_path):
    client = RecordingClient()
    provider = _provider(client, tmp_path, agent_context="cron")

    provider.on_session_end(MESSAGES)
    provider.shutdown()

    assert client.stored == []


def test_on_session_end_does_not_raise_when_store_fails(tmp_path):
    provider = _provider(RecordingClient(raiser=RecallError("down")), tmp_path)

    provider.on_session_end(MESSAGES)
    provider.shutdown()


def test_on_session_end_does_not_raise_on_malformed_messages(tmp_path):
    provider = _provider(RecordingClient(), tmp_path)

    provider.on_session_end([None, 42, {"role": "user"}])  # type: ignore[list-item]
    provider.shutdown()


def test_on_session_switch_rescopes_the_session_id(tmp_path):
    client = RecordingClient()
    provider = _provider(client, tmp_path)

    provider.on_session_switch("s2", parent_session_id="s1")

    assert provider._session_id == "s2"


def test_on_session_switch_with_reset_clears_the_prefetch_cache(tmp_path):
    client = RecordingClient()
    provider = _provider(client, tmp_path)
    provider.queue_prefetch("a genuine question about something", session_id="s1")
    provider.shutdown()
    assert provider._prefetch_cache

    provider.on_session_switch("s2", reset=True)

    assert provider._prefetch_cache == {}
    assert provider._prefetch_counts == {}


def test_on_session_switch_with_reset_rearms_the_auth_warning(tmp_path):
    provider = _provider(RecordingClient(), tmp_path)
    provider._auth_warned = True

    provider.on_session_switch("s2", reset=True)

    assert provider._auth_warned is False


def test_on_session_switch_with_reset_joins_the_in_flight_write(tmp_path):
    """reset is a session boundary, so joining there is allowed — and expected.

    The store is still blocked when the hook is entered (the releaser waits for
    ``switching``), so the hook can only return with the write landed by having
    joined it. Deleting the ``_join_threads`` call also leaves the finished
    thread in ``_threads``: nothing else prunes the registry here.
    """
    client = GatedRecordingClient()
    provider = _provider(client, tmp_path)
    provider.on_session_end(MESSAGES)
    assert client.entered.acquire(timeout=5.0), "the store must have started"

    switching = threading.Event()
    releaser = threading.Thread(
        target=lambda: (switching.wait(timeout=5.0), client.gate.set()), daemon=True
    )
    releaser.start()

    switching.set()
    provider.on_session_switch("s2", reset=True)

    assert provider._threads == []
    assert len(client.stored) == 1, "reset must not strand the session's write"
    releaser.join(timeout=5.0)


def test_on_session_switch_rewound_invalidates_the_current_cache(tmp_path):
    provider = _provider(RecordingClient(), tmp_path)
    provider.queue_prefetch("a genuine question about something", session_id="s1")
    provider.shutdown()
    assert "s1" in provider._prefetch_cache

    provider.on_session_switch("s1", rewound=True)

    assert "s1" not in provider._prefetch_cache


def test_on_session_switch_without_reset_keeps_other_cached_sessions(tmp_path):
    provider = _provider(RecordingClient(), tmp_path)
    provider.queue_prefetch("a genuine question about something", session_id="s9")
    provider.shutdown()

    provider.on_session_switch("s2", parent_session_id="s1")

    assert "s9" in provider._prefetch_cache


def test_on_session_switch_without_reset_does_not_join(tmp_path):
    """A plain switch is on the hot path of a subagent handoff: stay O(1).

    The store is held on an Event for the duration of the hook, so a join would
    cost the full 10 s gate wait rather than a scheduling hiccup.
    """
    client = GatedRecordingClient()
    provider = _provider(client, tmp_path)
    provider.on_session_end(MESSAGES)
    assert client.entered.acquire(timeout=5.0), "the store must have started"

    started = time.monotonic()
    provider.on_session_switch("s2", parent_session_id="s1")
    elapsed = time.monotonic() - started

    assert elapsed < 1.0, f"a plain switch blocked for {elapsed:.2f}s"
    assert client.stored == [], "the write is still in flight, not joined"

    client.gate.set()
    provider.shutdown()
    assert len(client.stored) == 1


def test_on_session_switch_does_not_raise_on_bad_input(tmp_path):
    provider = _provider(RecordingClient(), tmp_path)
    provider.on_session_switch("", parent_session_id="", reset=False, rewound=False, extra=1)


def test_shutdown_joins_threads_and_returns_within_the_budget(tmp_path):
    client = RecordingClient(delay=0.2)
    provider = _provider(client, tmp_path)

    provider.on_session_end(MESSAGES)
    started = time.monotonic()
    provider.shutdown()
    elapsed = time.monotonic() - started

    assert client.stored, "the summary thread must have completed"
    assert elapsed < 5.0


def test_shutdown_is_safe_to_call_twice(tmp_path):
    provider = _provider(RecordingClient(), tmp_path)
    provider.shutdown()
    provider.shutdown()


def test_shutdown_leaves_no_non_daemon_threads(tmp_path):
    provider = _provider(RecordingClient(), tmp_path)
    provider.on_session_end(MESSAGES)

    for thread in provider._threads:
        assert thread.daemon is True

    provider.shutdown()
    assert threading.active_count() >= 1
