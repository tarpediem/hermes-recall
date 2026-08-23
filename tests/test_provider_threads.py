"""Thread-lifecycle hardening: spawning never blocks a turn, and background
targets can never surface an unhandled traceback.

Everything here is driven by ``threading.Event`` — no sleeps, no wall-clock
guesses beyond the "did this call return promptly" assertions.
"""

from __future__ import annotations

import threading
import time

from recall._client import RecallClient
from recall._provider import MAX_LIVE_THREADS, RecallMemoryProvider

USER = "Why did marker extraction move back to the GPU for the ingestion pipeline?"
ASSISTANT = "Because page-by-page chunking keeps VRAM under the 8 GB cap per page."

# A generous ceiling: the point is "did not join a blocked thread", and the
# unhardened code blocks for the full 5 s join timeout.
FAST = 1.0


class GatedClient(RecallClient):
    """A store that blocks until the test releases it."""

    def __init__(self) -> None:
        super().__init__(api_key="rag_k", base_url="https://recall.example")
        self.gate = threading.Event()
        self.entered = threading.Semaphore(0)
        self.stored: list[str] = []
        self._lock = threading.Lock()

    def store(self, content, *, memory_type="context", scope="project", tags=None):
        self.entered.release()
        self.gate.wait(timeout=10.0)
        with self._lock:
            self.stored.append(content)
        return "mem-1"

    def search(self, query, *, limit=5, rerank=True, graph_boost=False, tags=None):
        self.entered.release()
        self.gate.wait(timeout=10.0)
        return [
            {
                "id": "m",
                "type": "context",
                "timestamp": "2026-01-01T00:00:00Z",
                "snippet": "a snippet from the old session",
            }
        ]


def _provider(client, tmp_path, session_id="s1"):
    provider = RecallMemoryProvider(client)
    provider.initialize(
        session_id, hermes_home=str(tmp_path), platform="cli", agent_identity="charlotte"
    )
    return provider


def test_second_sync_turn_does_not_block_on_the_first(tmp_path):
    """Ruling 1: spawning is O(1) — no join on the turn path."""
    client = GatedClient()
    provider = _provider(client, tmp_path)

    provider.sync_turn(USER, ASSISTANT)
    assert client.entered.acquire(timeout=5.0), "the first store must have started"

    started = time.monotonic()
    provider.sync_turn(USER, ASSISTANT + " Confirmed on the second turn.")
    elapsed = time.monotonic() - started
    assert elapsed < FAST, f"sync_turn blocked for {elapsed:.2f}s on the in-flight store"

    client.gate.set()
    provider.shutdown()
    assert len(client.stored) == 2


def test_queue_prefetch_does_not_block_on_an_in_flight_store(tmp_path):
    """The prefetch path shares the thread registry; it must stay O(1) too."""
    client = GatedClient()
    provider = _provider(client, tmp_path)

    provider.sync_turn(USER, ASSISTANT)
    assert client.entered.acquire(timeout=5.0)

    started = time.monotonic()
    provider.queue_prefetch("a genuine question about the ingestion pipeline")
    elapsed = time.monotonic() - started
    assert elapsed < FAST, f"queue_prefetch blocked for {elapsed:.2f}s"

    client.gate.set()
    provider.shutdown()


def test_initialize_does_not_join_and_leaves_no_stale_join_for_the_turn_path(tmp_path):
    """Ruling 2: re-init mid-flight is fast, and never defers a join to a turn."""
    client = GatedClient()
    provider = _provider(client, tmp_path)

    provider.sync_turn(USER, ASSISTANT)
    assert client.entered.acquire(timeout=5.0), "the first store must have started"

    started = time.monotonic()
    provider.initialize("s2", hermes_home=str(tmp_path), platform="cli")
    init_elapsed = time.monotonic() - started
    assert init_elapsed < FAST, f"initialize blocked for {init_elapsed:.2f}s"

    started = time.monotonic()
    provider.sync_turn(USER, ASSISTANT + " Now in the new session.")
    turn_elapsed = time.monotonic() - started
    assert turn_elapsed < FAST, (
        f"the first turn of the new session blocked for {turn_elapsed:.2f}s "
        "on the previous session's thread"
    )

    # The old session's daemon thread is left to finish: it is writing a valid
    # memory and nothing is gained by discarding it.
    client.gate.set()
    provider.shutdown()
    assert len(client.stored) == 2


def test_a_raising_background_target_is_logged_not_traced(tmp_path, caplog, capsys):
    """Ruling 3: the whole target runs inside the failure handler."""
    provider = _provider(GatedClient(), tmp_path)
    done = threading.Event()

    def boom() -> None:
        try:
            raise RuntimeError("target exploded")
        finally:
            done.set()

    with caplog.at_level("WARNING", logger="recall._provider"):
        provider._spawn("boom", boom)
        assert done.wait(timeout=5.0)
        provider.shutdown()

    assert "target exploded" in caplog.text
    assert "background" in caplog.text
    captured = capsys.readouterr()
    assert "Traceback" not in captured.err
    assert "Traceback" not in captured.out


def test_live_threads_are_bounded_under_a_stuck_recall(tmp_path):
    """A wedged Recall cannot grow the thread pool without bound."""
    client = GatedClient()
    provider = _provider(client, tmp_path)

    for index in range(MAX_LIVE_THREADS + 5):
        provider.sync_turn(USER, f"{ASSISTANT} Turn number {index}.")

    assert len(provider._threads) <= MAX_LIVE_THREADS

    client.gate.set()
    provider.shutdown()


def test_spawn_prunes_the_finished_thread_of_a_previous_turn(tmp_path):
    """_spawn — not shutdown — is what keeps the registry from growing."""
    client = GatedClient()
    client.gate.set()  # never blocks
    provider = _provider(client, tmp_path)

    provider.sync_turn(USER, ASSISTANT)
    assert len(provider._threads) == 1
    provider._threads[0].join(timeout=5.0)
    assert not provider._threads[0].is_alive()

    provider.sync_turn(USER, ASSISTANT + " A second turn.")

    assert len(provider._threads) == 1, "the finished thread was not pruned"
    provider.shutdown()
    assert len(client.stored) == 2


def test_spawned_threads_are_daemons(tmp_path):
    client = GatedClient()
    provider = _provider(client, tmp_path)

    provider.sync_turn(USER, ASSISTANT)
    assert provider._threads
    for thread in provider._threads:
        assert thread.daemon is True

    client.gate.set()
    provider.shutdown()
    assert threading.active_count() >= 1


def test_an_in_flight_prefetch_cannot_resurrect_a_flushed_cache_entry(tmp_path):
    """Ruling 2's leak: re-init clears the cache, the old search must not refill it."""
    client = GatedClient()
    provider = _provider(client, tmp_path, session_id="old")

    provider.queue_prefetch("a genuine question about the ingestion pipeline")
    assert client.entered.acquire(timeout=5.0), "the search must have started"

    provider.initialize("new", hermes_home=str(tmp_path), platform="cli")
    assert provider._prefetch_cache == {}

    client.gate.set()
    provider.shutdown()

    assert provider._prefetch_cache == {}, "a dead session's entry was written back"
