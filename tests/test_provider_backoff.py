"""Read backoff: stop paying the connect budget every turn to a dead host.

Deterministic — the clock is a fake, nothing here sleeps.
"""

from recall import _provider as provider_module
from recall._client import RecallAuthError, RecallClient, RecallError
from recall._provider import (
    READ_BACKOFF_SECONDS,
    READ_BACKOFF_THRESHOLD,
    RecallMemoryProvider,
)

QUERY = "why did the graph indexer stop extracting entities last week"


class FakeClock:
    """Stands in for the ``time`` module inside ``recall._provider``."""

    def __init__(self, now=1000.0):
        self.now = now

    def monotonic(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class CountingClient(RecallClient):
    def __init__(self, raiser=None, items=None):
        super().__init__(api_key="rag_k", base_url="https://recall.example")
        self.raiser = raiser
        self.items = items if items is not None else []
        self.searches = 0

    def search(self, query, *, limit=5, rerank=True, graph_boost=False, tags=None, timeout=None):
        self.searches += 1
        if self.raiser is not None:
            raise self.raiser
        return self.items

    def store(self, content, *, memory_type="context", scope="project", tags=None):
        return "mem-1"


def _provider(client, tmp_path, clock, monkeypatch):
    monkeypatch.setattr(provider_module, "time", clock)
    p = RecallMemoryProvider(client)
    p.initialize("s1", hermes_home=str(tmp_path), platform="cli")
    return p


HIT = [
    {
        "id": "m1",
        "type": "bugfix",
        "timestamp": "2026-07-10T02:17:00Z",
        "snippet": "An empty ids list was falsy, so delete() dropped the collection",
    }
]


def test_the_cold_path_keeps_trying_below_the_threshold(tmp_path, monkeypatch):
    client = CountingClient(raiser=RecallError("connection refused"))
    p = _provider(client, tmp_path, FakeClock(), monkeypatch)

    for _ in range(READ_BACKOFF_THRESHOLD):
        p.prefetch(QUERY, session_id="s1")

    assert client.searches == READ_BACKOFF_THRESHOLD


def test_the_cold_path_stops_searching_after_three_transport_failures(tmp_path, monkeypatch):
    client = CountingClient(raiser=RecallError("connection refused"))
    p = _provider(client, tmp_path, FakeClock(), monkeypatch)

    for _ in range(READ_BACKOFF_THRESHOLD):
        p.prefetch(QUERY, session_id="s1")
    at_threshold = client.searches

    block = p.prefetch(QUERY, session_id="s1")

    assert client.searches == at_threshold, "the fourth turn must not reach the transport"
    assert block == ""
    assert p.recall_status() is None


def test_the_backoff_expires_and_the_cold_path_tries_again(tmp_path, monkeypatch):
    clock = FakeClock()
    client = CountingClient(raiser=RecallError("connection refused"))
    p = _provider(client, tmp_path, clock, monkeypatch)

    for _ in range(READ_BACKOFF_THRESHOLD + 1):
        p.prefetch(QUERY, session_id="s1")
    paused_at = client.searches

    clock.advance(READ_BACKOFF_SECONDS + 1)
    p.prefetch(QUERY, session_id="s1")

    assert client.searches == paused_at + 1


def test_a_successful_search_resets_the_counter(tmp_path, monkeypatch):
    client = CountingClient(raiser=RecallError("boom"))
    p = _provider(client, tmp_path, FakeClock(), monkeypatch)

    for _ in range(READ_BACKOFF_THRESHOLD - 1):
        p.prefetch(QUERY, session_id="s1")
    client.raiser, client.items = None, HIT
    p.prefetch(QUERY, session_id="s1")
    client.raiser = RecallError("boom")

    # Two more failures. Four have now failed in all, but only two in a row —
    # without the reset this would already be paused.
    for _ in range(READ_BACKOFF_THRESHOLD - 1):
        p.prefetch(QUERY, session_id="s1")

    assert p._read_backoff_until == 0.0
    before = client.searches
    p.prefetch(QUERY, session_id="s1")
    assert client.searches == before + 1


def test_a_successful_search_clears_an_active_backoff(tmp_path, monkeypatch):
    """The warm-ups keep trying while the cold path is paused — the first one
    that lands must let the turn path back in immediately."""
    client = CountingClient(raiser=RecallError("boom"))
    p = _provider(client, tmp_path, FakeClock(), monkeypatch)

    for _ in range(READ_BACKOFF_THRESHOLD):
        p.prefetch(QUERY, session_id="s2")
    assert p._read_backoff_until > 0.0

    client.raiser, client.items = None, HIT
    # An off-turn warm-up, run inline: the same code path a thread would take.
    p._search_and_cache(QUERY, "s2", None, None)

    assert p._read_backoff_until == 0.0
    before = client.searches
    p.prefetch("a completely different question about the extractor", session_id="s3")
    assert client.searches == before + 1


def test_an_auth_failure_does_not_arm_the_backoff(tmp_path, monkeypatch):
    """A rejected key is a permanent, already-reported condition — not a sign
    the host is unreachable, and the fix is not to wait 60 s."""
    client = CountingClient(raiser=RecallAuthError("rejected"))
    p = _provider(client, tmp_path, FakeClock(), monkeypatch)

    for _ in range(READ_BACKOFF_THRESHOLD + 2):
        p.prefetch(QUERY, session_id="s1")

    assert client.searches == READ_BACKOFF_THRESHOLD + 2
    assert p._read_backoff_until == 0.0


def test_the_backoff_is_logged_once_not_once_per_turn(tmp_path, monkeypatch, caplog):
    client = CountingClient(raiser=RecallError("connection refused"))
    p = _provider(client, tmp_path, FakeClock(), monkeypatch)

    with caplog.at_level("WARNING"):
        for _ in range(READ_BACKOFF_THRESHOLD + 5):
            p.prefetch(QUERY, session_id="s1")

    entries = [r for r in caplog.records if "pausing turn-path" in r.getMessage()]
    assert len(entries) == 1
    assert entries[0].levelname == "WARNING"


def test_warm_ups_still_run_while_the_cold_path_is_paused(tmp_path, monkeypatch):
    """Documented: a warm-up is off the turn path, so its latency costs the
    user nothing and it is what discovers that Recall is back."""
    client = CountingClient(raiser=RecallError("connection refused"))
    p = _provider(client, tmp_path, FakeClock(), monkeypatch)

    for _ in range(READ_BACKOFF_THRESHOLD + 1):
        p.prefetch(QUERY, session_id="s1")
    before = client.searches

    p.queue_prefetch(QUERY, session_id="s1")
    p.shutdown()

    assert client.searches == before + 1


def test_a_new_session_starts_with_a_clean_slate(tmp_path, monkeypatch):
    client = CountingClient(raiser=RecallError("connection refused"))
    p = _provider(client, tmp_path, FakeClock(), monkeypatch)

    for _ in range(READ_BACKOFF_THRESHOLD + 1):
        p.prefetch(QUERY, session_id="s1")
    before = client.searches

    p.initialize("s2", hermes_home=str(tmp_path))
    p.prefetch(QUERY, session_id="s2")

    assert client.searches == before + 1
