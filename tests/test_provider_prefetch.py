"""Prefetch cache, injected format, and the recall_status contract."""

import threading

from recall._client import READ_TIMEOUT, SLOW_READ_TIMEOUT, RecallClient, RecallError
from recall._provider import RecallMemoryProvider


class RecordingClient(RecallClient):
    """A RecallClient whose search() is scripted — no HTTP involved."""

    def __init__(self, results=None, raiser=None):
        super().__init__(api_key="rag_k", base_url="https://recall.example")
        self.results = results if results is not None else []
        self.raiser = raiser
        self.calls = []

    def search(self, query, *, limit=5, rerank=True, graph_boost=False, tags=None, timeout=None):
        self.calls.append(
            {
                "query": query,
                "limit": limit,
                "rerank": rerank,
                "graph_boost": graph_boost,
                "tags": tags,
                "timeout": timeout,
            }
        )
        if self.raiser is not None:
            raise self.raiser
        return self.results


class GatedClient(RecordingClient):
    """The FIRST search blocks until the test releases it; later ones run free.

    Lets a test hold a background prefetch mid-flight and drive the foreground
    path against it deterministically — Events only, never sleeps.
    """

    def __init__(self, results=None):
        super().__init__(results)
        self.entered = threading.Event()
        self.release = threading.Event()

    def search(self, query, **kwargs):
        first = not self.calls
        result = super().search(query, **kwargs)
        if first:
            self.entered.set()
            assert self.release.wait(timeout=5.0)
        return result


ITEMS = [
    {
        "id": "m1",
        "type": "decision",
        "timestamp": "2026-07-21T03:11:00Z",
        "snippet": "Marker extraction moved back to GPU with page-by-page chunking",
        "score": 0.91,
    },
    {
        "id": "m2",
        "type": "bugfix",
        "timestamp": "2026-07-10T02:17:00Z",
        "snippet": "pgvector delete() with an empty ids list wiped the collection",
        "score": 0.84,
    },
]


def _provider(client, tmp_path, **init):
    provider = RecallMemoryProvider(client)
    provider.initialize(init.pop("session_id", "s1"), hermes_home=str(tmp_path), **init)
    return provider


def test_prefetch_cold_cache_calls_search_and_returns_the_block(tmp_path):
    client = RecordingClient(ITEMS)
    provider = _provider(client, tmp_path)

    block = provider.prefetch("where did marker extraction go", session_id="s1")

    assert block.startswith("Relevant memories (Recall):")
    assert "[decision, 2026-07-21]" in block
    assert "[bugfix, 2026-07-10]" in block
    assert "Marker extraction moved back to GPU" in block
    assert len(client.calls) == 1


def test_the_cold_path_searches_unreranked_inside_the_turn_budget(tmp_path):
    """A cache miss must fit the 3 s turn budget, so it drops rerank.

    A reranked query costs ~4.5 s against a real instance: keeping rerank here
    would mean no memory at all on the first turn of every session.
    """
    client = RecordingClient(ITEMS)
    provider = _provider(client, tmp_path)

    provider.prefetch("a real question about the graph indexer", session_id="s1")

    call = client.calls[0]
    assert call["limit"] == 5
    assert call["rerank"] is False
    assert call["timeout"] == READ_TIMEOUT
    assert call["graph_boost"] is False
    assert call["tags"] is None


def test_the_warm_up_keeps_the_configured_rerank_and_the_off_turn_budget(tmp_path):
    client = RecordingClient(ITEMS)
    provider = _provider(client, tmp_path)

    provider.queue_prefetch("a real question about the graph indexer", session_id="s1")
    provider.shutdown()

    call = client.calls[0]
    assert call["limit"] == 5
    assert call["rerank"] is True  # the configured value, not the cold-path override
    assert call["timeout"] == SLOW_READ_TIMEOUT


def test_shutdown_makes_an_abandoned_warm_up_discard_its_result(tmp_path):
    """A warm-up that outlives the 5 s join must not write back afterwards."""
    client = RecordingClient(ITEMS)
    provider = _provider(client, tmp_path)
    generation = provider._cache_generation

    provider.shutdown()
    block = provider._search_and_cache("a real question", "s1", generation)

    assert block  # still returned to a synchronous caller
    assert provider._prefetch_cache == {}  # but never resurrected in the cache


def test_prefetch_truncates_each_snippet_to_300_chars(tmp_path):
    long_item = [
        {
            "id": "m",
            "type": "context",
            "timestamp": "2026-01-01T00:00:00Z",
            "snippet": "x" * 900,
        }
    ]
    provider = _provider(RecordingClient(long_item), tmp_path)

    block = provider.prefetch("a genuine question about something", session_id="s1")

    body = block.splitlines()[1]
    assert len(body) <= len("- [context, 2026-01-01] ") + 300


def test_prefetch_returns_empty_for_a_trivial_prompt_without_calling(tmp_path):
    client = RecordingClient(ITEMS)
    provider = _provider(client, tmp_path)

    assert provider.prefetch("thanks!", session_id="s1") == ""
    assert client.calls == []


def test_prefetch_returns_empty_when_there_are_no_items(tmp_path):
    provider = _provider(RecordingClient([]), tmp_path)

    assert provider.prefetch("a genuine question about something", session_id="s1") == ""


def test_queue_prefetch_warms_the_cache_so_prefetch_makes_no_call(tmp_path):
    client = RecordingClient(ITEMS)
    provider = _provider(client, tmp_path)

    provider.queue_prefetch("a genuine question about something", session_id="s1")
    provider.shutdown()
    calls_after_warm = len(client.calls)

    block = provider.prefetch("a genuine question about something", session_id="s1")

    assert block.startswith("Relevant memories (Recall):")
    assert len(client.calls) == calls_after_warm == 1


def test_queue_prefetch_is_a_noop_for_a_trivial_prompt(tmp_path):
    client = RecordingClient(ITEMS)
    provider = _provider(client, tmp_path)

    provider.queue_prefetch("ok", session_id="s1")
    provider.shutdown()

    assert client.calls == []


def test_cache_is_scoped_per_session(tmp_path):
    client = RecordingClient(ITEMS)
    provider = _provider(client, tmp_path)

    provider.queue_prefetch("a genuine question about something", session_id="s1")
    provider.shutdown()
    provider.prefetch("a genuine question about something", session_id="s2")

    assert len(client.calls) == 2


def test_prefetch_consumes_the_cache_once(tmp_path):
    client = RecordingClient(ITEMS)
    provider = _provider(client, tmp_path)

    provider.queue_prefetch("a genuine question about something", session_id="s1")
    provider.shutdown()
    provider.prefetch("a genuine question about something", session_id="s1")
    provider.prefetch("a genuine question about something", session_id="s1")

    assert len(client.calls) == 2


def test_prefetch_returns_empty_when_search_raises(tmp_path):
    provider = _provider(RecordingClient(raiser=RecallError("timeout")), tmp_path)

    assert provider.prefetch("a genuine question about something", session_id="s1") == ""


def test_prefetch_returns_empty_when_the_client_raises_anything(tmp_path):
    provider = _provider(RecordingClient(raiser=RuntimeError("kaboom")), tmp_path)

    assert provider.prefetch("a genuine question about something", session_id="s1") == ""


def test_recall_status_reports_the_last_prefetch(tmp_path):
    provider = _provider(RecordingClient(ITEMS), tmp_path)

    provider.prefetch("a genuine question about something", session_id="s1")
    status = provider.recall_status()

    assert status is not None
    assert status.provider_label == "Recall"
    assert status.count == 2
    assert status.glyph == "🧠"


def test_recall_status_is_none_before_any_prefetch(tmp_path):
    assert _provider(RecordingClient(ITEMS), tmp_path).recall_status() is None


def test_recall_status_never_returns_a_stale_count(tmp_path):
    client = RecordingClient(ITEMS)
    provider = _provider(client, tmp_path)

    provider.prefetch("a genuine question about something", session_id="s1")
    assert provider.recall_status().count == 2

    client.results = []
    provider.prefetch("another genuine question about something", session_id="s1")

    assert provider.recall_status() is None


def test_recall_status_is_none_after_a_trivial_prompt(tmp_path):
    provider = _provider(RecordingClient(ITEMS), tmp_path)

    provider.prefetch("a genuine question about something", session_id="s1")
    provider.prefetch("thanks", session_id="s1")

    assert provider.recall_status() is None


def test_recall_status_is_none_after_a_failed_prefetch(tmp_path):
    client = RecordingClient(ITEMS)
    provider = _provider(client, tmp_path)

    provider.prefetch("a genuine question about something", session_id="s1")
    client.raiser = RecallError("down")
    provider.prefetch("another genuine question about something", session_id="s1")

    assert provider.recall_status() is None


# -- publication atomicity (regression: block and count were two dict writes) --


def test_a_warm_cache_entry_is_one_atomic_block_and_count_pair(tmp_path):
    client = RecordingClient(ITEMS)
    provider = _provider(client, tmp_path)

    provider.queue_prefetch("a genuine question about something", session_id="s1")
    provider.shutdown()

    entry = provider._prefetch_cache["s1"]
    block, count = entry
    assert block.startswith("Relevant memories (Recall):")
    assert count == len(ITEMS) == len(block.splitlines()) - 1


def test_a_warm_hit_reports_its_count_without_a_second_lookup(tmp_path):
    client = RecordingClient(ITEMS)
    provider = _provider(client, tmp_path)

    provider.queue_prefetch("a genuine question about something", session_id="s1")
    provider.shutdown()
    # Anything published outside the cache entry is not part of the contract.
    provider._prefetch_counts.clear()

    block = provider.prefetch("a genuine question about something", session_id="s1")

    assert block.startswith("Relevant memories (Recall):")
    assert provider.recall_status().count == 2


def test_a_foreground_prefetch_during_a_background_search_stays_consistent(tmp_path):
    client = GatedClient(ITEMS)
    provider = _provider(client, tmp_path)

    provider.queue_prefetch("a genuine question about something", session_id="s1")
    assert client.entered.wait(timeout=5.0)  # the background search is in flight

    block = provider.prefetch("a genuine question about something", session_id="s1")
    status = provider.recall_status()

    assert block.startswith("Relevant memories (Recall):")
    assert status is not None and status.count == 2

    client.release.set()
    provider.shutdown()

    published_block, published_count = provider._prefetch_cache["s1"]
    assert published_count == len(published_block.splitlines()) - 1 == 2
