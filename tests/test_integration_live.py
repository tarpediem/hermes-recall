"""Live round-trip against a real Recall instance.

Skipped unless ``RECALL_TEST_API_KEY`` is set; ``RECALL_TEST_BASE_URL``
optionally points it at a preprod instance instead of the public one. This is
the ONLY module in the suite that touches the network.

The memories it writes are tagged ``hermes-recall-live-test`` plus a
``marker:<uuid>`` tag. Nothing is ever deleted — the plugin has no delete path
— so an operator purges them with that tag when they get in the way.
"""

from __future__ import annotations

import json
import os
import time
import uuid

import pytest
from recall._client import (
    DEFAULT_BASE_URL,
    SLOW_READ_TIMEOUT,
    RecallAuthError,
    RecallClient,
)
from recall._provider import RecallMemoryProvider

API_KEY = os.environ.get("RECALL_TEST_API_KEY", "")
BASE_URL = os.environ.get("RECALL_TEST_BASE_URL", "") or DEFAULT_BASE_URL

PURGE_TAG = "hermes-recall-live-test"

# The server embeds asynchronously: a freshly stored memory is not instantly
# retrievable. Bounded so a broken instance fails the test instead of hanging.
SEARCH_ATTEMPTS = 8
SEARCH_INTERVAL_SECONDS = 2.0
# The provider block may arrive from the reranked warm-up or from the cold
# unreranked fallback; give both a bounded chance rather than one 5 s join.
BLOCK_POLL_SECONDS = 20.0

pytestmark = pytest.mark.skipif(
    not API_KEY, reason="set RECALL_TEST_API_KEY to run the live integration test"
)


def _client() -> RecallClient:
    return RecallClient(api_key=API_KEY, base_url=BASE_URL)


@pytest.fixture(scope="module")
def stored_memory():
    """Store ONE memory for the whole module and wait until it is retrievable.

    Module-scoped on purpose: every test below reads the same memory, so the
    live suite costs one write and a handful of reads.
    """
    # The server returns a ~100-char layer-1 snippet, so the marker goes FIRST:
    # anything further in loses its tail to truncation and never matches.
    marker = uuid.uuid4().hex[:8]
    content = (
        f"{marker} hermes-recall live integration test: the hermes-recall plugin "
        "stored this memory from its automated test suite to prove the "
        f"store/search round-trip works. Safe to delete (tag {PURGE_TAG})."
    )
    client = _client()

    memory_id = client.store(
        content,
        memory_type="context",
        scope="project",
        tags=["hermes", PURGE_TAG, f"marker:{marker}"],
    )
    assert isinstance(memory_id, str)

    query = f"{marker} hermes-recall live integration test"
    for attempt in range(SEARCH_ATTEMPTS):
        items = client.search(query, limit=10, rerank=True, timeout=SLOW_READ_TIMEOUT)
        if any(marker in str(item) for item in items):
            return {
                "marker": marker,
                "content": content,
                "query": query,
                "memory_id": memory_id,
                "attempts": attempt + 1,
            }
        time.sleep(SEARCH_INTERVAL_SECONDS)

    pytest.fail(
        f"stored memory {marker} was not retrievable after "
        f"~{SEARCH_ATTEMPTS * SEARCH_INTERVAL_SECONDS:.0f} s (id={memory_id or 'unknown'})"
    )


def test_store_then_search_finds_the_memory(stored_memory):
    """The fixture itself is the assertion: it fails if the marker never lands."""
    assert stored_memory["marker"] in stored_memory["content"]
    assert stored_memory["attempts"] >= 1


def test_search_returns_a_well_formed_item_list(stored_memory):
    # rerank=True costs a cross-encoder round-trip (~4.5 s measured), so an
    # off-turn caller must ask for the off-turn budget — exactly what the
    # provider's background warm-up and search tool do.
    items = _client().search(
        stored_memory["query"], limit=5, rerank=True, timeout=SLOW_READ_TIMEOUT
    )

    assert isinstance(items, list)
    assert items, "the marker query returned nothing"
    for item in items:
        assert isinstance(item, dict)
        assert "snippet" in item or "id" in item


def test_an_unreranked_search_fits_the_turn_path_budget(stored_memory):
    """The 3 s turn-path budget is only viable without rerank.

    This is the canary for READ_TIMEOUT: if a plain hybrid search stops
    fitting in it, the synchronous injection path is dead in production.
    """
    items = _client().search(stored_memory["query"], limit=3, rerank=False)

    assert isinstance(items, list)


def test_the_provider_injects_a_block_containing_the_stored_memory(
    stored_memory, tmp_path, write_recall_config
):
    """Full provider path with a real client: warm, drain, inject, report."""
    write_recall_config(tmp_path, {"base_url": BASE_URL, "limit": 5})
    provider = RecallMemoryProvider(_client())
    provider.initialize(
        "live-test",
        hermes_home=str(tmp_path),
        platform="cli",
        agent_identity="hermes-recall-tests",
        agent_context="primary",
    )

    try:
        provider.queue_prefetch(stored_memory["query"], session_id="live-test")

        # Do NOT bet on one join: a reranked warm-up runs 4.3-4.6 s against a
        # 5 s shutdown budget. Poll instead, and assert the CONTENT rather than
        # which path produced it — a warm hit and a cold unreranked hit are both
        # correct outcomes here, and both must contain the marker.
        block = ""
        deadline = time.monotonic() + BLOCK_POLL_SECONDS
        while True:
            provider.shutdown()  # drains whatever the warm-up left in flight
            block = provider.prefetch(stored_memory["query"], session_id="live-test")
            if block or time.monotonic() >= deadline:
                break

        assert block.startswith("Relevant memories (Recall):")
        assert stored_memory["marker"] in block

        status = provider.recall_status()
        assert status is not None
        assert status.count >= 1
    finally:
        provider.shutdown()


def test_the_search_tool_returns_the_stored_memory(stored_memory, tmp_path, write_recall_config):
    write_recall_config(tmp_path, {"base_url": BASE_URL})
    provider = RecallMemoryProvider(_client())
    provider.initialize("live-test", hermes_home=str(tmp_path), platform="cli")

    try:
        payload = json.loads(
            provider.handle_tool_call(
                "recall_search", {"query": stored_memory["query"], "limit": 5}
            )
        )

        assert "error" not in payload
        assert payload["count"] >= 1
        assert any(stored_memory["marker"] in r["snippet"] for r in payload["results"])
    finally:
        provider.shutdown()


def test_a_bad_key_is_reported_as_an_auth_error():
    bad = RecallClient(api_key="rag_definitely_not_a_valid_key", base_url=BASE_URL)

    with pytest.raises(RecallAuthError):
        bad.search("anything")
