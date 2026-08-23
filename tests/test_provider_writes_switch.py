"""``writes_enabled`` — the one switch that covers every write path.

The four throttles (``sync_turns``, ``session_summary``, ``min_chars``,
``max_chars``) each govern a single path; none of them can make the plugin
read-only. This module pins the master switch to *every* path that stores,
so a future write path added without going through ``_store_async`` shows up
here as a failure rather than as an unexpected memory in production.
"""

import json

from recall._client import RecallClient
from recall._provider import RecallMemoryProvider

MESSAGES = [
    {"role": "user", "content": "I always want commit messages written in English please"},
    {
        "role": "assistant",
        "content": "Understood. We decided to keep marker extraction on the GPU.",
    },
    {"role": "user", "content": "And why did the collection get wiped during the night?"},
    {"role": "assistant", "content": "The root cause was an empty ids list being falsy."},
]

TURN = (
    "Why did the graph indexer stop extracting entities after the upgrade?",
    "Because GLiNER was replaced by an LLM-first extractor whose endpoint was unset.",
)


class RecordingClient(RecallClient):
    def __init__(self):
        super().__init__(api_key="rag_k", base_url="https://recall.example")
        self.stored = []

    def store(self, content, *, memory_type="context", scope="project", tags=None):
        self.stored.append({"content": content, "memory_type": memory_type})
        return "mem-1"

    def search(self, query, *, limit=5, rerank=True, graph_boost=False, tags=None, timeout=None):
        return [
            {
                "id": "m1",
                "type": "bugfix",
                "timestamp": "2026-07-10T02:17:00Z",
                "snippet": "An empty ids list was falsy, so delete() dropped the collection",
            }
        ]


def _provider(tmp_path, write_recall_config, config=None, **init):
    if config is not None:
        write_recall_config(tmp_path, config)
    provider = RecallMemoryProvider(RecordingClient())
    provider.initialize(
        "s1", hermes_home=str(tmp_path), platform="cli", agent_identity="charlotte", **init
    )
    return provider


def _exercise_every_write_path(provider):
    provider.sync_turn(*TURN, session_id="s1")
    provider.on_session_end(MESSAGES)
    provider.on_pre_compress(MESSAGES)
    provider.on_delegation("summarise the incident", "the SSD was failing")
    provider.on_memory_write("add", "user", "Olivier prefers fish over bash")
    raw = provider.handle_tool_call(
        "recall_store", {"content": "a durable fact worth keeping about the extractor"}
    )
    provider.shutdown()
    return raw


# -- the switch off ----------------------------------------------------------


def test_writes_enabled_false_stops_every_write_path(tmp_path, write_recall_config):
    provider = _provider(tmp_path, write_recall_config, {"writes_enabled": False})

    _exercise_every_write_path(provider)

    assert provider._client.stored == []


def test_writes_enabled_false_makes_the_store_tool_return_a_clear_error(
    tmp_path, write_recall_config
):
    provider = _provider(tmp_path, write_recall_config, {"writes_enabled": False})

    raw = provider.handle_tool_call("recall_store", {"content": "something worth keeping"})
    provider.shutdown()

    result = json.loads(raw)
    assert "error" in result
    assert "writes_enabled" in result["error"]
    assert provider._client.stored == []


def test_writes_enabled_false_still_reads(tmp_path, write_recall_config):
    """Read-only means read-only, not off."""
    provider = _provider(tmp_path, write_recall_config, {"writes_enabled": False})

    raw = provider.handle_tool_call("recall_search", {"query": "why was the collection wiped"})
    provider.shutdown()

    assert "error" not in json.loads(raw)


def test_writes_enabled_false_still_returns_the_pre_compress_insights(
    tmp_path, write_recall_config
):
    """The insight block is a READ of the messages already in context — it
    leaves nothing on the server, so the write switch must not suppress it."""
    provider = _provider(tmp_path, write_recall_config, {"writes_enabled": False})

    text = provider.on_pre_compress(MESSAGES)
    provider.shutdown()

    assert text.startswith("Insights to preserve")
    assert provider._client.stored == []


# -- the switch on (default) -------------------------------------------------


def test_writes_enabled_defaults_to_on(tmp_path, write_recall_config):
    provider = _provider(tmp_path, write_recall_config, {})

    _exercise_every_write_path(provider)

    assert len(provider._client.stored) == 6


def test_the_agent_context_gate_still_applies_with_writes_enabled_on(
    tmp_path, write_recall_config
):
    """Two independent gates: config, and the non-primary agent context."""
    provider = _provider(
        tmp_path, write_recall_config, {"writes_enabled": True}, agent_context="subagent"
    )

    raw = _exercise_every_write_path(provider)

    assert provider._client.stored == []
    assert "subagent" in json.loads(raw)["error"]


# -- the pre-compress archive is a session synthesis -------------------------


def test_pre_compress_archive_honours_session_summary(tmp_path, write_recall_config):
    """It writes the same kind of memory ``on_session_end`` does, so the same
    key must govern it — otherwise turning summaries off still writes one."""
    provider = _provider(tmp_path, write_recall_config, {"session_summary": False})

    text = provider.on_pre_compress(MESSAGES)
    provider.shutdown()

    assert provider._client.stored == []
    assert text.startswith("Insights to preserve")


def test_pre_compress_archives_when_session_summary_is_on(tmp_path, write_recall_config):
    provider = _provider(tmp_path, write_recall_config, {"session_summary": True})

    provider.on_pre_compress(MESSAGES)
    provider.shutdown()

    assert len(provider._client.stored) == 1
