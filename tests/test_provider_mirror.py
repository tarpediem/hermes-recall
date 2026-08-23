"""Mirroring the built-in memory tool, and observing delegations."""

import json

from recall._client import RecallClient, RecallError
from recall._provider import RecallMemoryProvider


class RecordingClient(RecallClient):
    def __init__(self, raiser=None):
        super().__init__(api_key="rag_k", base_url="https://recall.example")
        self.stored = []
        self.raiser = raiser

    def store(self, content, *, memory_type="context", scope="project", tags=None):
        self.stored.append(
            {"content": content, "memory_type": memory_type, "tags": list(tags or [])}
        )
        if self.raiser is not None:
            raise self.raiser
        return "mem-1"


def _provider(client, tmp_path, **init):
    provider = RecallMemoryProvider(client)
    provider.initialize(
        "s1", hermes_home=str(tmp_path), platform="cli",
        agent_identity="charlotte", **init
    )
    return provider


def test_on_memory_write_add_to_user_stores_a_preference(tmp_path):
    client = RecordingClient()
    provider = _provider(client, tmp_path)

    provider.on_memory_write("add", "user", "Olivier prefers fish, not bash or zsh")
    provider.shutdown()

    record = client.stored[0]
    assert record["memory_type"] == "preference"
    assert "Olivier prefers fish" in record["content"]
    assert "builtin-mirror" in record["tags"]


def test_on_memory_write_add_to_memory_stores_context(tmp_path):
    client = RecordingClient()
    provider = _provider(client, tmp_path)

    provider.on_memory_write("add", "memory", "The RAG backend lives on LXC 123")
    provider.shutdown()

    assert client.stored[0]["memory_type"] == "context"


def test_on_memory_write_replace_is_mirrored(tmp_path):
    client = RecordingClient()
    provider = _provider(client, tmp_path)

    provider.on_memory_write("replace", "memory", "The RAG backend now lives on LXC 127")
    provider.shutdown()

    assert len(client.stored) == 1


def test_on_memory_write_remove_is_a_noop(tmp_path):
    client = RecordingClient()
    provider = _provider(client, tmp_path)

    provider.on_memory_write("remove", "memory", "The RAG backend lives on LXC 123")
    provider.shutdown()

    assert client.stored == []


def test_on_memory_write_is_callable_with_three_positional_args_hermes_0_19_1(tmp_path):
    client = RecordingClient()
    provider = _provider(client, tmp_path)

    provider.on_memory_write("add", "memory", "A three-positional-argument call site")
    provider.shutdown()

    assert len(client.stored) == 1


def test_on_memory_write_is_callable_with_metadata_hermes_0_20_4(tmp_path):
    client = RecordingClient()
    provider = _provider(client, tmp_path)

    provider.on_memory_write(
        "add", "memory", "A four-argument call site",
        metadata={"write_origin": "tool", "session_id": "s1"},
    )
    provider.shutdown()

    assert len(client.stored) == 1


def test_on_memory_write_skips_empty_content(tmp_path):
    client = RecordingClient()
    provider = _provider(client, tmp_path)

    provider.on_memory_write("add", "memory", "   ")
    provider.shutdown()

    assert client.stored == []


def test_on_memory_write_does_not_write_for_non_primary_context(tmp_path):
    client = RecordingClient()
    provider = _provider(client, tmp_path, agent_context="cron")

    provider.on_memory_write("add", "memory", "Something a cron run produced")
    provider.shutdown()

    assert client.stored == []


def test_on_memory_write_does_not_raise_when_store_fails(tmp_path):
    provider = _provider(RecordingClient(raiser=RecallError("down")), tmp_path)

    provider.on_memory_write("add", "memory", "A perfectly reasonable memory entry")
    provider.shutdown()


def test_on_delegation_stores_task_and_result(tmp_path):
    client = RecordingClient()
    provider = _provider(client, tmp_path)

    provider.on_delegation(
        "Audit the pgvector adapter for falsy-guard bugs",
        "Found one: delete() with an empty ids list wipes the collection.",
        child_session_id="child-9",
    )
    provider.shutdown()

    record = client.stored[0]
    assert record["memory_type"] == "context"
    assert "Audit the pgvector adapter" in record["content"]
    assert "Found one" in record["content"]
    assert "delegation" in record["tags"]


def test_on_delegation_truncates_at_max_chars(tmp_path):
    (tmp_path / "recall.json").write_text(json.dumps({"max_chars": 150}))
    client = RecordingClient()
    provider = _provider(client, tmp_path)

    provider.on_delegation("t" * 5000, "r" * 5000, child_session_id="c")
    provider.shutdown()

    assert len(client.stored[0]["content"]) == 150


def test_on_delegation_skips_an_empty_result(tmp_path):
    client = RecordingClient()
    provider = _provider(client, tmp_path)

    provider.on_delegation("A task that produced nothing at all", "")
    provider.shutdown()

    assert client.stored == []


def test_on_delegation_does_not_write_for_non_primary_context(tmp_path):
    client = RecordingClient()
    provider = _provider(client, tmp_path, agent_context="subagent")

    provider.on_delegation("A delegated task", "A delegated result worth remembering")
    provider.shutdown()

    assert client.stored == []


def test_on_delegation_does_not_raise_when_store_fails(tmp_path):
    provider = _provider(RecordingClient(raiser=RuntimeError("kaboom")), tmp_path)

    provider.on_delegation("A delegated task", "A delegated result worth remembering")
    provider.shutdown()
