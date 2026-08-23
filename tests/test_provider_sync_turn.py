"""sync_turn: filters, tags, truncation, threading, and what never leaves."""

import json

from recall._client import RecallClient, RecallError
from recall._provider import RecallMemoryProvider

USER = "Why did the pgvector delete wipe the entire collection last night?"
ASSISTANT = "Because an empty ids list is falsy, so the id filter was never appended."


class RecordingClient(RecallClient):
    def __init__(self, raiser=None):
        super().__init__(api_key="rag_k", base_url="https://recall.example")
        self.stored = []
        self.raiser = raiser

    def store(self, content, *, memory_type="context", scope="project", tags=None):
        self.stored.append(
            {"content": content, "memory_type": memory_type, "scope": scope,
             "tags": list(tags or [])}
        )
        if self.raiser is not None:
            raise self.raiser
        return "mem-1"


def _provider(client, tmp_path, **init):
    provider = RecallMemoryProvider(client)
    provider.initialize(
        init.pop("session_id", "s1"),
        hermes_home=str(tmp_path),
        platform=init.pop("platform", "cli"),
        agent_identity=init.pop("agent_identity", "charlotte"),
        **init,
    )
    return provider


def test_sync_turn_stores_the_condensed_turn(tmp_path):
    client = RecordingClient()
    provider = _provider(client, tmp_path)

    provider.sync_turn(USER, ASSISTANT, session_id="s1")
    provider.shutdown()

    assert len(client.stored) == 1
    record = client.stored[0]
    assert record["content"] == f"User: {USER}\nAssistant: {ASSISTANT}"
    assert record["memory_type"] == "context"
    assert record["scope"] == "project"
    assert record["tags"] == ["hermes", "session:s1", "platform:cli", "agent:charlotte"]


def test_sync_turn_skips_a_trivial_prompt(tmp_path):
    client = RecordingClient()
    provider = _provider(client, tmp_path)

    provider.sync_turn("thanks!", ASSISTANT, session_id="s1")
    provider.shutdown()

    assert client.stored == []


def test_sync_turn_skips_a_turn_below_min_chars(tmp_path):
    client = RecordingClient()
    provider = _provider(client, tmp_path)

    provider.sync_turn("why", "because", session_id="s1")
    provider.shutdown()

    assert client.stored == []


def test_sync_turn_respects_sync_turns_false(tmp_path):
    (tmp_path / "recall.json").write_text(json.dumps({"sync_turns": False}))
    client = RecordingClient()
    provider = _provider(client, tmp_path)

    provider.sync_turn(USER, ASSISTANT, session_id="s1")
    provider.shutdown()

    assert client.stored == []


def test_sync_turn_truncates_at_max_chars(tmp_path):
    (tmp_path / "recall.json").write_text(json.dumps({"max_chars": 200}))
    client = RecordingClient()
    provider = _provider(client, tmp_path)

    provider.sync_turn("u" * 5000, "a" * 5000, session_id="s1")
    provider.shutdown()

    assert len(client.stored[0]["content"]) == 200


def test_sync_turn_never_transmits_messages(tmp_path):
    client = RecordingClient()
    provider = _provider(client, tmp_path)
    messages = [
        {"role": "assistant", "content": "", "tool_calls": [{"id": "1"}]},
        {"role": "tool", "tool_call_id": "1", "content": "/home/olivier/secret-path"},
    ]

    provider.sync_turn(USER, ASSISTANT, session_id="s1", messages=messages)
    provider.shutdown()

    assert "secret-path" not in client.stored[0]["content"]
    assert client.stored[0]["content"] == f"User: {USER}\nAssistant: {ASSISTANT}"


def test_sync_turn_does_not_write_for_a_non_primary_agent_context(tmp_path):
    client = RecordingClient()
    provider = _provider(client, tmp_path, agent_context="subagent")

    provider.sync_turn(USER, ASSISTANT, session_id="s1")
    provider.shutdown()

    assert client.stored == []


def test_sync_turn_does_not_raise_when_store_fails(tmp_path):
    client = RecordingClient(raiser=RecallError("unreachable"))
    provider = _provider(client, tmp_path)

    provider.sync_turn(USER, ASSISTANT, session_id="s1")
    provider.shutdown()  # must not raise


def test_sync_turn_returns_immediately_and_does_not_raise_on_arbitrary_errors(tmp_path):
    client = RecordingClient(raiser=RuntimeError("kaboom"))
    provider = _provider(client, tmp_path)

    assert provider.sync_turn(USER, ASSISTANT, session_id="s1") is None
    provider.shutdown()


def test_sync_turn_uses_the_call_session_id_for_tags(tmp_path):
    client = RecordingClient()
    provider = _provider(client, tmp_path, session_id="s1")

    provider.sync_turn(USER, ASSISTANT, session_id="s-other")
    provider.shutdown()

    assert "session:s-other" in client.stored[0]["tags"]
