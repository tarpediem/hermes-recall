"""on_pre_compress does two jobs: archive off-thread, return text synchronously."""

from recall._client import RecallClient, RecallError
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


def test_on_pre_compress_returns_a_non_empty_insight_block(tmp_path):
    provider = _provider(RecordingClient(), tmp_path)

    text = provider.on_pre_compress(MESSAGES)
    provider.shutdown()

    assert text.startswith("Insights to preserve from the discarded context:")
    assert "marker extraction on the GPU" in text
    assert "root cause" in text


def test_on_pre_compress_archives_the_discarded_messages(tmp_path):
    client = RecordingClient()
    provider = _provider(client, tmp_path)

    provider.on_pre_compress(MESSAGES)
    provider.shutdown()

    assert len(client.stored) == 1
    assert "pre-compress" in client.stored[0]["tags"]
    assert "Session summary" in client.stored[0]["content"]


def test_on_pre_compress_returns_empty_when_there_is_nothing_to_preserve(tmp_path):
    provider = _provider(RecordingClient(), tmp_path)

    assert provider.on_pre_compress([{"role": "user", "content": "hey"}]) == ""
    provider.shutdown()


def test_on_pre_compress_returns_empty_on_empty_input(tmp_path):
    provider = _provider(RecordingClient(), tmp_path)

    assert provider.on_pre_compress([]) == ""
    provider.shutdown()


def test_on_pre_compress_caps_the_number_of_insight_lines(tmp_path):
    messages = []
    for i in range(30):
        messages.append({"role": "assistant", "content": f"We decided to adopt approach {i}."})
    provider = _provider(RecordingClient(), tmp_path)

    text = provider.on_pre_compress(messages)
    provider.shutdown()

    assert len(text.splitlines()) <= 1 + 8


def test_on_pre_compress_still_returns_text_for_a_non_primary_context(tmp_path):
    client = RecordingClient()
    provider = _provider(client, tmp_path, agent_context="subagent")

    text = provider.on_pre_compress(MESSAGES)
    provider.shutdown()

    assert text != ""
    assert client.stored == [], "non-primary contexts read but never write"


def test_on_pre_compress_returns_text_even_when_the_archive_write_fails(tmp_path):
    provider = _provider(RecordingClient(raiser=RecallError("down")), tmp_path)

    text = provider.on_pre_compress(MESSAGES)
    provider.shutdown()

    assert text.startswith("Insights to preserve")


def test_on_pre_compress_never_raises_on_malformed_messages(tmp_path):
    provider = _provider(RecordingClient(), tmp_path)

    assert provider.on_pre_compress([None, 7, {"role": "tool"}]) == ""  # type: ignore[list-item]
    provider.shutdown()


def test_on_pre_compress_never_leaks_tool_output(tmp_path):
    messages = MESSAGES + [
        {
            "role": "tool",
            "tool_call_id": "1",
            "content": "We decided /home/olivier/secret is root cause",
        }
    ]
    client = RecordingClient()
    provider = _provider(client, tmp_path)

    text = provider.on_pre_compress(messages)
    provider.shutdown()

    assert "secret" not in text
    assert "secret" not in client.stored[0]["content"]
