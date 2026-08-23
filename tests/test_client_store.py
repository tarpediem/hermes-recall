"""RecallClient.store — payload, single retry, auth handling."""

import pytest
from recall import _client
from recall._client import RecallAuthError, RecallClient, RecallError


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.text = text

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class ScriptedRequests:
    """Replays a scripted sequence of outcomes for POST calls."""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        outcome = self.outcomes.pop(0) if self.outcomes else FakeResponse(200, {})
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def get(self, url, **kwargs):  # unused here
        raise AssertionError("store must not issue GET")


@pytest.fixture()
def client():
    return RecallClient(api_key="rag_testkey", base_url="https://recall.example")


def test_store_posts_the_expected_payload(monkeypatch, client):
    fake = ScriptedRequests([FakeResponse(200, {"memory_id": "abc123"})])
    monkeypatch.setattr(_client, "requests", fake)

    memory_id = client.store(
        "User: hi\nAssistant: hello",
        memory_type="context",
        scope="project",
        tags=["hermes", "session:s1"],
    )

    url, kwargs = fake.calls[0]
    assert url == "https://recall.example/api/v1/memories"
    assert kwargs["json"] == {
        "content": "User: hi\nAssistant: hello",
        "memory_type": "context",
        "scope": "project",
        "tags": ["hermes", "session:s1"],
    }
    assert kwargs["timeout"] == _client.WRITE_TIMEOUT
    assert kwargs["headers"]["User-Agent"] == _client.USER_AGENT
    assert memory_id == "abc123"


def test_store_returns_empty_string_when_response_has_no_id(monkeypatch, client):
    monkeypatch.setattr(_client, "requests", ScriptedRequests([FakeResponse(201, {"ok": True})]))

    assert client.store("content") == ""


def test_store_without_api_key_makes_no_call(monkeypatch):
    fake = ScriptedRequests([FakeResponse(200, {})])
    monkeypatch.setattr(_client, "requests", fake)

    assert RecallClient(api_key="", base_url="https://recall.example").store("x") == ""
    assert fake.calls == []


def test_store_with_blank_content_makes_no_call(monkeypatch, client):
    fake = ScriptedRequests([FakeResponse(200, {})])
    monkeypatch.setattr(_client, "requests", fake)

    assert client.store("   ") == ""
    assert fake.calls == []


def test_store_rejects_a_non_canonical_memory_type(monkeypatch, client):
    fake = ScriptedRequests([FakeResponse(200, {})])
    monkeypatch.setattr(_client, "requests", fake)

    with pytest.raises(ValueError):
        client.store("content", memory_type="nonsense")
    assert fake.calls == []


def test_store_retries_once_on_a_transport_failure(monkeypatch, client):
    fake = ScriptedRequests([OSError("connection reset"), FakeResponse(200, {"memory_id": "z"})])
    monkeypatch.setattr(_client, "requests", fake)

    assert client.store("content") == "z"
    assert len(fake.calls) == 2


def test_store_retries_once_on_a_5xx(monkeypatch, client):
    fake = ScriptedRequests([FakeResponse(503, {}), FakeResponse(200, {"memory_id": "z"})])
    monkeypatch.setattr(_client, "requests", fake)

    assert client.store("content") == "z"
    assert len(fake.calls) == 2


def test_store_gives_up_after_exactly_one_retry(monkeypatch, client):
    fake = ScriptedRequests([OSError("down"), OSError("still down")])
    monkeypatch.setattr(_client, "requests", fake)

    with pytest.raises(RecallError):
        client.store("content")
    assert len(fake.calls) == 2


def test_store_does_not_retry_on_401(monkeypatch, client):
    fake = ScriptedRequests([FakeResponse(401, {}, "rag_testkey")])
    monkeypatch.setattr(_client, "requests", fake)

    with pytest.raises(RecallAuthError) as excinfo:
        client.store("content")
    assert len(fake.calls) == 1
    assert "rag_testkey" not in str(excinfo.value)


def test_store_does_not_retry_on_4xx_client_error(monkeypatch, client):
    fake = ScriptedRequests([FakeResponse(422, {})])
    monkeypatch.setattr(_client, "requests", fake)

    with pytest.raises(RecallError):
        client.store("content")
    assert len(fake.calls) == 1


def test_store_defaults_to_context_project_and_no_tags(monkeypatch, client):
    fake = ScriptedRequests([FakeResponse(200, {"memory_id": "d"})])
    monkeypatch.setattr(_client, "requests", fake)

    client.store("content")

    assert fake.calls[0][1]["json"] == {
        "content": "content",
        "memory_type": "context",
        "scope": "project",
        "tags": [],
    }
