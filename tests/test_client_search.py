"""RecallClient.search — parameters, parsing, auth, error mapping."""

import types

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


class FakeRequests:
    """Stand-in for the `requests` module inside recall._client."""

    def __init__(self, response=None, raiser=None):
        self.response = response
        self.raiser = raiser
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append(("GET", url, kwargs))
        if self.raiser is not None:
            raise self.raiser
        return self.response

    def post(self, url, **kwargs):
        self.calls.append(("POST", url, kwargs))
        if self.raiser is not None:
            raise self.raiser
        return self.response


@pytest.fixture()
def client():
    return RecallClient(api_key="rag_testkey", base_url="https://recall.example")


def test_search_sends_the_specified_parameters(monkeypatch, client):
    fake = FakeRequests(FakeResponse(200, {"items": []}))
    monkeypatch.setattr(_client, "requests", fake)

    client.search("where did we move marker extraction", limit=7)

    method, url, kwargs = fake.calls[0]
    assert method == "GET"
    assert url == "https://recall.example/api/v1/memory/search"
    assert kwargs["params"] == {
        "query": "where did we move marker extraction",
        "limit": 7,
        "rerank": "true",
        "graph_boost": "false",
        "scope": "all",
    }
    assert kwargs["timeout"] == (_client.CONNECT_TIMEOUT, _client.READ_TIMEOUT)
    assert kwargs["headers"]["Authorization"] == "ApiKey rag_testkey"
    assert kwargs["headers"]["User-Agent"] == _client.USER_AGENT


def test_search_returns_only_dict_items(monkeypatch, client):
    payload = {"items": [{"id": "a", "snippet": "s"}, "junk", {"id": "b"}], "meta": {}}
    monkeypatch.setattr(_client, "requests", FakeRequests(FakeResponse(200, payload)))

    items = client.search("q")

    assert items == [{"id": "a", "snippet": "s"}, {"id": "b"}]


def test_search_without_api_key_makes_no_call(monkeypatch):
    fake = FakeRequests(FakeResponse(200, {"items": [{"id": "a"}]}))
    monkeypatch.setattr(_client, "requests", fake)

    assert RecallClient(api_key="", base_url="https://recall.example").search("q") == []
    assert fake.calls == []


def test_search_with_blank_query_makes_no_call(monkeypatch, client):
    fake = FakeRequests(FakeResponse(200, {"items": []}))
    monkeypatch.setattr(_client, "requests", fake)

    assert client.search("   ") == []
    assert fake.calls == []


def test_search_truncates_a_very_long_query(monkeypatch, client):
    fake = FakeRequests(FakeResponse(200, {"items": []}))
    monkeypatch.setattr(_client, "requests", fake)

    client.search("x" * 5000)

    assert len(fake.calls[0][2]["params"]["query"]) == _client.MAX_QUERY_CHARS


def test_search_clamps_limit_to_the_server_range(monkeypatch, client):
    fake = FakeRequests(FakeResponse(200, {"items": []}))
    monkeypatch.setattr(_client, "requests", fake)

    client.search("q", limit=500)
    assert fake.calls[0][2]["params"]["limit"] == 100

    client.search("q", limit=0)
    assert fake.calls[1][2]["params"]["limit"] == 1


def test_search_sends_tags_as_csv_when_given(monkeypatch, client):
    fake = FakeRequests(FakeResponse(200, {"items": []}))
    monkeypatch.setattr(_client, "requests", fake)

    client.search("q", tags=["hermes", "session:abc"])

    assert fake.calls[0][2]["params"]["tags"] == "hermes,session:abc"


def test_search_401_raises_auth_error(monkeypatch, client):
    monkeypatch.setattr(_client, "requests", FakeRequests(FakeResponse(401, {}, "nope")))

    with pytest.raises(RecallAuthError):
        client.search("q")


def test_search_500_raises_recall_error(monkeypatch, client):
    monkeypatch.setattr(_client, "requests", FakeRequests(FakeResponse(500, {}, "boom")))

    with pytest.raises(RecallError) as excinfo:
        client.search("q")
    assert "500" in str(excinfo.value)


def test_search_malformed_json_raises_recall_error(monkeypatch, client):
    bad = FakeResponse(200, ValueError("not json"))
    monkeypatch.setattr(_client, "requests", FakeRequests(bad))

    with pytest.raises(RecallError):
        client.search("q")


def test_errors_never_contain_the_api_key(monkeypatch, client):
    monkeypatch.setattr(_client, "requests", FakeRequests(FakeResponse(401, {}, "rag_testkey")))

    with pytest.raises(RecallAuthError) as excinfo:
        client.search("q")
    assert "rag_testkey" not in str(excinfo.value)
    assert "rag_testkey" not in repr(client)


def test_client_reads_env_when_not_given_explicit_values(monkeypatch):
    monkeypatch.setenv("RECALL_API_KEY", "rag_fromenv")
    monkeypatch.setenv("RECALL_BASE_URL", "https://preprod.example/")

    c = RecallClient()

    assert c.api_key == "rag_fromenv"
    assert c.base_url == "https://preprod.example"


def test_requests_seam_is_a_module_attribute():
    assert isinstance(_client.requests, types.ModuleType)


def test_the_timeout_is_a_connect_read_tuple(monkeypatch, client):
    """A scalar timeout is a READ budget only: requests applies it to the
    connect phase separately, so a wedged host could spend 3 s connecting AND
    3 s reading. The tuple is what the documented budget actually costs."""
    fake = FakeRequests(FakeResponse(200, {"items": []}))
    monkeypatch.setattr(_client, "requests", fake)

    client.search("anything")

    timeout = fake.calls[0][2]["timeout"]
    assert isinstance(timeout, tuple)
    assert timeout == (_client.CONNECT_TIMEOUT, _client.READ_TIMEOUT)
    assert _client.CONNECT_TIMEOUT == 1.5
