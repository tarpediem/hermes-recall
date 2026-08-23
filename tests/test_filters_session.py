"""Session-level filters: pair extraction, synthesis, insight extraction."""

from recall._filters import extract_insights, iter_turn_pairs, summarize_session


def _messages():
    return [
        {"role": "system", "content": "you are helpful"},
        {"role": "user", "content": "Should we move marker extraction back to the GPU?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "1", "function": {"name": "bash", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "1", "content": "/home/olivier/secret-path"},
        {
            "role": "assistant",
            "content": "We decided to move it back to the GPU with page-by-page chunking.",
        },
        {"role": "user", "content": "thanks!"},
        {"role": "assistant", "content": "Any time."},
        {"role": "user", "content": "And what about the timeout setting for extraction?"},
        {"role": "assistant", "content": "EXTRACT_TIMEOUT_SECONDS is now 1800 instead of 300."},
    ]


def test_iter_turn_pairs_keeps_only_user_assistant_text():
    pairs = iter_turn_pairs(_messages())

    assert pairs == [
        (
            "Should we move marker extraction back to the GPU?",
            "We decided to move it back to the GPU with page-by-page chunking.",
        ),
        ("thanks!", "Any time."),
        (
            "And what about the timeout setting for extraction?",
            "EXTRACT_TIMEOUT_SECONDS is now 1800 instead of 300.",
        ),
    ]


def test_iter_turn_pairs_never_returns_tool_output():
    flattened = " ".join(u + a for u, a in iter_turn_pairs(_messages()))
    assert "secret-path" not in flattened


def test_iter_turn_pairs_handles_list_content_blocks():
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "block question"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "block answer"}]},
    ]
    assert iter_turn_pairs(messages) == [("block question", "block answer")]


def test_iter_turn_pairs_on_empty_input():
    assert iter_turn_pairs(None) == []
    assert iter_turn_pairs([]) == []


def test_summarize_session_returns_none_below_min_turns():
    messages = [
        {"role": "user", "content": "A single substantive question about the pgvector adapter"},
        {"role": "assistant", "content": "A single substantive answer about the pgvector adapter"},
    ]
    assert summarize_session(messages, min_turns=2) is None


def test_summarize_session_produces_content_and_type():
    result = summarize_session(_messages(), min_turns=2)

    assert result is not None
    content, memory_type = result
    assert "Session summary" in content
    assert "Topics discussed" in content
    assert "marker extraction" in content
    assert memory_type == "decision"


def test_summarize_session_types_as_context_without_decisions():
    messages = [
        {"role": "user", "content": "How many cores does the workstation have in total?"},
        {"role": "assistant", "content": "It has twelve cores and twenty four threads."},
        {"role": "user", "content": "And how much memory is installed on that machine?"},
        {"role": "assistant", "content": "Ninety six gigabytes of DDR5 memory are installed."},
    ]
    result = summarize_session(messages, min_turns=2)

    assert result is not None
    assert result[1] == "context"


def test_summarize_session_skips_trivial_turns_when_counting():
    messages = [
        {"role": "user", "content": "ok"},
        {"role": "assistant", "content": "sure"},
        {"role": "user", "content": "A genuinely substantive question about the graph indexer"},
        {"role": "assistant", "content": "A genuinely substantive answer about the graph indexer"},
    ]
    assert summarize_session(messages, min_turns=2) is None


def test_summarize_session_respects_max_chars():
    messages = []
    for i in range(40):
        messages.append({"role": "user", "content": f"Question number {i} " + "q" * 200})
        messages.append({"role": "assistant", "content": f"We decided answer {i} " + "a" * 200})

    result = summarize_session(messages, max_chars=1000)

    assert result is not None
    assert len(result[0]) <= 1000


def test_extract_insights_returns_decisions_and_preferences():
    messages = [
        {"role": "user", "content": "I always want commit messages written in English please"},
        {"role": "assistant", "content": "We decided to keep marker extraction on the GPU."},
        {"role": "assistant", "content": "The root cause was an empty ids list being falsy."},
    ]

    insights = extract_insights(messages)

    joined = "\n".join(insights)
    assert "marker extraction on the GPU" in joined
    assert "root cause" in joined
    assert len(insights) <= 8


def test_extract_insights_returns_empty_for_small_talk():
    messages = [
        {"role": "user", "content": "hey"},
        {"role": "assistant", "content": "Hello, how can I help you today?"},
    ]
    assert extract_insights(messages) == []


def test_extract_insights_deduplicates_and_caps():
    messages = [
        {"role": "assistant", "content": "We decided to use pgvector."} for _ in range(20)
    ]
    insights = extract_insights(messages, max_items=3)

    assert len(insights) == 1


def test_extract_insights_never_reads_tool_messages():
    messages = [
        {
            "role": "tool",
            "tool_call_id": "1",
            "content": "We decided /secret/path is the root cause",
        },
    ]
    assert extract_insights(messages) == []
