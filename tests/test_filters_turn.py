"""Pure turn filters: what is worth storing, and how a turn is condensed."""

from recall._filters import condense_turn, is_trivial_prompt, is_worth_storing, truncate


def test_is_trivial_prompt_matches_the_core_gate():
    assert is_trivial_prompt("") is True
    assert is_trivial_prompt("   ") is True
    assert is_trivial_prompt("/reset") is True
    assert is_trivial_prompt("thanks!") is True
    assert is_trivial_prompt("ok") is True
    assert is_trivial_prompt("hey.") is True
    assert is_trivial_prompt("k8s cluster is down") is False
    assert is_trivial_prompt("why did the pgvector delete wipe the table") is False


def test_trivial_user_prompt_is_not_worth_storing():
    assert is_worth_storing("thanks!", "You are welcome, glad it worked out fine.") is False


def test_short_combined_turn_is_not_worth_storing():
    assert is_worth_storing("why", "because", min_chars=40) is False


def test_substantive_turn_is_worth_storing():
    user = "Why did the pgvector delete wipe the whole collection?"
    assistant = "Because an empty ids list is falsy, so the id filter was never appended."
    assert is_worth_storing(user, assistant, min_chars=40) is True


def test_empty_assistant_reply_is_not_worth_storing():
    assert is_worth_storing("A perfectly long and substantive question about pgvector", "") is False


def test_min_chars_counts_user_plus_assistant():
    user = "a" * 20
    assistant = "b" * 25
    assert is_worth_storing(user, assistant, min_chars=40) is True
    assert is_worth_storing(user, assistant, min_chars=100) is False


def test_condense_turn_produces_the_documented_shape():
    assert condense_turn("hello there", "general kenobi") == (
        "User: hello there\nAssistant: general kenobi"
    )


def test_condense_turn_truncates_at_max_chars():
    out = condense_turn("u" * 5000, "a" * 5000, max_chars=4000)
    assert len(out) == 4000
    assert out.endswith("…")


def test_condense_turn_strips_surrounding_whitespace():
    assert condense_turn("  q  ", "  a  ") == "User: q\nAssistant: a"


def test_truncate_leaves_short_text_untouched():
    assert truncate("short", 100) == "short"


def test_truncate_marks_cut_text_with_an_ellipsis():
    out = truncate("x" * 50, 10)
    assert len(out) == 10
    assert out == "x" * 9 + "…"
