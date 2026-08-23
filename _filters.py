"""Pure, network-free decision and condensation helpers.

Nothing here touches the Recall API, threads, or provider state — which is
exactly why it is a separate module: these are the only rules worth testing
without a provider instance.
"""

from __future__ import annotations

import re
from typing import Any

try:  # Hermes >= 0.19 ships the canonical gate; use it so we cannot drift.
    from agent.memory_provider import is_trivial_prompt  # type: ignore
except Exception:  # pragma: no cover - only hit outside a Hermes checkout
    _TRIVIAL_RE = re.compile(
        r"^(yes|no|ok|okay|sure|thanks|thank you|y|n|yep|nope|yeah|nah|"
        r"hi|hey|hello|yo|sup|"
        r"continue|go ahead|do it|proceed|got it|cool|nice|great|done|next|lgtm|k)"
        r"[\s!?.:;,\"'~‘’“”—–…()\[\]{}<>*&^%$#@!+=` ]*$",
        re.IGNORECASE,
    )

    def is_trivial_prompt(text: str | None) -> bool:
        if not text:
            return True
        stripped = text.strip()
        if not stripped:
            return True
        if stripped.startswith("/"):
            return True
        return bool(_TRIVIAL_RE.match(stripped))


ELLIPSIS = "…"


def truncate(text: str, limit: int) -> str:
    """Cut ``text`` to ``limit`` characters, marking the cut with an ellipsis."""
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    if limit == 1:
        return ELLIPSIS
    return text[: limit - 1] + ELLIPSIS


def is_worth_storing(
    user_content: str,
    assistant_content: str,
    *,
    min_chars: int = 40,
) -> bool:
    """Decide whether a completed turn earns a Recall write.

    Filters, in the order the spec lists them: trivial user prompt, then a
    combined length below ``min_chars``. An empty assistant reply is never
    worth storing — there is no answer to remember.
    """
    user = (user_content or "").strip()
    assistant = (assistant_content or "").strip()
    if not user or not assistant:
        return False
    if is_trivial_prompt(user):
        return False
    return len(user) + len(assistant) >= min_chars


def condense_turn(
    user_content: str,
    assistant_content: str,
    *,
    max_chars: int = 4000,
) -> str:
    """Render one turn as the exact text stored in Recall."""
    user = (user_content or "").strip()
    assistant = (assistant_content or "").strip()
    return truncate(f"User: {user}\nAssistant: {assistant}", max_chars)


# -- session-level helpers -------------------------------------------------

DECISION_RE = re.compile(
    r"\b(we decided|decided to|decision|we will|we'll|chose|chosen|"
    r"switch(?:ed|ing)? to|going with|instead of|agreed to|settled on)\b",
    re.IGNORECASE,
)
PREFERENCE_RE = re.compile(
    r"\b(i (?:always|never|prefer|want|like|hate)|please always|please never|"
    r"from now on|my preference|don't ever|do not ever)\b",
    re.IGNORECASE,
)
FACT_RE = re.compile(
    r"\b(root cause|because|the fix (?:was|is)|turns out|caused by|"
    r"the reason (?:was|is)|is located at|runs on|listens on)\b",
    re.IGNORECASE,
)

_INSIGHT_LINE_CHARS = 200
_SUMMARY_TOPIC_CHARS = 120
_SUMMARY_LINE_CHARS = 220


def _text_of(message: dict[str, Any]) -> str:
    """Return the plain text of a message, or "" when it carries none."""
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "\n".join(parts).strip()
    return ""


def iter_turn_pairs(messages: list[dict[str, Any]] | None) -> list[tuple[str, str]]:
    """Pair each user message with the next plain-text assistant reply.

    Drops ``system`` and ``tool`` roles entirely, and drops assistant messages
    that carry ``tool_calls`` — the returned pairs are the ONLY thing that may
    ever leave the device, so tool output can never reach a payload.
    """
    pairs: list[tuple[str, str]] = []
    pending_user: str | None = None
    for message in messages or []:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role == "user":
            text = _text_of(message)
            if text:
                pending_user = text
            continue
        if role != "assistant":
            continue
        if message.get("tool_calls"):
            continue
        text = _text_of(message)
        if text and pending_user is not None:
            pairs.append((pending_user, text))
            pending_user = None
    return pairs


def _useful_pairs(messages: list[dict[str, Any]] | None) -> list[tuple[str, str]]:
    return [(u, a) for u, a in iter_turn_pairs(messages) if not is_trivial_prompt(u)]


def summarize_session(
    messages: list[dict[str, Any]] | None,
    *,
    max_chars: int = 4000,
    min_turns: int = 2,
) -> tuple[str, str] | None:
    """Condense a conversation into one synthesis, not a dump.

    Returns ``(content, memory_type)`` — ``memory_type`` is ``"decision"``
    when explicit decisions were identified, ``"context"`` otherwise — or
    ``None`` when there are fewer than ``min_turns`` useful turns.
    """
    pairs = _useful_pairs(messages)
    if len(pairs) < min_turns:
        return None

    topics = [truncate(user, _SUMMARY_TOPIC_CHARS) for user, _ in pairs]
    decisions: list[str] = []
    facts: list[str] = []
    for _user, assistant in pairs:
        for line in assistant.splitlines():
            stripped = line.strip("-* \t")
            if not stripped:
                continue
            if DECISION_RE.search(stripped):
                decisions.append(truncate(stripped, _SUMMARY_LINE_CHARS))
            elif FACT_RE.search(stripped) or PREFERENCE_RE.search(stripped):
                facts.append(truncate(stripped, _SUMMARY_LINE_CHARS))

    decisions = _dedupe(decisions)[:10]
    facts = _dedupe(facts)[:10]

    lines = [f"Session summary ({len(pairs)} turns).", "", "Topics discussed:"]
    lines += [f"- {topic}" for topic in topics[:12]]
    if decisions:
        lines += ["", "Decisions:"] + [f"- {item}" for item in decisions]
    if facts:
        lines += ["", "Facts learned:"] + [f"- {item}" for item in facts]

    memory_type = "decision" if decisions else "context"
    return truncate("\n".join(lines), max_chars), memory_type


def extract_insights(
    messages: list[dict[str, Any]] | None,
    *,
    max_items: int = 8,
) -> list[str]:
    """Pull durable decisions, preferences and causal facts out of messages.

    Reads only ``user`` and ``assistant`` plain text — never ``tool`` output.
    """
    found: list[str] = []
    for message in messages or []:
        if not isinstance(message, dict):
            continue
        if message.get("role") not in {"user", "assistant"}:
            continue
        if message.get("tool_calls"):
            continue
        for line in _text_of(message).splitlines():
            stripped = line.strip("-* \t")
            if len(stripped) < 20:
                continue
            if (
                DECISION_RE.search(stripped)
                or PREFERENCE_RE.search(stripped)
                or FACT_RE.search(stripped)
            ):
                found.append(truncate(stripped, _INSIGHT_LINE_CHARS))
    return _dedupe(found)[:max_items]


def _dedupe(items: list[str]) -> list[str]:
    seen = set()
    unique = []
    for item in items:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique
