"""Pure, network-free decision and condensation helpers.

Nothing here touches the Recall API, threads, or provider state — which is
exactly why it is a separate module: these are the only rules worth testing
without a provider instance.
"""

from __future__ import annotations

import re

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
