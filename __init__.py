"""Recall memory provider plugin for Hermes Agent.

Makes a Recall tenant the agent's persistent cross-session memory over direct
REST: relevant memories injected before each turn, filtered writes after each
turn, a session synthesis at the end, and two explicit tools.

Zero runtime dependencies beyond ``requests`` (already pinned by Hermes) and
the stdlib, so the whole install is "paste the repo in the dashboard".
"""

from __future__ import annotations

from pathlib import Path

from ._client import RecallAuthError, RecallClient, RecallError
from ._provider import RecallMemoryProvider

__version__ = "1.2.0"

SKILLS_DIR = Path(__file__).parent / "skills"

__all__ = [
    "RecallAuthError",
    "RecallClient",
    "RecallError",
    "RecallMemoryProvider",
    "SKILLS_DIR",
    "register",
]


def register(ctx) -> None:
    """Plugin entry point.

    The provider is registered FIRST so that a failure in any later
    registration degrades to "no skill", never to "no memory" — the loader's
    ``_ProviderCollector`` keeps whatever was registered before a raise.
    """
    client = RecallClient()
    ctx.register_memory_provider(RecallMemoryProvider(client))
    ctx.register_skill(
        "memory",
        SKILLS_DIR / "memory" / "SKILL.md",
        "Use Recall as persistent memory: when to search, what to store",
    )
