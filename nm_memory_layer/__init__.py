"""nm-memory-layer: standalone agent memory layer (SQLite + FTS5).

Provides a Hermes-style memory architecture:

- Episodic memory: SessionStore persists every agent turn; the agent
  deliberately searches past sessions via the session_search tool.
- Prompt memory: PromptMemory curates the always-on MEMORY.md / USER.md
  pair injected into every session's system prompt.
"""

from .nudge import DEFAULT_NUDGE_INTERVAL, NudgePolicy, build_nudge_prompt, flatten_transcript
from .prompt_memory import MEMORY_CHAR_LIMIT, PromptMemory, create_memory_manage_tool
from .store import SessionSearchHit, SessionStore, create_session_search_tool

__all__ = [
    "DEFAULT_NUDGE_INTERVAL",
    "MEMORY_CHAR_LIMIT",
    "NudgePolicy",
    "PromptMemory",
    "SessionSearchHit",
    "SessionStore",
    "build_nudge_prompt",
    "create_memory_manage_tool",
    "create_session_search_tool",
    "flatten_transcript",
]
