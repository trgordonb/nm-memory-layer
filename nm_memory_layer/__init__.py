"""nm-memory-layer: standalone agent memory layer (SQLite + FTS5).

Provides a Hermes-style memory architecture:

- Episodic memory: SessionStore persists every agent turn; the agent
  deliberately searches past sessions via the session_search tool.
- Prompt memory: PromptMemory curates the always-on MEMORY.md / USER.md
  pair injected into every session's system prompt.
"""

from .nudge import DEFAULT_NUDGE_INTERVAL, NudgePolicy, build_nudge_prompt, flatten_transcript
from .prompt_memory import MEMORY_CHAR_LIMIT, PromptMemory, create_memory_manage_tool
from .skills import SkillLibrary, create_load_skill_tool, create_skill_manage_tool
from .store import SessionSearchHit, SessionStore, create_session_search_tool
from .summarizer import SearchSummarizer, SummarizedSearch, create_openrouter_summarizer

__all__ = [
    "DEFAULT_NUDGE_INTERVAL",
    "MEMORY_CHAR_LIMIT",
    "NudgePolicy",
    "PromptMemory",
    "SearchSummarizer",
    "SessionSearchHit",
    "SessionStore",
    "SkillLibrary",
    "SummarizedSearch",
    "build_nudge_prompt",
    "create_load_skill_tool",
    "create_memory_manage_tool",
    "create_openrouter_summarizer",
    "create_session_search_tool",
    "create_skill_manage_tool",
    "flatten_transcript",
]
