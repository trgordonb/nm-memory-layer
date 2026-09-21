"""Periodic nudge: agent-curated memory (the learning loop's curation step).

At a set interval of completed turns, the consumer agent fires an internal
LLM call — no user input — asking the agent to review the recent turn and
decide what, if anything, is worth persisting for future sessions:

- ``memory_manage`` → always-on prompt memory (MEMORY.md / USER.md) for
  knowledge needed in EVERY future session;
- nothing → episodic detail, which the session archive already holds.

This replaces server-side "overview generation" with agent judgment, and is
what keeps prompt memory curated instead of becoming a transcript dump.

The module provides three pieces:

- :class:`NudgePolicy` — turn-count trigger state per session.
- :func:`build_nudge_prompt` — the system prompt for the nudge LLM call.
- :func:`flatten_transcript` — renders LangChain messages as a plain-text
  transcript, so the nudge sees recent activity (including tool calls)
  without sending provider-invalid tool_call/tool_result pairings bound to
  a different toolset.

The consumer owns the nudge LLM call itself: bind its memory tool, run a
short tool loop, print/log the summary. See alt-main.py in langgraph-demo.
"""

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from .prompt_memory import MEMORY_CHAR_LIMIT

DEFAULT_NUDGE_INTERVAL = 5


class NudgePolicy:
    """Tracks turns-since-last-nudge per session; fires every ``interval`` turns."""

    def __init__(self, interval: int = DEFAULT_NUDGE_INTERVAL):
        self.interval = max(1, interval)
        self._since: dict[str, int] = {}

    def record_turn(self, session_id: str) -> None:
        self._since[session_id] = self._since.get(session_id, 0) + 1

    def should_nudge(self, session_id: str) -> bool:
        return self._since.get(session_id, 0) >= self.interval

    def mark_nudged(self, session_id: str) -> None:
        self._since[session_id] = 0

    def turns_pending(self, session_id: str) -> int:
        return self._since.get(session_id, 0)


def build_nudge_prompt(chars_used: int, char_budget: int = MEMORY_CHAR_LIMIT) -> str:
    """System prompt for the internal nudge LLM call."""
    return (
        "You are the memory curator inside an AI agent. You are reviewing one "
        "completed conversation turn — no user is present and nothing is asked of the user.\n"
        "\n"
        "Decide what, if anything, is worth persisting for FUTURE sessions, using the "
        "memory_manage tool:\n"
        "1. target=\"memory\" — facts, decisions, environment quirks, or procedures that "
        "will matter in EVERY future session (one terse line per entry).\n"
        "2. target=\"user\" — durable user preferences or working style observed in the turn.\n"
        "3. Nothing — task-specific and episodic detail already lives in the session "
        "archive (searchable via session_search) and must NOT be duplicated here.\n"
        "\n"
        "Rules:\n"
        "- Most turns produce NO writes. Silence is a valid and expected outcome; write "
        "only what clears the bar of 'useful in a future session'.\n"
        "- Never re-add what is already recorded — current memory contents are shown below.\n"
        f"- Keep every entry terse. Combined MEMORY.md + USER.md budget is {char_budget} chars "
        f"({chars_used} already used); writes over budget are rejected.\n"
        "- You may call memory_manage multiple times. When finished — or immediately, if "
        "nothing is worth keeping — reply with a one-line summary (e.g. 'No memory updates.' "
        "or 'Saved 1 entry to MEMORY.md.').\n"
    )


def flatten_transcript(messages: list[BaseMessage]) -> str:
    """Render messages as a plain-text transcript safe to send with a
    different toolset bound (no tool_call/tool_result pairing constraints)."""
    lines: list[str] = []
    for message in messages:
        if isinstance(message, HumanMessage):
            role = "user"
        elif isinstance(message, AIMessage):
            role = "assistant"
        elif isinstance(message, ToolMessage):
            role = f"tool result ({message.name or 'tool'})"
        else:
            role = message.__class__.__name__.lower()
        content = message.content if isinstance(message.content, str) else str(message.content)
        content = content.strip()
        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls and not content:
            content = "; ".join(
                f"[called {call['name']}({call.get('args')})]" for call in tool_calls
            )
        if content:
            lines.append(f"{role}: {content}")
    return "\n\n".join(lines)
