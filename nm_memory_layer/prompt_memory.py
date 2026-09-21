"""Prompt memory: the always-on memory layer (Hermes-style MEMORY.md / USER.md).

Two small markdown files are injected into the system prompt at the start of
every session:

- ``MEMORY.md`` — knowledge that matters in *every* future session: project
  facts, standing decisions, environment quirks.
- ``USER.md`` — who the user is, preferences, working style.

The combined size of both files is capped (default 3,575 chars) to force
curation over accumulation. The agent curates them itself via the
``memory_manage`` tool (add / replace / remove). Per the Hermes rule, edits
take effect from the *next* session: the consumer loads the block once at
session start and keeps the system-prompt prefix stable (which also keeps
provider prompt caches warm).
"""

import os
from pathlib import Path
from typing import Literal

from langchain_core.tools import tool

MEMORY_CHAR_LIMIT = 3575

_TARGET_FILES = {"memory": "MEMORY.md", "user": "USER.md"}


class PromptMemory:
    """Manages the always-on MEMORY.md / USER.md pair inside one directory."""

    def __init__(self, memory_dir: str | None = None):
        self.memory_dir = Path(memory_dir or os.getenv("MEMORY_DIR", "memories"))
        self.memory_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, target: str) -> Path:
        return self.memory_dir / _TARGET_FILES[target]

    def _read(self, target: str) -> str:
        path = self._path(target)
        return path.read_text() if path.exists() else ""

    def total_chars(self) -> int:
        return len(self._read("memory")) + len(self._read("user"))

    def load(self) -> str:
        """Render the always-on block for the system prompt ("" if nothing stored)."""
        memory = self._read("memory").strip()
        user = self._read("user").strip()
        if not memory and not user:
            return ""
        sections = []
        if memory:
            sections.append(f"<memory>\n{memory}\n</memory>")
        if user:
            sections.append(f"<user>\n{user}\n</user>")
        return "<agent_memory>\n" + "\n".join(sections) + "\n</agent_memory>"

    def _write_checked(self, target: str, new_content: str) -> str:
        other = len(self._read("user" if target == "memory" else "memory"))
        if other + len(new_content) > MEMORY_CHAR_LIMIT:
            return (
                f"Rejected: MEMORY.md + USER.md would exceed the combined "
                f"{MEMORY_CHAR_LIMIT}-char budget ({other + len(new_content)} needed). "
                f"Consolidate or remove entries first."
            )
        self._path(target).write_text(new_content)
        return f"OK: {target} updated ({self.total_chars()}/{MEMORY_CHAR_LIMIT} chars used)"

    def add(self, target: str, text: str) -> str:
        text = text.strip()
        if not text:
            return "Rejected: nothing to add"
        current = self._read(target)
        entry = text if text.startswith(("-", "*")) else f"- {text}"
        new_content = current.rstrip() + "\n" + entry + "\n" if current.strip() else entry + "\n"
        return self._write_checked(target, new_content)

    def replace(self, target: str, old_text: str, new_text: str) -> str:
        current = self._read(target)
        if old_text.strip() not in current:
            return f"Rejected: text not found in {_TARGET_FILES[target]}; use add() for new entries"
        new_content = current.replace(old_text.strip(), new_text.strip(), 1)
        return self._write_checked(target, new_content)

    def remove(self, target: str, old_text: str) -> str:
        current = self._read(target)
        if old_text.strip() not in current:
            return f"Rejected: text not found in {_TARGET_FILES[target]}"
        new_content = current.replace(old_text.strip(), "", 1)
        new_content = "\n".join(line for line in new_content.splitlines() if line.strip()) + "\n"
        return self._write_checked(target, new_content)


def create_memory_manage_tool(memory: PromptMemory):
    """Factory returning the agent-callable memory_manage tool bound to a PromptMemory."""

    @tool
    def memory_manage(
        operation: Literal["add", "replace", "remove"],
        target: Literal["memory", "user"],
        content: str = "",
        old_content: str = "",
    ) -> str:
        """Edit your always-on memory files (MEMORY.md / USER.md), injected into every
        future session's system prompt. Combined budget across both files: 3575 chars —
        keep entries terse. Edits take effect from the NEXT session.

        Use ONLY for knowledge needed in EVERY future session:
        - memory: project facts, standing decisions, environment quirks, learned
          procedures that keep applying (one short line per entry).
        - user: who the user is, preferences, communication/working style.
        Do NOT put topic-specific findings or one-off task history here — leave
        those to session_search (episodic archive) or say nothing.

        Args:
            operation: "add" appends a line (content); "replace" rewrites an existing
                line (old_content -> content); "remove" deletes an existing line (old_content).
            target: "memory" (MEMORY.md) or "user" (USER.md).
            content: new text (for add / replace).
            old_content: existing text to find (for replace / remove).
        """
        if operation == "add":
            return memory.add(target, content)
        if operation == "replace":
            return memory.replace(target, old_content, content)
        if operation == "remove":
            return memory.remove(target, old_content)
        return f"Rejected: unknown operation {operation!r}"

    return memory_manage
