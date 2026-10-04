"""Skills: procedural memory (Hermes-style, agentskills.io-compatible).

Skills are reusable procedures — markdown files the agent writes to capture
workflows that worked, then reloads instead of retracing steps. Storage
follows the agentskills.io open standard (SKILL.md with YAML frontmatter), so
files stay portable across compatible agents:

    skills/
    ├── edgar/
    │   ├── SKILL.md              # main instructions (required)
    │   └── references/           # supporting docs, loaded on demand
    └── devops/deploy-k8s/        # optional category nesting
        └── SKILL.md

Loading follows **progressive disclosure**: the consumer injects only names
and short descriptions (render_index) into the system prompt, and the full
SKILL.md enters context only when the agent decides it is relevant
(load_skill). Token cost stays flat no matter how many skills exist.

The agent curates the library itself through the ``skill_manage`` tool. Its
actions mirror the Hermes toolset — create, patch, edit, delete, write_file,
remove_file, enable, disable — with ``patch`` as the preferred update path: a
targeted string replacement is both safer (no risk of breaking working
content) and more token-efficient than a full rewrite.

Since 0.2, all storage goes through ``nm-skills-registry`` (SkillRegistry):
a local directory by default, or an S3-compatible registry with per-user
enable/disable when ``SKILLS_REGISTRY`` is set (e.g.
``SKILLS_REGISTRY=s3://bucket`` + R2/AWS credentials). The local ``skills/``
directory stays the materialized working copy either way, so skill scripts
keep running from real paths. ``SkillLibrary`` is now a thin facade over the
registry; behavior is pinned by parity tests on both sides.

Skill-creation triggers (evaluated by the periodic nudge): a turn with many
tool calls, recovery from an error, a user correction, or a non-obvious
workflow that worked.
"""

import os
import re
from pathlib import Path
from typing import Literal

from langchain_core.tools import tool

from nm_skills_registry import SkillRegistry, registry_from_env

_SKILL_FILE = "SKILL.md"
_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


class SkillLibrary:
    """Manages a directory of agentskills.io-style skills.

    A facade over :class:`nm_skills_registry.SkillRegistry`. With
    ``SKILLS_REGISTRY`` unset the registry is a plain local directory (the
    historical behavior); with it set, skills live in S3-compatible storage
    with the local directory as a write-through mirror.
    """

    def __init__(self, skills_dir: str | None = None, registry: SkillRegistry | None = None):
        self.skills_dir = Path(skills_dir or os.getenv("SKILLS_DIR", "skills"))
        self._registry = registry or registry_from_env(self.skills_dir)

    @property
    def registry(self) -> SkillRegistry:
        """The underlying SkillRegistry (for admin surfaces: list_all, set_enabled)."""
        return self._registry

    # -- resolution ---------------------------------------------------------

    def _all_skill_files(self) -> list[Path]:
        return [Path(s["path"]) for s in self._registry.list_skills()]

    def resolve(self, name: str) -> Path | None:
        """Find an enabled skill by frontmatter name or directory name."""
        entry = self._registry.resolve(name)
        if entry is None:
            return None
        return Path(self._registry.store.local_path(entry.key))

    # -- listing / loading --------------------------------------------------

    def list_skills(self) -> list[dict]:
        """Enabled skills as [{name, description, path}] — metadata only, no bodies."""
        return self._registry.list_skills()

    def list_all(self) -> list[dict]:
        """Admin view: every skill plus its enabled state."""
        return self._registry.list_all()

    def render_index(self) -> str:
        """System-prompt block: names + descriptions ONLY (progressive disclosure)."""
        return self._registry.render_index()

    def load_skill(self, name: str) -> str:
        """Full SKILL.md content — the progressive-disclosure second step."""
        return self._registry.load_skill(name)

    # -- curation -----------------------------------------------------------

    def create_skill(self, name: str, description: str, content: str, category: str | None = None) -> str:
        return self._registry.create_skill(name, description, content, category)

    def patch_skill(self, name: str, old_text: str, new_text: str) -> str:
        """Targeted update (the preferred action): replace one exact string."""
        return self._registry.patch_skill(name, old_text, new_text)

    def edit_skill(self, name: str, content: str) -> str:
        """Full body rewrite (keep frontmatter). Discouraged vs patch."""
        return self._registry.edit_skill(name, content)

    def delete_skill(self, name: str) -> str:
        return self._registry.delete_skill(name)

    def write_skill_file(self, name: str, relative_path: str, content: str) -> str:
        return self._registry.write_skill_file(name, relative_path, content)

    def remove_skill_file(self, name: str, relative_path: str) -> str:
        return self._registry.remove_skill_file(name, relative_path)

    def set_enabled(self, name: str, enabled: bool) -> str:
        """Hermes-style toggle. Disabled skills vanish from the index and
        load_skill until re-enabled (takes full effect next session)."""
        return self._registry.set_enabled(name, enabled)


def create_skill_manage_tool(library: SkillLibrary):
    """Factory returning the agent-callable skill_manage tool bound to a library."""

    @tool
    def skill_manage(
        action: Literal["create", "patch", "edit", "delete", "write_file", "remove_file", "enable", "disable"],
        name: str,
        description: str = "",
        content: str = "",
        old_content: str = "",
        relative_path: str = "",
        category: str = "",
    ) -> str:
        """Create and update reusable skill files (procedural memory) under skills/.

        A skill is a SKILL.md capturing a workflow that worked, so future sessions
        can follow it instead of re-deriving it. Create one when: a task needed 5+
        tool calls, you recovered from an error, the user corrected your approach,
        or a non-obvious workflow turned out to work.

        PREFER patch over edit: it replaces one exact string (safe + token-cheap),
        while edit rewrites the whole body and risks breaking working content.

        Args:
            action: "create" (name+description+content [+category]),
                "patch" (name+old_content+content, exact-string replacement),
                "edit" (name+content, full body rewrite),
                "delete" (name), "write_file"/"remove_file" (name+relative_path
                [+content], e.g. references/notes.md), or "enable"/"disable"
                (name) to toggle whether a skill appears in the skills index.
            name: skill slug (lowercase letters/digits/-/_).
            description: one sentence describing WHEN to use it (shown in the index).
            content: new text (create body / edit body / write_file contents /
                patch replacement).
            old_content: exact existing text to replace (patch only).
            relative_path: path inside the skill dir (write_file/remove_file).
            category: optional group directory for create (e.g. "devops").
        """
        if action == "create":
            return library.create_skill(name, description, content, category or None)
        if action == "patch":
            return library.patch_skill(name, old_content, content)
        if action == "edit":
            return library.edit_skill(name, content)
        if action == "delete":
            return library.delete_skill(name)
        if action == "write_file":
            return library.write_skill_file(name, relative_path, content)
        if action == "remove_file":
            return library.remove_skill_file(name, relative_path)
        if action in ("enable", "disable"):
            return library.set_enabled(name, action == "enable")
        return f"Rejected: unknown action {action!r}"

    return skill_manage


def create_load_skill_tool(library: SkillLibrary):
    """Factory returning the load_skill tool — progressive disclosure step 2."""

    @tool
    def load_skill(name: str) -> str:
        """Load the FULL instructions of a skill from the skills index.

        Call this BEFORE doing a task when a listed skill matches it, then follow
        the loaded instructions with your regular tools. The index only shows
        names/descriptions; this returns the complete SKILL.md (steps, tool calls,
        references to other files inside the skill).

        Args:
            name: skill name from the skills index.
        """
        return library.load_skill(name)

    return load_skill
