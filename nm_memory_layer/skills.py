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
remove_file — with ``patch`` as the preferred update path: a targeted
string replacement is both safer (no risk of breaking working content) and
more token-efficient than a full rewrite.

Skill-creation triggers (evaluated by the periodic nudge): a turn with many
tool calls, recovery from an error, a user correction, or a non-obvious
workflow that worked.
"""

import os
import re
import shutil
from pathlib import Path
from typing import Literal

import yaml
from langchain_core.tools import tool

_SKILL_FILE = "SKILL.md"
_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """Split a SKILL.md into (frontmatter dict, body). Missing/malformed frontmatter -> ({}, full text)."""
    if not text.startswith("---"):
        return {}, text
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text
    try:
        meta = yaml.safe_load(parts[1])
    except yaml.YAMLError:
        return {}, parts[2].lstrip("\n")
    return (meta if isinstance(meta, dict) else {}), parts[2].lstrip("\n")


class SkillLibrary:
    """Manages a directory of agentskills.io-style skills."""

    def __init__(self, skills_dir: str | None = None):
        self.skills_dir = Path(skills_dir or os.getenv("SKILLS_DIR", "skills"))

    # -- resolution ---------------------------------------------------------

    def _all_skill_files(self) -> list[Path]:
        if not self.skills_dir.exists():
            return []
        return sorted(self.skills_dir.rglob(_SKILL_FILE))

    def _read_skill(self, path: Path) -> tuple[dict, str]:
        return _parse_frontmatter(path.read_text())

    def resolve(self, name: str) -> Path | None:
        """Find a skill by frontmatter name or directory name."""
        for path in self._all_skill_files():
            meta, _ = self._read_skill(path)
            if meta.get("name") == name or path.parent.name == name:
                return path
        return None

    # -- listing / loading --------------------------------------------------

    def list_skills(self) -> list[dict]:
        """All skills as [{name, description, path}] — metadata only, no bodies."""
        skills = []
        for path in self._all_skill_files():
            meta, _ = self._read_skill(path)
            skills.append(
                {
                    "name": str(meta.get("name") or path.parent.name),
                    "description": str(meta.get("description") or ""),
                    "path": str(path),
                }
            )
        return skills

    def render_index(self) -> str:
        """System-prompt block: names + descriptions ONLY (progressive disclosure)."""
        skills = self.list_skills()
        if not skills:
            return ""
        lines = [f"- {s['name']}: {s['description']}" for s in skills]
        return "<skills_index>\n" + "\n".join(lines) + "\n</skills_index>"

    def load_skill(self, name: str) -> str:
        """Full SKILL.md content — the progressive-disclosure second step."""
        path = self.resolve(name)
        if path is None:
            return f"Rejected: no skill named {name!r}"
        return path.read_text()

    # -- curation -----------------------------------------------------------

    def _skill_dir(self, name: str, category: str | None = None) -> Path:
        base = self.skills_dir / category if category else self.skills_dir
        return base / name

    @staticmethod
    def _frontmatter(name: str, description: str) -> str:
        safe_description = description.replace('"', "'")
        return f"---\nname: {name}\ndescription: \"{safe_description}\"\nversion: 1.0.0\n---\n\n"

    def create_skill(self, name: str, description: str, content: str, category: str | None = None) -> str:
        if not _NAME_RE.match(name):
            return f"Rejected: skill name must match {_NAME_RE.pattern!r} (got {name!r})"
        if not _NAME_RE.match(category or "x"):
            return f"Rejected: category must be a slug (got {category!r})"
        skill_dir = self._skill_dir(name, category)
        if (skill_dir / _SKILL_FILE).exists():
            return f"Rejected: skill {name!r} already exists; use patch or edit"
        if not content.strip():
            return "Rejected: skill body is empty"
        skill_dir.mkdir(parents=True, exist_ok=True)
        (skill_dir / _SKILL_FILE).write_text(self._frontmatter(name, description) + content.strip() + "\n")
        return f"OK: created skill {name!r} at {skill_dir / _SKILL_FILE}"

    def patch_skill(self, name: str, old_text: str, new_text: str) -> str:
        """Targeted update (the preferred action): replace one exact string."""
        path = self.resolve(name)
        if path is None:
            return f"Rejected: no skill named {name!r}"
        text = path.read_text()
        if old_text not in text:
            return f"Rejected: old_text not found in {name}'s SKILL.md"
        path.write_text(text.replace(old_text, new_text, 1))
        return f"OK: patched {name!r}"

    def edit_skill(self, name: str, content: str) -> str:
        """Full body rewrite (keep frontmatter). Discouraged vs patch."""
        path = self.resolve(name)
        if path is None:
            return f"Rejected: no skill named {name!r}"
        meta, _ = self._read_skill(path)
        if not content.strip():
            return "Rejected: skill body is empty"
        path.write_text(self._frontmatter(str(meta.get("name") or path.parent.name), str(meta.get("description") or "")) + content.strip() + "\n")
        return f"OK: rewrote body of {name!r}"

    def delete_skill(self, name: str) -> str:
        path = self.resolve(name)
        if path is None:
            return f"Rejected: no skill named {name!r}"
        shutil.rmtree(path.parent)
        return f"OK: deleted skill {name!r}"

    def _guarded_skill_file(self, name: str, relative_path: str) -> Path | str:
        path = self.resolve(name)
        if path is None:
            return f"Rejected: no skill named {name!r}"
        rel = Path(relative_path)
        if rel.is_absolute() or ".." in rel.parts or not rel.parts:
            return f"Rejected: invalid relative path {relative_path!r}"
        return path.parent / rel

    def write_skill_file(self, name: str, relative_path: str, content: str) -> str:
        target = self._guarded_skill_file(name, relative_path)
        if isinstance(target, str):
            return target
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
        return f"OK: wrote {relative_path!r} in {name!r}"

    def remove_skill_file(self, name: str, relative_path: str) -> str:
        target = self._guarded_skill_file(name, relative_path)
        if isinstance(target, str):
            return target
        if not target.exists():
            return f"Rejected: {relative_path!r} not found in {name!r}"
        target.unlink()
        return f"OK: removed {relative_path!r} from {name!r}"


def create_skill_manage_tool(library: SkillLibrary):
    """Factory returning the agent-callable skill_manage tool bound to a library."""

    @tool
    def skill_manage(
        action: Literal["create", "patch", "edit", "delete", "write_file", "remove_file"],
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
                [+content], e.g. references/notes.md).
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
