"""Wiki recall: the local llm-wiki (OKF layout) as a pre-flight context layer.

The OKF wiki layout is a directory of markdown pages grouped in subfolders
(``concepts/``, ``entities/``, ``notes/``, ``raw/``) plus ``index.md`` files:

    llm-wiki/
    ├── index.md
    ├── notes/*.md          # distilled pages, one idea per file
    ├── concepts/…          # concept pages
    ├── raw/                # immutable source documents (searchable)
    └── ...

Two capabilities:

1. **Pre-flight injection** (consumer calls :meth:`WikiStore.build_context`
   with the user's message before each turn): matched pages render as a
   ``<wiki_context>`` block for the system prompt so recorded knowledge is
   consulted even when the user doesn't mention the wiki. Empty string when
   the wiki is absent or nothing matches — never errors in either case.
2. **Deliberate retrieval**: the ``wiki_search`` tool lets the agent pull
   further excerpts on demand (same FTS-style ranking as session_search).

Matching is plain SQLite-free token-overlap scoring over titles + bodies
(wikis are small; no index file to build). Frontmatter is skipped; title
comes from the ``title:`` frontmatter field or the filename.
"""

import os
import re
from dataclasses import dataclass

from langchain_core.tools import tool

from .summarizer import _env_bool  # shared env helpers

DEFAULT_WIKI_DIR = "llm-wiki"

_TITLE_RE = re.compile(r"^title:\s*(.+?)\s*$", re.MULTILINE)
_FRONTMATTER_RE = re.compile(r"\A---\n.*?\n---\n?", re.DOTALL)
_TOKEN_RE = re.compile(r"[a-z]{3,}")

_STOPWORDS = {
    "the", "and", "for", "that", "this", "with", "from", "have", "was", "are",
    "were", "but", "not", "you", "your", "what", "how", "did", "does", "can",
    "about", "there", "their", "when", "which", "into", "out", "use", "using",
    "get", "got", "let", "our", "its", "them", "then", "than", "more", "most",
    "asked", "tell", "want", "need",
}


class WikiStore:
    """Searches the local llm-wiki directory (OKF layout)."""

    def __init__(self, wiki_dir: str | None = None):
        self.wiki_dir = os.path.realpath(wiki_dir or os.getenv("WIKI_DIR", DEFAULT_WIKI_DIR))

    @staticmethod
    def _parse_page(path: str, text: str) -> tuple[str, str, str]:
        """Return (title, body_tokens, body) — frontmatter stripped."""
        body = _FRONTMATTER_RE.sub("", text, count=1) if text.startswith("---") else text
        title_match = _TITLE_RE.search(text[:400])
        title = title_match.group(1).strip() if title_match else os.path.splitext(os.path.basename(path))[0]
        return title, body.strip(), _TOKEN_RE.findall(body.lower())
    def _pages(self) -> list[tuple[str, str]]:
        """[(relative posix path, raw text)] for all wiki pages except index.md."""
        pages = []
        for root, _dirs, files in os.walk(self.wiki_dir):
            for name in files:
                if not name.endswith(".md") or name == "index.md":
                    continue
                full_path = os.path.join(root, name)
                try:
                    pages.append((os.path.relpath(full_path, self.wiki_dir), open(full_path, encoding="utf-8", errors="replace").read()))
                except OSError:
                    continue
        return sorted(pages)

    def search(self, query: str, limit: int = 3, max_excerpt_chars: int = 1400) -> list[dict]:
        """Rank wiki pages by token overlap with the query.

        Returns [{path, title, excerpt, score}], best first. Empty list when
        the wiki is missing or nothing matches.
        """
        if not self.available():
            return []
        tokens = [t for t in _TOKEN_RE.findall(query.lower()) if t not in _STOPWORDS]
        if not tokens:
            return []

        hits = []
        for path, text in self._pages():
            title_match = _TITLE_RE.search(text[:400])
            title = title_match.group(1).strip() if title_match else os.path.splitext(os.path.basename(path))[0]
            _title, body, body_tokens = self._parse_page(path, text)
            lowered = (body + " " + path + " " + title).lower()
            body_tokens = set(body.lower().split())
            title_hits = sum(1 for t in tokens if t in title.lower() or t in os.path.basename(path).replace("-", " ").lower())
            body_hits = sum(1 for t in tokens if t in lowered)
            score = title_hits * 3 + body_hits
            if score <= 0:
                continue
            excerpt = body[:max_excerpt_chars] + (" …" if len(body) > max_excerpt_chars else "")
            hits.append({"path": path, "title": title, "excerpt": excerpt, "score": score})

        hits.sort(key=lambda h: h["score"], reverse=True)
        return hits[:limit]

    def available(self) -> bool:
        return bool(self._pages())

    def build_context(self, query: str, limit: int = 3) -> str:
        """Render the pre-turn ``<wiki_context>`` block ("" when nothing matches)."""
        hits = self.search(query, limit=limit)
        if not hits:
            return ""
        import time
        lines = [
            f'<wiki_context query="{query[:120]}" pages="{len(hits)}">',
            "Matched pages from the local llm-wiki. Cite them as known recorded"
            " knowledge; load a full page (e.g. read_file(llm-wiki/<path>)) before"
            " relying on details. Update the wiki via the llm-wiki-okf skill.",
        ]
        for hit in hits:
            lines.append(f'\n<page path="{hit["path"]}" title="{hit["title"]}">')
            lines.append(hit["excerpt"].strip())
            lines.append("</page>")
        lines.append("</wiki_context>")
        return "\n".join(lines)



def create_wiki_search_tool(wiki: WikiStore):
    """Agent-initiated deliberate wiki retrieval (progressive disclosure)."""

    @tool
    def wiki_search(query: str, limit: int = 5) -> str:
        """Search the local knowledge base (llm-wiki) for documented facts,
        decisions, and concepts. Use when the answer might already be recorded
        in the wiki — prior decisions, distilled rules, domain notes.

        Args:
            query: keywords to search for.
            limit: max pages to return (default 5).
        """
        if not wiki.available():
            return "No wiki available."
        hits = wiki.search(query, limit=limit)
        if not hits:
            return "No wiki matches found."
        blocks = []
        for hit in hits:
            blocks.append(f"--- {hit['path']} ({hit['title']}) ---\n{hit['excerpt']}")
        return "\n\n".join(blocks) + "\n\n(Full text: read_file(llm-wiki/<path>). Update via the llm-wiki-okf skill.)"

    return wiki_search
