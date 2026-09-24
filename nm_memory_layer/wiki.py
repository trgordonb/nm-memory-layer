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

import logging
import os
import re
from dataclasses import dataclass

from langchain_core.tools import tool

from .summarizer import _env_bool  # shared env helpers

DEFAULT_WIKI_DIR = "llm-wiki"
MAX_COSINE_DISTANCE = 0.35  # skill parity
RRF_K = 60  # skill parity (reciprocal-rank fusion constant)

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

    def _token_overlap(self, query: str, limit: int = 3, max_excerpt_chars: int = 1400) -> list[dict]:
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

    def search(self, query: str, limit: int = 3, max_excerpt_chars: int = 1400) -> list[dict]:
        """Hybrid when the wiki's vector cache + graph deps are available
        (skill-compatible RRF fusion); otherwise the pure token-overlap
        ranker. Falls back silently either way. After ranking, a 1-hop graph
        walk appends structurally linked pages (retriever tag "graph")."""
        hybrid = _hybrid_search(self, query, limit)
        if hybrid is not None:
            merged = list(hybrid)
        else:
            overlap = self._token_overlap(query, limit=limit)
            for hit in overlap:
                hit["retrievers"] = ["lexical"]
                hit["excerpt"] = hit.get("excerpt") or hit.get("snippet", "")
            merged = overlap

        # Graph walk seeded by matched pages: append structurally linked pages
        # the text rankers could not see (e.g. authored works, depends_on
        # companions), one hop out, path-deduped, excerpts filled from disk.
        if _graph_available(self.wiki_dir):
            seed_paths = []
            for hit in merged:
                if hit["path"] not in seed_paths:
                    seed_paths.append(hit["path"])
            existing_paths = {h["path"] for h in merged}
            for neighbor in graph_hops(self.wiki_dir, seed_paths, max_hops=1):
                if neighbor["path"] in existing_paths:
                    continue
                existing_paths.add(neighbor["path"])
                text = next((t for p, t in self._pages() if p == neighbor["path"]), "")
                if not neighbor["path"]:
                    continue  # graph-only nodes carry no page to cite
                neighbor["excerpt"] = text[:1400]
                neighbor["title"] = neighbor["title"] or neighbor["path"]
                neighbor["score"] = round(0.02 + 0.001 * len(merged), 4)
                merged.append(neighbor)

        return merged[: max(limit, 8)]

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
            "Matched pages from the local llm-wiki (graph-walked neighbors tagged as such). Cite them as known recorded"
            f" knowledge; load a full page (e.g. read_file({os.path.basename(self.wiki_dir)}/<path>))"
            " before relying on details. Update the wiki via the llm-wiki skill.",
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
        return "\n\n".join(blocks) + f"\n\n(Full text: read_file({os.path.basename(wiki.wiki_dir)}/<path>). Update via the llm-wiki skill.)"

    return wiki_search


def _heading_sections(wiki_dir: str):
    """Split every wiki page into sections exactly like the skill's
    wiki_search.py: line-level ATX headings outside code fences (fence-aware),
    flush the preamble before the first heading, heading-stack tracking, and a
    locator "<rel_path>\\x1f<section_index>" per section (the \\x1f separator
    matches semantic_sections rows already stored in the vector index)."""
    HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")

    def _split_page(rel_path: str, body: str) -> list[dict]:
        sections = []
        heading_stack: list[tuple[int, str]] = []
        current_path: list[str] = []
        current_level = 0
        lines: list[str] = []
        in_fence = False

        def append_section() -> None:
            sections.append({
                "text": "\n".join(lines),
                "section_index": len(sections),
                "heading_path": list(current_path),
            })

        for line in body.splitlines():
            stripped = line.strip()
            if stripped.startswith(("```", "~~~")):
                in_fence = not in_fence
                lines.append(line)
                continue
            heading = None if in_fence else HEADING_RE.match(line)
            if not heading:
                lines.append(line)
                continue
            append_section()
            current_level = len(heading.group(1))
            heading_text = heading.group(2).strip()
            while heading_stack and heading_stack[-1][0] >= current_level:
                heading_stack.pop()
            heading_stack.append((current_level, heading_text))
            current_path = [t for _, t in heading_stack]
            lines = []

        append_section()
        return sections

    # Template/meta files (SCHEMA.md, log.md, .evolution/*) are not knowledge
    # pages — the vector index never embeds them; mirror that filter.
    pages = [(p, t) for p, t in WikiStore(wiki_dir)._pages()
             if not p.endswith("SCHEMA.md") and p not in ("log.md", ".evolution/README.md",
                                                          ".page-template.md", ".pattern-template.md",
                                                          "graph/README.md")]
    out = []
    for rel_path, text in pages:
        body = text
        if body.startswith("---"):
            body = "---".join(body.split("---")[2:])
        for chunk in _split_page(rel_path, body):
            out.append({
                "locator": f"{rel_path}\x1f{chunk['section_index']}",
                "rel_path": rel_path,
                "text": chunk["text"],
                "heading_path": chunk["heading_path"],
            })
    return out

def _hybrid_search(self, query: str, limit: int):
    """True hybrid semantic+lexical retrieval, skill-compatible.

    Returns a result list on success; returns None whenever hybrid is
    unavailable (missing exports, missing cache) so callers fall back to the
    token-overlap ranker. Same fusion as wiki_search.py: RRF over
    lexical ranks + vec0 KNN ranks (MAX_COSINE_DISTANCE 0.35), max 2 sections
    per page, newest-relevance ordering.
    """
    backend = _load_hybrid_backend()
    if backend is None:
        return None
    model, sqlite_vec = backend

    import sqlite3

    index_path = os.path.join(self.wiki_dir, ".wiki-cache", "embeddings.sqlite")
    if not os.path.isfile(index_path):
        return None

    sections = _heading_sections(self.wiki_dir)
    if not sections:
        return None
    locator_to_frag_index = {s["locator"]: i for i, s in enumerate(sections)}

    connection = sqlite3.connect(f"file:{index_path}?mode=ro", uri=True)
    embedding_ranked: list[str] = []
    try:
        connection.enable_load_extension(True)
        sqlite_vec.load(connection)
        connection.enable_load_extension(False)
        locator_ids = dict(connection.execute("SELECT locator, id FROM semantic_sections"))
        query_vector = next(model.query_embed([query]))
        query_blob = sqlite_vec.serialize_float32(query_vector)
        k_rows = min(50, len(locator_ids))
        ranked_rows = connection.execute(
            "SELECT rowid, distance FROM semantic_vectors "
            "WHERE embedding MATCH ? AND k = ? ORDER BY distance",
            (query_blob, k_rows),
        ).fetchall()
    except Exception as exc:
        logging.warning("wiki hybrid search failed (%s) — falling back to token-overlap", exc)
        return None
    finally:
        try:
            connection.close()
        except Exception:
            pass

    for row_id, distance in ranked_rows:
        if distance > MAX_COSINE_DISTANCE:
            break  # results are distance-ordered
        for locator, sid in locator_ids.items():
            if sid == row_id:
                embedding_ranked.append(locator)
                break
    embedding_ranks = {
        locator_to_frag_index[locator]: rank
        for rank, locator in enumerate(embedding_ranked)
        if locator in locator_to_frag_index
    }

    # lexical: term-overlap over section texts (BM25 stand-in; the vec fusion
    # tolerates either ranker — what matters is stable rank ordering)
    tokens = [t for t in _TOKEN_RE.findall(query.lower()) if t not in _STOPWORDS]
    lexical_scores = []
    for frag_index, section in enumerate(sections):
        text_lower = section["text"].lower()
        hits = sum(1 for t in tokens if t in text_lower)
        if hits:
            lexical_scores.append((hits, frag_index))
    lexical_scores.sort(key=lambda item: -item[0])
    lexical_rank_list = [frag_index for _, frag_index in lexical_scores[:50]]
    lexical_ranks = {frag_index: rank for rank, frag_index in enumerate(lexical_rank_list, 1)}

    # RRF candidate merge (skill order: lexical candidates first, then
    # embedding-only candidates appended)
    candidate_order = list(lexical_rank_list)
    seen_candidates = set(candidate_order)
    for locator in embedding_ranked:
        frag_index = locator_to_frag_index.get(locator)
        if frag_index is not None and frag_index not in seen_candidates:
            candidate_order.append(frag_index)
            seen_candidates.add(frag_index)

    fused = []
    for frag_index in candidate_order:
        score = 0.0
        retrievers = []
        if frag_index in lexical_ranks:
            score += 1.0 / (RRF_K + lexical_ranks[frag_index])
            retrievers.append("lexical")
        if frag_index in embedding_ranks:
            score += 1.0 / (RRF_K + embedding_ranks[frag_index])
            retrievers.append("embedding")
        fused.append((score, frag_index, retrievers))
    fused.sort(key=lambda item: -item[0])
    page_counts: dict[str, int] = {}
    results = []
    for score, frag_index, retrievers in fused:
        section = sections[frag_index]
        page_counts[section["rel_path"]] = page_counts.get(section["rel_path"], 0) + 1
        if page_counts[section["rel_path"]] > 2:
            continue
        results.append({
            "path": section["rel_path"],
            "title": os.path.splitext(os.path.basename(section["rel_path"]))[0],
            "excerpt": section["text"][:1400],
            "score": round(score, 4),
            "retrievers": retrievers,
        })
        if len(results) >= limit * 2:
            break
    # only report hybrid when semantic evidence actually contributed
    if not any("embedding" in r["retrievers"] for r in results):
        return None
    return results


def _load_hybrid_backend():
    """(model, sqlite_vec) or None — mirrors the skill's loader.

    Same model name and cache dir (FASTEMBED_CACHE_PATH, default
    ~/.cache/llm-wiki/fastembed) so the memory layer shares the wiki
    tooling's on-disk model without double downloads.
    """
    try:
        import sqlite_vec
        from fastembed import TextEmbedding
    except ImportError:
        return None

    cache_dir = os.environ.get(
        "FASTEMBED_CACHE_PATH",
        os.path.expanduser(os.path.join("~", ".cache", "llm-wiki", "fastembed")),
    )
    try:
        model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5", cache_dir=str(cache_dir))
    except Exception as exc:
        logging.warning("wiki hybrid search: fastembed unavailable (%s)", exc)
        return None
    return model, sqlite_vec


# ── Graph hop walking (wiki/graph/graph.sqlite) ──────────────────────────────
#
# The llm-wiki skill maintains a typed knowledge graph (nodes/aliases/edges).
# Search surfaces TEXT matches; the graph answers a complementary question —
# which recorded nodes are structurally linked to what matched. A 1-2 hop walk
# from a hybrid hit's page node (authored / works_on / depends_on /
# mentions / summarised) surfaces neighbors the embeddings cannot see:
# e.g. "Artur Sepp" → authored → their works → mentions → concepts.

GRAPH_PREDICATE_ORDER = [
    "authored", "works_on", "depends_on", "summarizes_raw", "sourced_from",
    "mentions",
]


def _graph_available(wiki_dir: str) -> bool:
    return os.path.isfile(os.path.join(wiki_dir, "graph", "graph.sqlite"))


def _graph_store(wiki_dir: str):
    """Read-only connection to the wiki graph (None when absent)."""
    import sqlite3

    if not _graph_available(wiki_dir):
        return None
    try:
        conn = sqlite3.connect(f"file:{os.path.join(wiki_dir, 'graph', 'graph.sqlite')}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        return conn
    except Exception as exc:
        logging.warning("wiki graph: connection failed (%s)", exc)
        return None


def _locator_to_path(locator: str) -> str:
    """Locator '<rel_path>\\x1f<idx>' -> page rel_path (pre-flight key form)."""
    return locator.split("\x1f", 1)[0]


def graph_hops(wiki_dir: str, seed_paths: list[str], max_hops: int = 1,
               max_nodes: int = 6) -> list[dict]:
    """Walk the wiki graph 1..2 hops outward from pages that matched a query.

    Seeds are page relative paths (the hybrid hits). Returns one row per
    REACHED page/node: {node_id, title, page_type, path, edge_chain}, ranked by
    predicate weight. Graph-absent -> [].
    """
    conn = _graph_store(wiki_dir)
    if conn is None or not seed_paths:
        return []
    try:
        seed_by_path: dict[str, str] = {}
        for row in conn.execute("SELECT id, path FROM nodes"):
            seed_by_path.setdefault(row["path"], row["id"])

        visited_nodes: set[str] = set()
        results: list[dict] = []
        frontier_paths = seed_paths
        for hop in range(max_hops):
            next_frontier: list[str] = []
            for page_path in frontier_paths:
                node_id = seed_by_path.get(page_path) or f"page:{page_path}"
                if node_id in visited_nodes:
                    continue
                visited_nodes.add(node_id)
                # outgoing + incoming edges of this node's DIRECT subject/object
                edges = conn.execute(
                    """SELECT e.predicate, e.subject, e.object, e.confidence,
                              e.extraction_method, e.page
                       FROM edges e
                       WHERE e.subject = ? OR e.object = ?""",
                    (node_id, node_id),
                ).fetchall()
                for edge in edges:
                    for endpoint in (edge["subject"], edge["object"]):
                        if endpoint == node_id or endpoint in visited_nodes:
                            continue
                        node = conn.execute(
                            "SELECT id, slug, title, page_type, path FROM nodes WHERE id = ?",
                            (endpoint,),
                        ).fetchone()
                        if not node:
                            continue  # aliases/raw nodes may have no page
                        if hop == 0 and not edge["predicate"]:
                            continue
                        results.append({
                            "path": node["path"],
                            "title": node["title"] or node["slug"],
                            "excerpt": None,  # filled by caller from the page text
                            "score": 0.0,
                            "retrievers": ["graph"],
                            "hop": hop + 1,
                            "via": f"{edge['predicate']} ({edge['extraction_method']}) from",
                            "anchor": page_path,
                        })
                        next_frontier.append(node["path"])
            frontier_paths = next_frontier
        conn.close()
        return results
    except Exception as exc:
        import logging
        logging.warning("wiki graph walk failed (%s) — skipping", exc)
        return []

