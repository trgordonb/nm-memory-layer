"""Tests for the wiki layer (WikiStore, pre-turn context, wiki_search tool)."""

import os

import pytest

from nm_memory_layer import WikiStore, create_wiki_search_tool

PAGES = {
    "index.md": "---\ntitle: NM-Agent-CLI\n---\nSections:\n- raw/ drop zone\n",
    "notes/low-volume-mean-reversion-edge.md": (
        "---\ntitle: Low-Volume Mean-Reversion Edge\n---\n"
        "# Low-volume mean reversion\n\nBelow a volume threshold, mean-reversion strategies"
        " regain their edge; volume expansion kills it. Verified: 1200 bars of SSO data.\n"
    ),
    "notes/static-thresholds-beat-dynamic.md": (
        "# Static Thresholds\n\nStatic thresholds beat adaptive bands in backtests"
        " because they fail fast and stay interpretable.\n"
    ),
    "raw/benford-law-strategy-selection.md": "## Benford law\n\nAn attempt to use Benford's law for strategy selection hit a dead end.\n",
}


@pytest.fixture()
def wiki(tmp_path):
    d = tmp_path / "llm-wiki"
    for rel, text in PAGES.items():
        target = d / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
    return WikiStore(wiki_dir=str(d))


class TestAvailable:
    def test_available_with_pages(self, wiki):
        assert wiki.available() and wiki.search("anything") is not None

    def test_missing_dir_is_graceful(self, tmp_path):
        store = WikiStore(wiki_dir=str(tmp_path / "nope"))
        assert not store.available()
        assert store.search("anything") == []
        assert store.build_context("anything") == ""


class TestSearch:
    def test_ranks_matching_pages(self, wiki):
        hits = wiki.search("mean reversion low volume")
        assert hits, "expected matches"
        assert hits[0]["path"].endswith("low-volume-mean-reversion-edge.md")

    def test_matches_raw_sources_too(self, wiki):
        hits = wiki.search("benford law strategy selection")
        assert any(h["path"].endswith("benford-law-strategy-selection.md") for h in hits)

    def test_title_falls_back_to_filename(self, wiki):
        hits = wiki.search("static thresholds interpretable")
        assert hits and hits[0]["title"] == "static-thresholds-beat-dynamic"

    def test_frontmatter_title_used_when_present(self, wiki):
        hits = wiki.search("mean reversion volume edge")
        assert any(h["title"] == "Low-Volume Mean-Reversion Edge" for h in hits)

    def test_index_md_excluded(self, wiki):
        for hit in wiki.search("sections concepts NM-Agent-CLI index", limit=10):
            assert "index.md" not in hit["path"]

    def test_no_match_empty(self, wiki):
        assert wiki.search("quantum chromodynamics") == []

    def test_limit(self, wiki):
        assert len(wiki.search("thresholds reversion static interpretation benford", limit=1)) == 1


class TestBuildContext:
    def test_block_format(self, wiki):
        block = wiki.build_context("low volume mean reversion edge")
        assert block.startswith("<wiki_context")
        assert f'query="low volume mean reversion edge"' in block
        assert "<page path=" in block and "</wiki_context>" in block
        assert "low-volume-mean-reversion-edge.md" in block

    def test_no_match_renders_empty(self, wiki):
        assert wiki.build_context("quantum chromodynamics") == ""

    def test_nonexistent_wiki_renders_empty(self, tmp_path):
        store = WikiStore(wiki_dir=str(tmp_path / "absent"))
        assert store.build_context("anything") == ""


class TestTool:
    def test_tool_reports_matches_with_path(self, wiki):
        tool = __import__("nm_memory_layer.wiki", fromlist=["create_wiki_search_tool"]).create_wiki_search_tool(wiki)
        out = tool.invoke({"query": "mean reversion"})
        assert "low-volume-mean-reversion-edge.md" in out
        assert "read_file" in out  # pointer to progressive disclosure

    def test_tool_no_match_message(self, wiki):
        tool = __import__("nm_memory_layer.wiki", fromlist=["create_wiki_search_tool"]).create_wiki_search_tool(wiki)
        assert tool.invoke({"query": "quantum chromodynamics"}) == "No wiki matches found."

    def test_tool_without_wiki(self, tmp_path):
        from nm_memory_layer.wiki import create_wiki_search_tool

        absent = WikiStore(wiki_dir=str(tmp_path / "void"))
        assert create_wiki_search_tool(absent).invoke({"query": "x"}) == "No wiki available."


class TestGraphWalk:
    def test_graph_absent_yields_no_graph_tags(self, tmp_path):
        from nm_memory_layer.wiki import graph_hops
        d = tmp_path / "w"
        d.mkdir()
        assert graph_hops(str(d), ["pages/whatever.md"], max_hops=1) == []

    def test_graph_hops_on_skill_schema(self, tmp_path):
        """Mini graph.sqlite mirroring the skill's nodes/edges schema: a
        1-hop authored-walk expands the concept page as a graph retrieval."""
        import sqlite3
        from nm_memory_layer.wiki import graph_hops

        d = tmp_path / "w"
        (d / "graph").mkdir(parents=True)
        (d / "notes").mkdir()
        (d / "notes" / "fx.md").write_text("# FX\n\ncarry and momentum notes\n")
        conn = sqlite3.connect(d / "graph" / "graph.sqlite")
        conn.executescript(
            """
            CREATE TABLE nodes (id TEXT PRIMARY KEY, slug TEXT UNIQUE, title TEXT,
                                page_type TEXT, path TEXT, created TEXT, updated TEXT,
                                metadata_json TEXT);
            CREATE TABLE edges (id TEXT PRIMARY KEY, subject TEXT, predicate TEXT,
                                object TEXT, confidence TEXT, status TEXT,
                                extraction_method TEXT, page TEXT);
            INSERT INTO nodes VALUES ('concept:fx', 'fx', 'FX', 'concept', 'notes/fx.md', '', '', '{}');
            INSERT INTO nodes VALUES ('person:me', 'me', 'Me', 'person', '', '', '', '{}');
            INSERT INTO edges VALUES ('e1', 'person:me', 'authored', 'concept:fx', 'high', 'current',
                                      'explicit_graph_frontmatter', 'notes/fx.md');
            """
        )
        conn.commit()
        conn.close()

        hops = graph_hops(str(d), ["notes/fx.md"], max_hops=1)
        assert hops, "expected at least one graph neighbor"
        person_row = next(r for r in hops if r["path"] == "")
        assert person_row["hop"] == 1

    def test_search_tags_graph_neighbors(self, tmp_path):
        import sqlite3
        from nm_memory_layer.wiki import WikiStore

        d = tmp_path / "w"
        (d / "graph").mkdir(parents=True)
        (d / "notes").mkdir()
        (d / "notes" / "fx.md").write_text("# FX\n\ncarry and momentum notes\n")
        (d / "notes" / "adjacent.md").write_text("# Adjacent\n\nlinked from fx via mentions\n")
        conn = sqlite3.connect(d / "graph" / "graph.sqlite")
        conn.executescript(
            """
            CREATE TABLE nodes (id TEXT PRIMARY KEY, slug TEXT UNIQUE, title TEXT,
                                page_type TEXT, path TEXT, created TEXT, updated TEXT,
                                metadata_json TEXT);
            CREATE TABLE edges (id TEXT PRIMARY KEY, subject TEXT, predicate TEXT,
                                object TEXT, confidence TEXT, status TEXT,
                                extraction_method TEXT, page TEXT);
            INSERT INTO nodes VALUES ('concept:fx', 'fx', 'FX', 'concept', 'notes/fx.md', '', '', '{}');
            INSERT INTO nodes VALUES ('concept:adjacent', 'adjacent', 'Adjacent', 'concept', 'notes/adjacent.md', '', '', '{}');
            INSERT INTO edges VALUES ('e1', 'concept:fx', 'mentions', 'concept:adjacent', 'low', 'current',
                                      'body_wikilink', 'notes/fx.md');
            """
        )
        conn.commit()
        conn.close()

        wiki = WikiStore(wiki_dir=str(d))
        hits = wiki.search("carry momentum", limit=3)
        path_terms = [h["path"] for h in hits]
        assert "notes/adjacent.md" in path_terms, "graph-only neighbor must be included"
        tagged = next(h for h in hits if h["path"] == "notes/adjacent.md")
        assert tagged["retrievers"] == ["graph"]
