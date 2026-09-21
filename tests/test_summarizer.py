"""Tests for the secondary-LLM session-search summarizer (env-toggled)."""

import json

import pytest

from nm_memory_layer import SearchSummarizer, SessionStore, create_openrouter_summarizer, create_session_search_tool
from nm_memory_layer.summarizer import _env_bool, SummarizedSearch
from langchain_core.messages import HumanMessage


@pytest.fixture()
def store(tmp_path):
    s = SessionStore(db_path=str(tmp_path / "sessions.db"))
    sid = s.new_session_id()
    s.record_turn(sid, [
        HumanMessage(content="Backfilled EURUSD via the S3 fallback bucket after 429 throttling"),
        HumanMessage(content="RSI values live in workspace/RSI-EURUSD.csv"),
    ])
    yield s
    s.close()


class FakeSummarizerModel:
    """Sync .invoke model stand-in; scripted responses."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.model_name = "fake-summarizer"

    def invoke(self, messages):
        self.calls.append(messages)
        return self.responses.pop(0)


class Resp:
    def __init__(self, content):
        self.content = content


class TestFactory:
    def test_disabled_by_default(self, monkeypatch):
        for var in ("SEARCH_SUMMARIZER_ENABLED", "OPENROUTER_API_KEY", "OPENROUTER_MODEL", "OPENROUTER_BASE_URL"):
            monkeypatch.delenv(var, raising=False)
        assert create_openrouter_summarizer() is None

    def test_enabled_but_unconfigured_returns_none(self, monkeypatch):
        monkeypatch.setenv("SEARCH_SUMMARIZER_ENABLED", "true")
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.delenv("OPENROUTER_MODEL", raising=False)
        assert create_openrouter_summarizer() is None

    def test_enabled_and_configured_builds_summarizer(self, monkeypatch):
        monkeypatch.setenv("SEARCH_SUMMARIZER_ENABLED", "TRUE")
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
        monkeypatch.setenv("OPENROUTER_MODEL", "test/fast-model")
        monkeypatch.setenv("OPENROUTER_BASE_URL", "https://example.invalid/v1")
        s = create_openrouter_summarizer()
        assert isinstance(s, SearchSummarizer)
        assert s.label == "openrouter:test/fast-model"

    def test_env_bool_variants(self, monkeypatch):
        for v in ("1", "true", "YES", "on", " On "):
            monkeypatch.setenv("SEARCH_SUMMARIZER_ENABLED", v)
            assert _env_bool("SEARCH_SUMMARIZER_ENABLED")
        for v in ("0", "false", "", "off"):
            monkeypatch.setenv("SEARCH_SUMMARIZER_ENABLED", v)
            assert not _env_bool("SEARCH_SUMMARIZER_ENABLED")


class TestToolWithSummarizer:
    def test_summary_returned_with_mode_header(self, store):
        model = FakeSummarizerModel([Resp("EURUSD was backfilled via S3 fallback [1]. RSI at workspace/RSI-EURUSD.csv [2].")])
        summarizer = SearchSummarizer(model, label="test:model")
        tool = create_session_search_tool(store, summarizer=summarizer)
        result = tool.invoke({"query": "eurusd backfill"})
        assert result.startswith("[session_search: condensed by test:model in")
        assert "S3 fallback" in result
        assert "[1]" in result
        # summarizer received the query and the excerpts
        assert len(model.calls) == 1
        assert "eurusd backfill" in model.calls[0][1][1]

    def test_summarizer_gets_more_raw_excerpts_than_limit(self, store):
        sid = store.new_session_id()
        for i in range(10):
            store.record_turn(sid, [HumanMessage(content=f"fibonacci note {i} with plenty of text to search")])
        captured = {}

        def spy_summarize(query, hits):
            captured["n"] = len(hits)
            return SummarizedSearch(text="condensed", model_label="t", elapsed_ms=1, excerpt_count=len(hits))

        summarizer = SearchSummarizer(FakeSummarizerModel([]))
        summarizer.summarize = spy_summarize
        tool = create_session_search_tool(store, summarizer=summarizer)
        result = tool.invoke({"query": "fibonacci", "limit": 3})
        assert captured["n"] >= 8  # widened material, not the agent's limit of 3
        assert result.startswith("[session_search: condensed")

    def test_summarizer_crash_falls_back_to_raw(self, store):
        def boom(query, hits):
            raise RuntimeError("provider down")

        summarizer = SearchSummarizer(FakeSummarizerModel([]))
        summarizer.summarize = boom
        tool = create_session_search_tool(store, summarizer=summarizer)
        result = tool.invoke({"query": "eurusd backfill"})
        assert not result.startswith("[session_search: condensed")
        assert "Backfilled EURUSD" in result  # raw excerpts returned

    def test_explicit_none_relevant_becomes_no_relevance(self, store):
        summarizer = SearchSummarizer(FakeSummarizerModel([Resp("none relevant")]))
        tool = create_session_search_tool(store, summarizer=summarizer)
        # query hits the archive, but the summarizer explicitly judges nothing relevant
        result = tool.invoke({"query": "RSI csv"})
        assert result == "No past session matches relevant to this query."

    def test_empty_summary_falls_back_to_raw(self, store):
        """Empty model content is a provider quirk (e.g. reasoning budget exhausted
        on OpenRouter), NOT a relevance judgment — must fall back to raw excerpts."""
        summarizer = SearchSummarizer(FakeSummarizerModel([Resp("")]))
        tool = create_session_search_tool(store, summarizer=summarizer)
        result = tool.invoke({"query": "RSI csv"})
        assert not result.startswith("[session_search")
        assert "RSI values live in workspace/RSI-EURUSD.csv" in result

    def test_no_summarizer_returns_raw_excerpts(self, store):
        tool = create_session_search_tool(store)
        result = tool.invoke({"query": "eurusd backfill"})
        assert not result.startswith("[session_search")
        assert "Backfilled EURUSD" in result


class TestWeakModelRobustness:
    """Tiny/free models leak chat-template tokens and hallucinate tool calls
    instead of answering; the summarizer must degrade to raw excerpts."""

    def test_block_list_content_joined_to_text(self, store):
        model = FakeSummarizerModel([Resp([{"type": "text", "text": "condensed fine"}])])
        tool = create_session_search_tool(store, summarizer=SearchSummarizer(model, label="m"))
        result = tool.invoke({"query": "eurusd backfill"})
        assert "condensed fine" in result
        assert "'text':" not in result  # no str(list) repr garbage

    def test_special_tokens_stripped(self, store):
        model = FakeSummarizerModel([Resp("good stuff <|fim_pad|> more good stuff")])
        tool = create_session_search_tool(store, summarizer=SearchSummarizer(model, label="m"))
        result = tool.invoke({"query": "eurusd backfill"})
        assert "good stuff  more good stuff" in result
        assert "<|" not in result

    def test_pseudo_tool_call_payload_falls_back_to_raw(self, store):
        garbage = (
            "<|tool_call_start|>[summarize(excerpts=['[1] session=a76517b8',"
            " 'Output: workspace containing RSI scripts'])]<|tool_call_end|>"
        )
        model = FakeSummarizerModel([Resp([{"type": "text", "text": garbage}])])
        tool = create_session_search_tool(store, summarizer=SearchSummarizer(model, label="m"))
        result = tool.invoke({"query": "eurusd backfill"})
        assert not result.startswith("[session_search")
        assert "Backfilled EURUSD" in result  # raw excerpts, no garbage
        assert "tool_call" not in result

    def test_other_pseudo_calls_also_caught(self, store):
        model = FakeSummarizerModel([Resp("<|tool_call_start|>[answer(question='CONDENSE: NVDA')]")])
        tool = create_session_search_tool(store, summarizer=SearchSummarizer(model, label="m"))
        result = tool.invoke({"query": "eurusd backfill"})
        assert "Backfilled EURUSD" in result
