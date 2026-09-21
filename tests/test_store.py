"""Tests for the episodic session store (SessionStore + session_search tool)."""

import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from nm_memory_layer import SessionStore, create_session_search_tool


@pytest.fixture()
def store(tmp_path):
    s = SessionStore(db_path=str(tmp_path / "sessions.db"))
    yield s
    s.close()


@pytest.fixture()
def tool(store):
    return create_session_search_tool(store)


def _sample_turn():
    return [
        HumanMessage(content="Remember: the quarterly report lives in reports/q3.md"),
        AIMessage(
            content="",
            tool_calls=[{"name": "read_file", "args": {"file_path": "reports/q3.md"}, "id": "call_1"}],
        ),
        ToolMessage(content="Q3 revenue up 12%", tool_call_id="call_1", name="read_file"),
        AIMessage(content="Noted: Q3 revenue up 12% per reports/q3.md"),
    ]


class TestRecordAndResume:
    def test_record_returns_incrementing_turn_numbers(self, store):
        sid = store.new_session_id()
        assert store.record_turn(sid, [HumanMessage(content="first")]) == 1
        assert store.record_turn(sid, [HumanMessage(content="second")]) == 2

    def test_roundtrip_reconstructs_tool_call_pairs(self, store):
        sid = store.new_session_id()
        store.record_turn(sid, _sample_turn())
        messages = store.load_session(sid)
        assert [m.__class__.__name__ for m in messages] == [
            "HumanMessage",
            "AIMessage",
            "ToolMessage",
            "AIMessage",
        ]
        call = messages[1].tool_calls[0]
        assert call["name"] == "read_file"
        assert messages[2].tool_call_id == call["id"] == "call_1"
        assert messages[2].name == "read_file"
        assert "Noted: Q3 revenue" in str(messages[3].content)

    def test_ai_message_without_calls_or_content_is_skipped(self, store):
        sid = store.new_session_id()
        turn = store.record_turn(sid, [HumanMessage(content="hi"), AIMessage(content=""), AIMessage(content="visible")])
        assert turn == 1
        assert len(store.load_session(sid)) == 2

    def test_empty_message_list_records_nothing(self, store):
        sid = store.new_session_id()
        assert store.record_turn(sid, []) == 0
        assert store.load_session(sid) == []

    def test_resume_is_provider_safe_format(self, store):
        """Reconstructed AIMessage tool_calls must carry name/args/id keys."""
        sid = store.new_session_id()
        store.record_turn(sid, _sample_turn())
        call = store.load_session(sid)[1].tool_calls[0]
        assert {"name", "args", "id"} <= set(call.keys())


class TestSearch:
    def test_fts_match_returns_hits(self, store, tool):
        sid = store.new_session_id()
        store.record_turn(sid, _sample_turn())
        result = tool.invoke({"query": "quarterly report"})
        assert "quarterly report lives in reports/q3.md" in result
        assert f"session={sid[:8]}" in result

    def test_and_miss_falls_back_to_or(self, store, tool):
        """No single message contains all three tokens; OR fallback must still hit."""
        sid = store.new_session_id()
        store.record_turn(sid, _sample_turn())
        result = tool.invoke({"query": "quarterly report revenue"})
        assert "quarterly report lives" in result
        assert "revenue up 12%" in result

    def test_invalid_fts_syntax_falls_back_to_like(self, store, tool):
        sid = store.new_session_id()
        store.record_turn(sid, [HumanMessage(content='the "unbalanced quote topic')])
        result = tool.invoke({"query": '"unbalanced quote'})
        assert "unbalanced quote topic" in result

    def test_no_match_returns_sentinel(self, tool):
        assert tool.invoke({"query": "zzz_no_such_term_zzz"}) == "No past session matches found."

    def test_empty_query_is_rejected(self, tool):
        assert tool.invoke({"query": "   "}).startswith("Error:")

    def test_limit_is_clamped(self, store, tool):
        sid = store.new_session_id()
        for i in range(3):
            store.record_turn(sid, [HumanMessage(content=f" fibonacci note {i}")])
        result = tool.invoke({"query": "fibonacci", "limit": 2})
        assert result.count("fibonacci note") == 2


class TestSessionsListing:
    def test_list_sessions(self, store):
        sid = store.new_session_id()
        store.record_turn(sid, [HumanMessage(content="a")])
        sessions = store.list_sessions()
        assert len(sessions) == 1
        assert sessions[0]["session_id"] == sid
        assert sessions[0]["turns"] == 1


class TestCrossThread:
    def test_connection_created_in_worker_thread_usable_from_main_thread(self, store):
        """Regression: LangGraph runs sync tools (session_search) in a worker
        thread, so the first store touch can create the connection there;
        record_turn/close then happen on the event-loop thread. Without
        check_same_thread=False this raises sqlite3.ProgrammingError."""
        import threading

        sid = store.new_session_id()
        worker = threading.Thread(
            target=lambda: store.record_turn(sid, [HumanMessage(content="written from worker thread")])
        )
        worker.start()
        worker.join()

        messages = store.load_session(sid)  # main thread
        assert len(messages) == 1
        store.close()  # main thread closing a worker-created connection
