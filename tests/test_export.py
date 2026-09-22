"""Tests for the session exporter (offline self-evolution / datagen substrate)."""

import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from nm_memory_layer import SessionStore


@pytest.fixture()
def store(tmp_path):
    s = SessionStore(db_path=str(tmp_path / "sessions.db"))
    sid = s.new_session_id()
    s.record_turn(sid, [
        HumanMessage(content="Backfill EURUSD and compute RSI-14"),
        AIMessage(content="", tool_calls=[{"name": "execute", "args": {"command": "jetta backfill eurusd"}, "id": "call_1"}]),
        ToolMessage(content="backfill complete: 23 days", tool_call_id="call_1", name="execute"),
        AIMessage(content="Done. RSI-14 saved to workspace/RSI-EURUSD.csv"),
    ])
    s.record_compression(sid, summary="compressed notes", summarized_first_turn=1,
                         summarized_last_turn=1, message_count=4, model="openrouter:test")
    yield s, sid
    s.close()


class TestExportSession:
    def test_export_shape(self, store):
        s, sid = store
        out = s.export_session(sid)
        assert out["session_id"] == sid
        assert out["turns"] == 1  # fixture records a single 4-message turn
        assert out["message_count"] == 4
        assert len(out["messages"]) == 4

    def test_tool_calls_round_trip_as_parsed_json(self, store):
        s, sid = store
        out = s.export_session(sid)
        msg = out["messages"][1]
        assert msg["role"] == "assistant"
        assert msg["tool_calls"] == [{"name": "execute", "args": {"command": "jetta backfill eurusd"}, "id": "call_1"}]
        tool = out["messages"][2]
        assert tool["role"] == "tool" and tool["tool_name"] == "execute" and tool["tool_call_id"] == "call_1"

    def test_conversation_projection_is_openai_style(self, store):
        s, sid = store
        conv = s.export_session(sid)["conversation"]
        assert conv[0] == {"role": "user", "content": "Backfill EURUSD and compute RSI-14"}
        assert conv[1]["role"] == "assistant"
        assert conv[1]["tool_calls"][0]["function"]["name"] == "execute"
        assert json.loads(conv[1]["tool_calls"][0]["function"]["arguments"])["command"] == "jetta backfill eurusd"
        assert conv[2]["role"] == "tool" and conv[2]["tool_call_id"] == "call_1"

    def test_pass_at_20_and_1(self, store):
        """The self-evolution GET_PASS@K convention: a recent assistant task
        response and the first user task are the mined (task, response) pair."""
        s, sid = store
        out = s.export_session(sid)
        assert out["conversation"][0]["content"].startswith("Backfill EURUSD")
        assert out["conversation"][-1]["role"] == "assistant"

    def test_compression_lineage_included(self, store):
        s, sid = store
        out = s.export_session(sid)
        assert len(out["compressions"]) == 1
        assert out["compressions"][0]["model"] == "openrouter:test"

    def test_unknown_session_returns_none(self, store):
        s, _ = store
        assert s.export_session("nope") is None


class TestExportJsonl:
    def test_one_trajectory_per_line(self, store, tmp_path):
        s, sid = store
        s2 = s.new_session_id()
        s.record_turn(s2, [HumanMessage(content="second session question")])
        path = str(tmp_path / "dump.jsonl")
        assert s.export_to_jsonl(path) == 2
        records = [json.loads(line) for line in open(path).read().strip().splitlines()]
        assert len(records) == 2
        ids = [r["session_id"] for r in records]
        assert set(ids) == {sid, s2}
        # newest-first order matches list_sessions()
        assert ids[0] == s2
        main = next(r for r in records if r["session_id"] == sid)
        assert main["messages"][0]["content"] == "Backfill EURUSD and compute RSI-14"

    def test_specific_session_ids(self, store, tmp_path):
        s, sid = store
        s2 = s.new_session_id()
        s.record_turn(s2, [HumanMessage(content="other")])
        path = str(tmp_path / "one.jsonl")
        assert s.export_to_jsonl(path, session_ids=[s2]) == 1
        assert json.loads(open(path).read())["session_id"] == s2

    def test_empty_archive_writes_nothing(self, tmp_path):
        s = SessionStore(db_path=str(tmp_path / "empty.db"))
        path = str(tmp_path / "x.jsonl")
        assert s.export_to_jsonl(path) == 0
        assert open(path).read() == ""
        s.close()
