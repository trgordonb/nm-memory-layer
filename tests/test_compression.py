"""Tests for context compression with lineage (Phase 5)."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from nm_memory_layer import (
    ConversationCompressor,
    SessionStore,
    create_openrouter_compressor,
    estimate_tokens,
    split_into_turns,
)
from nm_memory_layer.compression import DEFAULT_COMPRESSION_TOKEN_THRESHOLD


class FakeCompressorModel:
    """Returns a fixed summary for every call."""

    model_name = "fake-compressor"

    def __init__(self, content="Compressed: user analyzed NVDA RSI; files in workspace/; S3 fallback used."):
        self.content = content
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        return AIMessage(content=self.content)


def make_history(turn_texts, tool_pair_in=None):
    """Build one segment per text; optionally append a tool-call pair to a turn."""
    messages = []
    for i, text in enumerate(turn_texts):
        messages.append(HumanMessage(content=text))
        messages.append(AIMessage(content=f"reply to: {text}"))
    return messages


class TestSegmentation:
    def test_splits_on_human_messages(self):
        msgs = [
            HumanMessage(content="t1"), AIMessage(content="a1"),
            HumanMessage(content="t2"), AIMessage(content="a2"),
            HumanMessage(content="t3"), AIMessage(content="a3"),
        ]
        segs = split_into_turns(msgs)
        assert len(segs) == 3
        assert segs[0][0].content == "t1" and segs[2][0].content == "t3"

    def test_leading_non_human_message_kept_in_first_segment(self):
        msgs = [SystemMessage(content="sys"), HumanMessage(content="t1"), AIMessage(content="a1")]
        segs = split_into_turns(msgs)
        assert len(segs) == 1 and segs[0][0].content == "sys"

    def test_empty_history(self):
        assert split_into_turns([]) == []


class TestTokenEstimate:
    def test_rough_chars_per_token(self):
        msgs = [HumanMessage(content="x" * 400)]
        assert 90 <= estimate_tokens(msgs) <= 110

    def test_default_threshold_constant(self):
        assert DEFAULT_COMPRESSION_TOKEN_THRESHOLD == 24000


class TestCompress:
    def _long_history(self, n_turns=6, filler=2500):
        texts = [f"turn {i}: analyze dataset {i} " + "x" * filler for i in range(1, n_turns + 1)]
        return make_history(texts)

    def test_provider_reported_prompt_tokens_drive_the_trigger(self):
        """The real usage measure is the last AIMessage's reported prompt_tokens
        (includes system prompt + tool schemas), matching what LangSmith shows —
        even when message content alone is tiny."""
        model = FakeCompressorModel()
        c = ConversationCompressor(model, token_threshold=24000, keep_recent_turns=1)
        msgs = make_history(["t1", "t2", "t3", "t4"])  # tiny content, would never estimate past 24K
        msgs[-1] = AIMessage(  # the last message of history carries the provider usage
            content="done",
            response_metadata={"token_usage": {"prompt_tokens": 29535, "completion_tokens": 500}},
        )
        res = c.compress(msgs)
        assert res.compressed
        assert res.summarized_first_turn == 2 and res.summarized_last_turn == 3

    def test_estimate_fallback_without_metadata(self):
        model = FakeCompressorModel()
        c = ConversationCompressor(model, token_threshold=10, keep_recent_turns=1)
        msgs = make_history(["t1", "t2", "t3"])  # no response_metadata anywhere
        assert c.compress(msgs).compressed  # falls back to chars/4 estimate

    def test_below_threshold_is_noop(self):
        c = ConversationCompressor(FakeCompressorModel(), token_threshold=10**9)
        msgs = self._long_history()
        res = c.compress(msgs)
        assert not res.compressed and res.reason == "below threshold"
        assert res.compressed_messages == msgs

    def test_too_few_turns_is_noop(self):
        c = ConversationCompressor(FakeCompressorModel(), token_threshold=10)
        msgs = self._long_history(n_turns=3)  # keep_recent=2 -> need >= 4 segments
        res = c.compress(msgs)
        assert not res.compressed and res.reason == "no middle turns"

    def test_compression_keeps_first_and_recent_summarizes_middle(self):
        model = FakeCompressorModel()
        c = ConversationCompressor(model, token_threshold=10, keep_recent_turns=2)
        msgs = self._long_history(n_turns=6)
        res = c.compress(msgs)
        assert res.compressed
        assert res.summarized_first_turn == 2 and res.summarized_last_turn == 4
        # structure: turn1 verbatim (human+ai), summary SystemMessage, turns 5-6 verbatim
        assert res.compressed_messages[0].content.startswith("turn 1:")
        assert res.compressed_messages[1].content.startswith("reply to: turn 1:")
        summary_msg = res.compressed_messages[2]
        assert isinstance(summary_msg, SystemMessage)
        assert "conversation_summary" in summary_msg.content
        assert "turns=\"2-4\"" in summary_msg.content
        assert res.compressed_messages[3].content.startswith("turn 5:")
        assert res.compressed_messages[-2].content.startswith("turn 6:")
        assert res.compressed_messages[-1].content.startswith("reply to: turn 6:")
        assert res.compressed_count < res.original_count
        # the model saw ONLY the middle turns
        transcript = model.calls[0][1][1]
        assert "turn 2: analyze" in transcript and "turn 4: analyze" in transcript
        assert "turn 1: analyze" not in transcript and "turn 5: analyze" not in transcript

    def test_tool_pairing_never_dangling(self):
        """Middle turns may contain tool pairs; removal must keep pairs intact
        (they are removed as whole segments)."""
        model = FakeCompressorModel()
        c = ConversationCompressor(model, token_threshold=10, keep_recent_turns=1)
        msgs = self._long_history(n_turns=5)
        msgs += [
            HumanMessage(content="turn 6 final"),
            AIMessage(content="", tool_calls=[{"name": "execute", "args": {"command": "ls"}, "id": "c9"}]),
            ToolMessage(content="out", tool_call_id="c9", name="execute"),
        ]
        res = c.compress(msgs)
        assert res.compressed
        compressed = res.compressed_messages
        for i, m in enumerate(compressed):
            tc = getattr(m, "tool_calls", None)
            if tc:
                following = compressed[i + 1]
                assert isinstance(following, ToolMessage)
                assert following.tool_call_id == tc[0]["id"]

    def test_summary_truncated_to_budget(self):
        model = FakeCompressorModel(content="y" * 5000)
        c = ConversationCompressor(model, token_threshold=10, max_summary_chars=300)
        res = c.compress(self._long_history())
        assert res.compressed
        assert len(res.summary) == 300  # truncated to fit the budget
        assert res.summary.endswith("[…]")

    def test_model_error_aborts_compression(self):
        class Exploding:
            model_name = "x"
            def invoke(self, messages):
                raise RuntimeError("provider down")

        c = ConversationCompressor(Exploding(), token_threshold=10)
        msgs = self._long_history()
        res = c.compress(msgs)
        assert not res.compressed and res.compressed_messages == msgs

    def test_empty_model_output_aborts(self):
        c = ConversationCompressor(FakeCompressorModel(content=""), token_threshold=10)
        msgs = self._long_history()
        res = c.compress(msgs)
        assert not res.compressed

    def test_pseudo_tool_call_output_aborts(self):
        c = ConversationCompressor(FakeCompressorModel(content="<|tool_call_start|>[answer(q='x')]"), token_threshold=10)
        res = c.compress(self._long_history())
        assert not res.compressed

    def test_special_tokens_stripped_from_summary(self):
        model = FakeCompressorModel(content="<|fim|>clean summary<|end|>")
        c = ConversationCompressor(model, token_threshold=10)
        res = c.compress(self._long_history())
        assert res.compressed and "<|" not in res.summary and "clean summary" in res.summary

    def test_empty_history_noop(self):
        c = ConversationCompressor(FakeCompressorModel(), token_threshold=10)
        assert not c.compress([]).compressed


class TestFactory:
    def test_disabled_by_default(self, monkeypatch):
        for var in ("COMPRESSION_ENABLED", "OPENROUTER_API_KEY", "OPENROUTER_MODEL"):
            monkeypatch.delenv(var, raising=False)
        assert create_openrouter_compressor() is None

    def test_enabled_but_unconfigured_returns_none(self, monkeypatch):
        monkeypatch.setenv("COMPRESSION_ENABLED", "true")
        for var in ("OPENROUTER_API_KEY", "OPENROUTER_MODEL"):
            monkeypatch.delenv(var, raising=False)
        assert create_openrouter_compressor() is None

    def test_enabled_and_configured_builds_compressor(self, monkeypatch):
        monkeypatch.setenv("COMPRESSION_ENABLED", "yes")
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
        monkeypatch.setenv("OPENROUTER_MODEL", "test/model")
        monkeypatch.setenv("COMPRESSION_TOKEN_THRESHOLD", "1000")
        monkeypatch.setenv("COMPRESSION_KEEP_RECENT_TURNS", "3")
        c = create_openrouter_compressor()
        assert isinstance(c, ConversationCompressor)
        assert c.label == "openrouter:test/model"
        assert c.token_threshold == 1000 and c.keep_recent_turns == 3


class TestLineage:
    def test_record_and_list_compressions(self, tmp_path):
        store = SessionStore(db_path=str(tmp_path / "s.db"))
        sid = store.new_session_id()
        assert store.list_compressions(sid) == []
        store.record_compression(sid, summary="turns 2-4 about NVDA", summarized_first_turn=2,
                                 summarized_last_turn=4, message_count=18, model="openrouter:test")
        store.record_compression(sid, summary="turns 2-3 again", summarized_first_turn=2,
                                 summarized_last_turn=3, message_count=9)
        events = store.list_compressions(sid)
        assert len(events) == 2
        assert events[0]["summarized_first_turn"] == 2 and events[0]["model"] == "openrouter:test"
        assert events[1]["summary"] == "turns 2-3 again"
        assert events[0]["created_at"] <= events[1]["created_at"]
        store.close()
