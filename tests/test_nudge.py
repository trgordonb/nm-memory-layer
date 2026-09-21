"""Tests for the periodic nudge module (NudgePolicy, prompt builder, flattener)."""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from nm_memory_layer import (
    DEFAULT_NUDGE_INTERVAL,
    MEMORY_CHAR_LIMIT,
    NudgePolicy,
    build_nudge_prompt,
    flatten_transcript,
)


class TestNudgePolicy:
    def test_default_interval(self):
        assert NudgePolicy().interval == DEFAULT_NUDGE_INTERVAL == 5

    def test_fires_only_at_interval(self):
        policy = NudgePolicy(interval=3)
        sid = "s1"
        policy.record_turn(sid)
        policy.record_turn(sid)
        assert not policy.should_nudge(sid)
        policy.record_turn(sid)
        assert policy.should_nudge(sid)

    def test_mark_nudged_resets_counter(self):
        policy = NudgePolicy(interval=2)
        sid = "s1"
        policy.record_turn(sid)
        policy.record_turn(sid)
        assert policy.should_nudge(sid)
        policy.mark_nudged(sid)
        assert not policy.should_nudge(sid)
        assert policy.turns_pending(sid) == 0

    def test_sessions_are_independent(self):
        policy = NudgePolicy(interval=1)
        policy.record_turn("a")
        assert policy.should_nudge("a")
        assert not policy.should_nudge("b")

    def test_interval_floor_is_one(self):
        assert NudgePolicy(interval=0).interval == 1
        assert NudgePolicy(interval=-5).interval == 1

    def test_unknown_session_has_no_pending_turns(self):
        assert NudgePolicy().turns_pending("nope") == 0


class TestNudgePrompt:
    def test_contains_layer_boundary_and_budget(self):
        prompt = build_nudge_prompt(chars_used=123)
        assert "memory_manage" in prompt
        assert "session_search" in prompt
        assert "EVERY future session" in prompt
        assert "No memory updates" in prompt
        assert f"{MEMORY_CHAR_LIMIT} chars" in prompt
        assert "123 already used" in prompt

    def test_encourages_silence(self):
        prompt = build_nudge_prompt(chars_used=0)
        assert "Most turns produce NO writes" in prompt
        assert "Silence is a valid" in prompt


class TestFlattenTranscript:
    def test_roles_and_content(self):
        transcript = flatten_transcript(
            [
                HumanMessage(content="How do I backfill EUR/USD?"),
                AIMessage(
                    content="",
                    tool_calls=[{"name": "execute", "args": {"command": "jetta backfill"}, "id": "c1"}],
                ),
                ToolMessage(content="done: 23 days", tool_call_id="c1", name="execute"),
                AIMessage(content="Backfilled successfully."),
            ]
        )
        assert transcript.startswith("user: How do I backfill EUR/USD?")
        assert "assistant: [called execute({'command': 'jetta backfill'})]" in transcript
        assert "tool result (execute): done: 23 days" in transcript
        assert "assistant: Backfilled successfully." in transcript

    def test_tool_call_text_preserved_when_content_empty(self):
        transcript = flatten_transcript(
            [AIMessage(content="", tool_calls=[{"name": "ls", "args": {"directory": "."}, "id": "c"}])]
        )
        assert "ls" in transcript and "directory" in transcript

    def test_empty_messages_yield_empty_string(self):
        assert flatten_transcript([]) == ""
