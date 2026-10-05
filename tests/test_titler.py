"""Tests for the session titler (SessionTitler + factory)."""

import pytest

from nm_memory_layer.titler import SessionTitler, create_openrouter_titler


class FakeTitleModel:
    """Sync .invoke model stand-in; scripted responses (raises if scripted to)."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.model_name = "fake-titler"

    def invoke(self, messages):
        self.calls.append(messages)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class Resp:
    def __init__(self, content):
        self.content = content


@pytest.fixture()
def titler():
    return SessionTitler(FakeTitleModel([Resp("Quarterly Report Q3")]), label="fake")


def test_title_happy_path(titler):
    title = titler.title_session(
        "Remember: the quarterly report lives in reports/q3.md",
        "Sure — the Q3 report is at reports/q3.md.",
    )
    assert title == "Quarterly Report Q3"


def test_title_prompt_carries_exchange(titler):
    model = titler.model
    titler.title_session("user text", "assistant text")
    assert len(model.calls) == 1
    flattened = str(model.calls[0])
    assert "user text" in flattened and "assistant text" in flattened


def test_title_model_error_returns_none():
    titler = SessionTitler(FakeTitleModel([RuntimeError("boom")]))
    assert titler.title_session("user", "assistant") is None


def test_title_junk_refusal_returns_none():
    titler = SessionTitler(FakeTitleModel([Resp("I'm sorry, I cannot help with that.")]))
    assert titler.title_session("user", "assistant") is None


def test_title_empty_user_returns_none_without_calling_model():
    model = FakeTitleModel([Resp("whatever")])
    titler = SessionTitler(model)
    assert titler.title_session("   ", "assistant") is None
    assert model.calls == []


def test_title_long_output_truncated():
    long_title = "A" * 200
    titler = SessionTitler(FakeTitleModel([Resp(long_title)]))
    title = titler.title_session("user", "assistant")
    assert title is not None and len(title) <= 60
    assert title.endswith("…")


def test_factory_disabled_by_env(monkeypatch):
    monkeypatch.delenv("SESSION_TITLER_ENABLED", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    # default ON — but without keys it must quietly disable, never raise
    monkeypatch.delenv("OPENROUTER_MODEL", raising=False)
    assert create_openrouter_titler() is None


def test_factory_kill_switch(monkeypatch):
    monkeypatch.setenv("SESSION_TITLER_ENABLED", "false")
    monkeypatch.setenv("OPENROUTER_API_KEY", "key")
    monkeypatch.setenv("OPENROUTER_MODEL", "some-model")
    assert create_openrouter_titler() is None


def test_factory_builds_when_enabled(monkeypatch):
    monkeypatch.setenv("SESSION_TITLER_ENABLED", "true")
    monkeypatch.setenv("OPENROUTER_API_KEY", "key")
    monkeypatch.setenv("OPENROUTER_MODEL", "some-model")

    import langchain_openai

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.model_name = "some-model"

    monkeypatch.setattr(langchain_openai, "ChatOpenAI", FakeChatOpenAI)
    titler = create_openrouter_titler()
    assert isinstance(titler, SessionTitler)
    assert titler.label == "openrouter:some-model"
