"""One-line session titles (Hermes-style sidebar labeling).

After a session's first completed turn, the consumer stores a title for the
session so UI lists can show "what was this about" instead of a raw session
id. The titler runs on the SECONDARY LLM (same OpenRouter configuration as
the search summarizer / context compressor) over the opening user message and
the assistant's reply; the consumer writes a synchronous fallback (truncated
first user message) first, so a title always exists even when the model is
disabled, slow, or failing.

Env (same provider settings as the other secondary-LLM consumers):

- ``SESSION_TITLER_ENABLED``     — "false"/"0" to disable (default ON when
                                   ``OPENROUTER_API_KEY`` + ``OPENROUTER_MODEL`` are set)
- ``OPENROUTER_API_KEY``         — secondary provider key
- ``OPENROUTER_BASE_URL``        — default https://openrouter.ai/api/v1
- ``OPENROUTER_MODEL``           — e.g. an affordable fast model id

Failure semantics: the titler never raises to the caller — any model error
returns None and the consumer keeps the fallback title.
"""

import logging
import os
import re

from .summarizer import _content_to_text, _env_bool

logger = logging.getLogger(__name__)

_SPECIAL_TOKEN_RE = re.compile(r"<\|[^|>]*\|>")
_MAX_TITLE_CHARS = 60

_TITLER_SYSTEM_PROMPT = (
    "You write short titles for AI-assistant conversations, shown in a session "
    "sidebar. Given the conversation's opening exchange, reply with ONLY the "
    "title: 3-6 words, plain text, no quotes, no punctuation at the end, no "
    "preface. Capture the user's topic, not the assistant's answer."
)


class SessionTitler:
    """Wraps any sync LangChain chat model as a session titler."""

    def __init__(self, model, label: str | None = None):
        self.model = model
        self.label = label or getattr(model, "model_name", None) or "llm"

    def title_session(self, first_user: str, first_assistant: str) -> str | None:
        """Return a one-line title, or None on any failure (caller keeps fallback)."""
        user = first_user.strip()[:500]
        assistant = first_assistant.strip()[:500]
        if not user:
            return None
        messages = [
            ("system", _TITLER_SYSTEM_PROMPT),
            ("human", f"USER: {user}\n\nASSISTANT: {assistant}\n\nTitle:"),
        ]
        try:
            response = self.model.invoke(messages)
        except Exception as exc:
            logger.warning("session titler [%s] failed: %s", self.label, str(exc)[:200])
            return None
        text = _content_to_text(response.content)
        text = _SPECIAL_TOKEN_RE.sub("", text).strip().strip('"').strip("'")
        text = re.sub(r"\s+", " ", text)
        if not text or _is_junk_title(text):
            logger.warning(
                "session titler [%s] produced no usable title — keeping fallback", self.label
            )
            return None
        if len(text) > _MAX_TITLE_CHARS:
            text = text[: _MAX_TITLE_CHARS - 1].rstrip() + "…"
        return text


def _is_junk_title(text: str) -> bool:
    """Refusals / preambles from weak models — keep the fallback instead."""
    lowered = text.lower()
    return lowered.startswith(("i'm sorry", "i am sorry", "title:", "here is", "here's"))


def create_openrouter_titler() -> SessionTitler | None:
    """Build the OpenRouter-backed titler if configured and not disabled, else None.

    Default is ON when OPENROUTER_API_KEY/OPENROUTER_MODEL are set (titles are
    one tiny call per session); ``SESSION_TITLER_ENABLED=false`` disables.
    Never raises: misconfiguration logs and disables, leaving consumers on the
    fallback title.
    """
    if not _env_bool("SESSION_TITLER_ENABLED", default="true"):
        return None
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    model = os.getenv("OPENROUTER_MODEL", "")
    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    if not api_key or not model:
        logger.info(
            "Session titler disabled: OPENROUTER_API_KEY/OPENROUTER_MODEL not set "
            "(fallback titles only)."
        )
        return None
    try:
        from langchain_openai import ChatOpenAI  # lazy: only needed when enabled

        llm = ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=0,
            # max_tokens, NOT max_completion_tokens: OpenRouter reasoning models
            # count reasoning tokens against max_completion_tokens and can return
            # empty content with finish_reason=length when the budget is exhausted.
            max_tokens=512,
            timeout=10,
            max_retries=1,
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Failed to construct session titler model (%s) — disabled.", exc)
        return None
    logger.info("session titler enabled: openrouter/%s", model)
    return SessionTitler(llm, label=f"openrouter:{model}")
