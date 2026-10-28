import httpx
import openai
import pytest
from langchain_core.messages import AIMessage
from pydantic import BaseModel

from airport_agent import config
from airport_agent.llm.resilience import CircuitBreaker
from airport_agent.llm.router import KEY_ENV, LLMRouter, make_chat_model

KEYS = {"OPENAI_API_KEY": "k", "ANTHROPIC_API_KEY": "k", "GOOGLE_API_KEY": "k"}
REQ = httpx.Request("POST", "https://api.example")


class Verdict(BaseModel):
    approved: bool


class FakeModel:
    """Stands in for a LangChain chat model: bind_tools / with_structured_output / invoke."""

    def __init__(self, provider, outcomes):
        self.provider, self.outcomes = provider, outcomes

    def bind_tools(self, tools):
        return self

    def with_structured_output(self, schema):
        return self

    def invoke(self, messages):
        outcome = self.outcomes[self.provider].pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


# Fixed chain: tests check fallback mechanics, not which provider production puts first.
CHAIN = [("openai", "gpt-5.4-mini"), ("anthropic", "claude-haiku-5-5")]


def router(outcomes, env=KEYS):
    return LLMRouter(CHAIN, factory=lambda p, m: FakeModel(p, outcomes), env=env,
                     breaker=CircuitBreaker(), sleep=lambda s: None)


def test_agent_uses_primary():
    msg, provider = router({"openai": [AIMessage("hi")]}).agent_invoke([], [])
    assert provider == "openai" and msg.content == "hi"


def test_agent_falls_back_on_503():
    err = openai.APIStatusError("x", response=httpx.Response(503, request=REQ), body=None)
    r = router({"openai": [err, err, err], "anthropic": [AIMessage("from claude")]})
    msg, provider = r.agent_invoke([], [])
    assert provider == "anthropic"


def test_reviewer_skips_writer_provider():
    r = router({"anthropic": [Verdict(approved=True)], "google": [Verdict(approved=False)]})
    verdict, provider = r.review_invoke([], Verdict, exclude_provider="anthropic")
    assert provider == "google" and verdict.approved is False


def test_providers_without_keys_are_dropped():
    r = router({"anthropic": [AIMessage("only claude")]}, env={"ANTHROPIC_API_KEY": "k"})
    assert [p for p, _ in r.agent_chain] == ["anthropic"]
    assert r.agent_invoke([], [])[1] == "anthropic"


def test_reviewer_degrades_to_same_provider_when_alone():
    r = router({"openai": [Verdict(approved=True)]}, env={"OPENAI_API_KEY": "k"})
    assert r.review_invoke([], Verdict, exclude_provider="openai")[1] == "openai"


def test_no_keys_fails_fast_with_actionable_message():
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        router({}, env={})


@pytest.mark.parametrize("provider,model", config.AGENT_CHAIN + config.REVIEWER_CHAIN)
def test_sdk_clients_never_retry_themselves(monkeypatch, provider, model):
    # Only resilience.py retries; SDK retries would multiply with ours.
    monkeypatch.setenv(KEY_ENV[provider], "dummy")
    assert make_chat_model(provider, model).max_retries == 0
