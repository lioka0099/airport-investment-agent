import anthropic
import httpx
import openai
import pytest
from google.genai import errors as genai_errors
from langchain_core.exceptions import OutputParserException

from airport_agent.llm.resilience import (
    AllProvidersFailed,
    CircuitBreaker,
    ErrorKind,
    classify_error,
    invoke_with_fallback,
)

REQ = httpx.Request("POST", "https://api.example")


def oa(status, headers=None):
    return openai.APIStatusError("x", response=httpx.Response(status, headers=headers, request=REQ),
                                 body=None)


def wrapped_google(code):
    try:
        raise genai_errors.ServerError(code, {"error": {"code": code, "message": "x"}})
    except genai_errors.APIError as inner:
        try:
            raise RuntimeError("ChatGoogleGenerativeAIError: wrapped") from inner
        except RuntimeError as outer:
            return outer


@pytest.mark.parametrize("exc,kind", [
    (oa(400), ErrorKind.FATAL),
    (oa(401), ErrorKind.FALLBACK),
    (oa(403), ErrorKind.FALLBACK),
    (oa(404), ErrorKind.FALLBACK),
    (oa(429), ErrorKind.FALLBACK),                          # no Retry-After
    (oa(429, {"retry-after": "30"}), ErrorKind.FALLBACK),   # too long to wait
    (oa(500), ErrorKind.RETRYABLE),
    (anthropic.APIStatusError("x", response=httpx.Response(529, request=REQ), body=None),
     ErrorKind.RETRYABLE),
    (openai.APITimeoutError(request=REQ), ErrorKind.RETRYABLE),
    (anthropic.APIConnectionError(request=REQ), ErrorKind.RETRYABLE),
    (wrapped_google(503), ErrorKind.RETRYABLE),
    (OutputParserException("bad json"), ErrorKind.FALLBACK),
    (KeyError("bug"), ErrorKind.FATAL),
])
def test_classify_error(exc, kind):
    assert classify_error(exc)[0] is kind


def test_429_with_short_retry_after_is_retryable_with_delay():
    assert classify_error(oa(429, {"retry-after": "2"})) == (ErrorKind.RETRYABLE, 2.0)


def _flaky(*outcomes):
    it = iter(outcomes)

    def call():
        o = next(it)
        if isinstance(o, BaseException):
            raise o
        return o

    return call


def test_retries_then_succeeds_on_same_provider():
    sleeps = []
    result, provider = invoke_with_fallback(
        [("openai", _flaky(oa(503), "ok"))], CircuitBreaker(), sleep=sleeps.append)
    assert (result, provider) == ("ok", "openai")
    assert len(sleeps) == 1


def test_falls_back_after_retries_exhausted_and_opens_circuit():
    breaker = CircuitBreaker()
    result, provider = invoke_with_fallback(
        [("openai", _flaky(oa(503), oa(503), oa(503))), ("anthropic", _flaky("ok"))],
        breaker, sleep=lambda s: None)
    assert provider == "anthropic"
    assert breaker.is_open("openai")


def test_auth_error_falls_back_immediately():
    sleeps = []
    _, provider = invoke_with_fallback(
        [("openai", _flaky(oa(401))), ("anthropic", _flaky("ok"))], CircuitBreaker(),
        sleep=sleeps.append)
    assert provider == "anthropic" and sleeps == []


def test_fatal_raises_without_fallback():
    with pytest.raises(openai.APIStatusError):
        invoke_with_fallback([("openai", _flaky(oa(400))), ("anthropic", _flaky("ok"))],
                             CircuitBreaker(), sleep=lambda s: None)


def test_open_circuit_skips_provider():
    breaker = CircuitBreaker()
    breaker.record_failure("openai")
    _, provider = invoke_with_fallback(
        [("openai", _flaky("should not run")), ("anthropic", _flaky("ok"))], breaker)
    assert provider == "anthropic"


def test_all_circuits_open_still_tries():
    breaker = CircuitBreaker()
    breaker.record_failure("openai")
    assert invoke_with_fallback([("openai", _flaky("ok"))], breaker) == ("ok", "openai")


def test_circuit_closes_after_window():
    now = [0.0]
    breaker = CircuitBreaker(open_s=60, clock=lambda: now[0])
    breaker.record_failure("openai")
    now[0] = 61
    assert not breaker.is_open("openai")


def test_all_providers_failed():
    with pytest.raises(AllProvidersFailed) as info:
        invoke_with_fallback([("openai", _flaky(oa(401))), ("anthropic", _flaky(oa(403)))],
                             CircuitBreaker())
    assert [p for p, _ in info.value.errors] == ["openai", "anthropic"]
