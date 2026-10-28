"""Provider error policy, circuit breaker, and the retry/fallback loop.

classify_error is the only code that knows SDK exception shapes. Everything else works
with ErrorKind.
"""

import logging
import random
import time
from collections.abc import Callable
from enum import Enum

import anthropic
import httpx
import openai
from langchain_core.exceptions import OutputParserException
from pydantic import ValidationError

from airport_agent import config

log = logging.getLogger(__name__)

CONNECTION_ERRORS = (TimeoutError, ConnectionError, httpx.TimeoutException, httpx.TransportError,
                     openai.APIConnectionError, anthropic.APIConnectionError)
PARSE_ERRORS = (OutputParserException, ValidationError)


class ErrorKind(Enum):
    RETRYABLE = "retryable"  # transient: back off and try the same provider again
    FALLBACK = "fallback"    # this provider cannot serve us now: move to the next one
    FATAL = "fatal"          # our bug: a fallback would only hide it


class AllProvidersFailed(Exception):
    def __init__(self, errors: list[tuple[str, str]]):
        super().__init__(f"All LLM providers failed: {errors}")
        self.errors = errors


def _chain(exc: BaseException):
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        yield exc
        exc = exc.__cause__ or exc.__context__


def _status(exc: BaseException) -> int | None:
    for attr in ("status_code", "code"):  # openai/anthropic use status_code, google-genai code
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
    return None


def _retry_after(exc: BaseException) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None) or {}
    try:
        return max(float(headers.get("retry-after")), 0.0)  # negative header would crash sleep()
    except (TypeError, ValueError):
        return None


def classify_error(exc: BaseException) -> tuple[ErrorKind, float | None]:
    for e in _chain(exc):
        if isinstance(e, CONNECTION_ERRORS):
            return ErrorKind.RETRYABLE, None
        if isinstance(e, PARSE_ERRORS):
            return ErrorKind.FALLBACK, None
        status = _status(e)
        if status is None:
            continue
        if status == 429:
            wait = _retry_after(e)
            if wait is not None and wait <= config.RETRY_AFTER_MAX_S:
                return ErrorKind.RETRYABLE, wait
            return ErrorKind.FALLBACK, None
        if status in (401, 403, 404):
            return ErrorKind.FALLBACK, None
        if status in (408, 409) or status >= 500:
            return ErrorKind.RETRYABLE, None
        if 400 <= status < 500:
            return ErrorKind.FATAL, None
    return ErrorKind.FATAL, None


class CircuitBreaker:
    """Skip a provider for open_s seconds after it failed. Shared by all sessions in the process."""

    def __init__(self, open_s: float = config.CIRCUIT_OPEN_S, clock=time.monotonic):
        self.open_s = open_s
        self.clock = clock
        self._failed_at: dict[str, float] = {}

    def is_open(self, provider: str) -> bool:
        failed_at = self._failed_at.get(provider)
        return failed_at is not None and self.clock() - failed_at < self.open_s

    def record_failure(self, provider: str) -> None:
        self._failed_at[provider] = self.clock()

    def record_success(self, provider: str) -> None:
        self._failed_at.pop(provider, None)


def invoke_with_fallback[T](chain: list[tuple[str, Callable[[], T]]], breaker: CircuitBreaker,
                         sleep=time.sleep, max_retries: int = config.LLM_MAX_RETRIES
                         ) -> tuple[T, str]:
    # If every circuit is open, try them all anyway rather than refusing to answer.
    candidates = [c for c in chain if not breaker.is_open(c[0])] or chain
    errors: list[tuple[str, str]] = []
    for provider, call in candidates:
        for attempt in range(max_retries + 1):
            try:
                result = call()
                breaker.record_success(provider)
                return result, provider
            except Exception as exc:
                kind, retry_after = classify_error(exc)
                # Log the type and kind only: messages can echo request content.
                log.warning("LLM %s failed: %s (%s, attempt %d)", provider,
                            type(exc).__name__, kind.name, attempt + 1)
                if kind is ErrorKind.FATAL:
                    raise
                if kind is ErrorKind.RETRYABLE and attempt < max_retries:
                    sleep(retry_after if retry_after is not None
                          else 2**attempt + random.uniform(0, 0.5))
                    continue
                breaker.record_failure(provider)
                errors.append((provider, f"{kind.name}: {type(exc).__name__}"))
                break
    raise AllProvidersFailed(errors)
