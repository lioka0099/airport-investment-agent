"""The one entry point for LLM calls: provider chains, key detection, fallback.

This is the adapter layer: the rest of the app sees agent_invoke / review_invoke and a
provider name, never an SDK.
"""

import os
import time
from collections.abc import Mapping

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from pydantic import BaseModel

from airport_agent import config
from airport_agent.llm.resilience import AllProvidersFailed, CircuitBreaker, invoke_with_fallback

KEY_ENV = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY",
           "google": "GOOGLE_API_KEY"}
BREAKER = CircuitBreaker()


def make_chat_model(provider: str, model: str) -> BaseChatModel:
    # max_retries=0: our policy in resilience.py does the retrying, exactly once.
    common = {"model": model, "timeout": config.LLM_TIMEOUT_S, "max_retries": 0}
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(**common)
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(**common)
    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(**common)
    raise ValueError(f"Unknown provider: {provider}")


class LLMRouter:
    def __init__(self, agent_chain=config.AGENT_CHAIN, reviewer_chain=config.REVIEWER_CHAIN, *,
                 breaker: CircuitBreaker | None = None, factory=make_chat_model,
                 env: Mapping[str, str] = os.environ, sleep=time.sleep):
        def has_key(provider: str) -> bool:
            return bool(env.get(KEY_ENV[provider]))

        self.agent_chain = [(p, m) for p, m in agent_chain if has_key(p)]
        self.reviewer_chain = [(p, m) for p, m in reviewer_chain if has_key(p)]
        if not self.agent_chain:
            names = " or ".join(KEY_ENV[p] for p, _ in agent_chain)
            raise RuntimeError(f"No API key for any agent provider. Set {names}.")
        self.breaker = breaker or BREAKER
        self.factory = factory
        self.sleep = sleep
        self._models: dict[tuple[str, str], BaseChatModel] = {}

    def _model(self, provider: str, model: str):
        key = (provider, model)
        if key not in self._models:
            self._models[key] = self.factory(provider, model)
        return self._models[key]

    def agent_invoke(self, messages: list, tools: list) -> tuple[AIMessage, str]:
        chain = [(p, lambda p=p, m=m: self._model(p, m).bind_tools(tools).invoke(messages))
                 for p, m in self.agent_chain]
        return invoke_with_fallback(chain, self.breaker, self.sleep)

    def review_invoke(self, messages: list, schema: type[BaseModel],
                      exclude_provider: str | None) -> tuple[BaseModel, str]:
        reviewers = [(p, m) for p, m in self.reviewer_chain if p != exclude_provider]
        if not reviewers:  # never let the writer check itself: the answer shows "Not verified"
            raise AllProvidersFailed([(exclude_provider or "?", "no other reviewer")])
        chain = [(p, lambda p=p, m=m: self._model(p, m).with_structured_output(schema)
                  .invoke(messages)) for p, m in reviewers]
        return invoke_with_fallback(chain, self.breaker, self.sleep)
