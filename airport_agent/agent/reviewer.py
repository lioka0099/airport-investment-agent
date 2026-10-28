"""Reviewer: a second LLM, from a different provider, checks conclusions against findings."""

import json

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from airport_agent.agent.grounding import format_value
from airport_agent.agent.prompts import REVIEWER_PROMPT

MAX_ISSUES = 5


class Verdict(BaseModel):
    approved: bool = Field(description="true if the answer can be shown as is")
    issues: list[str] = Field(default_factory=list,
                              description="specific problems the writer must fix")


def review(router, *, question: str, draft: dict, facts: dict[str, dict], findings: list[dict],
           assumptions: list[str], exclude_provider: str | None) -> tuple[Verdict, str]:
    payload = {
        "question": question,
        "answer": draft["text"],
        "facts": {ref: format_value(facts[ref]) for ref in draft["refs"] if ref in facts},
        "findings": findings,
        "assumptions": assumptions,
        "orphan_numbers": draft["orphans"],
    }
    # escape < and > so untrusted text cannot close the <review_input> tag (still valid JSON)
    body = json.dumps(payload, default=str).replace("<", "\\u003c").replace(">", "\\u003e")
    messages = [SystemMessage(REVIEWER_PROMPT),
                HumanMessage(f"<review_input>\n{body}\n</review_input>")]
    verdict, provider = router.review_invoke(messages, Verdict, exclude_provider)
    # cap in code, not in the schema: a schema violation would trigger a provider fallback
    verdict.issues = verdict.issues[:MAX_ISSUES]
    return verdict, provider
