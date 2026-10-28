"""Reviewer: a second LLM, from a different provider, checks conclusions against findings."""

import json
import re

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from airport_agent import config
from airport_agent.agent.grounding import format_value
from airport_agent.agent.prompts import REVIEWER_PROMPT

MAX_ISSUES = 5
# The writer learns the method from tool notes; the reviewer needs it too, or it flags the
# method's own labels and constants as "unsupported". Built from config: one source of truth.
METHOD = {
    "peer_group": f"US airports with at least {config.PEER_MIN_PASSENGERS:,} passengers in the "
                  "latest 12 months; percentiles and ranks are within this group",
    "opportunity_score": "0-100 weighted sum of national percentiles: " + ", ".join(
        f"{c} {w:.1%}" for c, w in config.WEIGHTS.items()),
    "bottleneck_classification": f"strain at or above the {config.STRAIN_HIGH_PCT * 100:.0f}th "
                                 "percentile is high; labels: Terminal-constrained, "
                                 "Airside-constrained, Both, Unconstrained",
    "unmet_demand": f"load factor above the {config.LF_TARGET:.0%} target; seat gap = extra seats "
                    "needed to bring load factor down to the target (an estimate)",
    "rank_stability": "rank range when each weight moves +/-10%; 'unstable' if the rank moves",
    "long_haul": f"flights of at least {config.LONG_HAUL_MI:,} miles",
    "comparison_significance": f"relative difference under {config.SIGNIFICANCE_SIMILAR:.0%} is "
                               f"similar, over {config.SIGNIFICANCE_LARGE:.0%} is large",
}


class Issue(BaseModel):
    # Field order is the order the model writes them: reason first, then decide whether it blocks
    quote: str = Field(description="exact words copied from the answer")
    problem: str = Field(description="one sentence: what is wrong and how to fix it")
    rule: int = Field(description="number of the blocking rule (1-6) this breaks, or 0 if none")
    blocks: bool = Field(description="true only if the answer would mislead the analyst as is")

    def __str__(self) -> str:
        return f'{self.problem} ("{self.quote}")'


class Verdict(BaseModel):
    issues: list[Issue] = Field(default_factory=list,
                                description="blocking problems the writer must fix")
    notes: list[str] = Field(default_factory=list,
                             description="optional suggestions; they never block approval")

    @property
    def approved(self) -> bool:  # decided in code: no blocking issue means approved
        return not self.issues


def _plain(text: str) -> str:
    return " ".join(re.sub(r"[*_`⁰¹²³⁴⁵⁶⁷⁸⁹]", "", text).lower().split())


def _evidenced(issue: Issue, answer: str) -> bool:
    """A blocking issue must say it blocks, name a rule and quote the answer; else it's a note.
    Catches "no blocking issue remains" filed as an issue, and complaints about absent text."""
    quote = _plain(issue.quote)
    return issue.blocks and 1 <= issue.rule <= 6 and bool(quote) and quote in answer


def _related(facts: dict[str, dict], refs: list[str]) -> dict[str, dict]:
    """Every fact about the airports the answer cites, not only the cited keys: the writer saw
    all of them, so a typed "high confidence" is checkable instead of looking invented."""
    prefixes = {r.split(".", 1)[0] for r in refs}
    return {k: v for k, v in facts.items() if k in refs or k.split(".", 1)[0] in prefixes}


def review(router, *, question: str, draft: dict, facts: dict[str, dict], findings: list[dict],
           assumptions: list[str], exclude_provider: str | None,
           previous_issues: list[str] = ()) -> tuple[Verdict, str]:
    payload = {
        "question": question,
        "answer": draft["text"],
        "facts": {k: format_value(v) for k, v in _related(facts, draft["refs"]).items()},
        "findings": findings,
        "assumptions": assumptions,
        "orphan_numbers": draft["orphans"],
        "method": METHOD,
    }
    if previous_issues:  # round 2 checks the fixes instead of moving the goalposts
        payload["previous_issues"] = list(previous_issues)
    # escape < and > so untrusted text cannot close the <review_input> tag (still valid JSON)
    body = json.dumps(payload, default=str).replace("<", "\\u003c").replace(">", "\\u003e")
    messages = [SystemMessage(REVIEWER_PROMPT),
                HumanMessage(f"<review_input>\n{body}\n</review_input>")]
    verdict, provider = router.review_invoke(messages, Verdict, exclude_provider)
    # cap in code, not in the schema: a schema violation would trigger a provider fallback
    answer = _plain(draft["text"])
    weak = [str(i) for i in verdict.issues if not _evidenced(i, answer)]
    verdict.issues = [i for i in verdict.issues if _evidenced(i, answer)][:MAX_ISSUES]
    verdict.notes = (verdict.notes + weak)[:MAX_ISSUES]
    return verdict, provider
