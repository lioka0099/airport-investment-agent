"""The agent graph: agent <-> tools, then ground -> review -> finalize, every loop bounded.

    agent --tool calls--> tools --> agent          (≤ config.TOOL_BUDGET tool calls)
    agent --draft--> ground --missing refs--> agent (≤ config.MAX_GROUND_FIXES)
    ground --> review --rejected--> agent           (≤ config.MAX_REVIEW_ROUNDS reviews)
    review --> finalize
"""

import sqlite3
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, TypedDict

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    trim_messages,
)
from langchain_core.messages.utils import count_tokens_approximately
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from airport_agent import config
from airport_agent.agent import prompts
from airport_agent.agent.envelope import ToolResult
from airport_agent.agent.grounding import missing_refs, parse_answer, render
from airport_agent.agent.reviewer import review
from airport_agent.agent.tools import base_facts, run_tool
from airport_agent.llm.resilience import AllProvidersFailed
from airport_agent.security.sanitize import clean_user_input, filter_output


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    question: str
    weights: dict | None
    facts: dict[str, dict]
    findings: list[dict]
    assumptions: list[str]
    tool_data: list[dict]
    tool_calls_used: int
    ground_attempts: int
    review_rounds: int
    provider_used: str | None
    draft: dict | None
    review_issues: list[str]
    verification: dict | None
    answer: dict | None


def turn_input(question: str, weights: dict | None = None) -> dict:
    """Input for one user turn. Messages accumulate; every other field restarts."""
    question = clean_user_input(question)
    return {"messages": [HumanMessage(question)], "question": question, "weights": weights,
            "facts": {}, "findings": [], "assumptions": [], "tool_data": [],
            "tool_calls_used": 0, "ground_attempts": 0, "review_rounds": 0,
            "review_issues": [], "provider_used": None, "draft": None,
            "verification": None, "answer": None}


def run_config(thread_id: str) -> dict:
    # 12 tool rounds = 24 steps, plus ground/review/fix loops; the default limit of 25 is too low
    return {"configurable": {"thread_id": thread_id}, "recursion_limit": 60}


def make_checkpointer(path: Path = config.CHAT_DB_PATH) -> SqliteSaver:
    # Streamlit reruns on other threads, so the connection must not be thread-bound
    return SqliteSaver(sqlite3.connect(str(path), check_same_thread=False))


def _history(state: AgentState) -> list:
    """Trim earlier turns to the token budget; always keep the current turn whole."""
    messages = state["messages"]
    start = max(i for i, m in enumerate(messages)
                if isinstance(m, HumanMessage) and m.content == state["question"])
    earlier = trim_messages(messages[:start], max_tokens=config.HISTORY_MAX_TOKENS,
                            token_counter=count_tokens_approximately, strategy="last",
                            start_on="human", include_system=False)
    return [*earlier, *messages[start:]]


def build_graph(router, tools: list, checkpointer, manifest: dict):
    by_name = {t.name: t for t in tools}
    always = {k: f.model_dump() for k, f in base_facts(manifest).items()}
    period = f"{manifest['l12m_start'][:7]} to {manifest['l12m_end'][:7]}"

    def agent(state: AgentState) -> dict:
        get_stream_writer()({"type": "stage", "stage": "thinking"})
        left = config.TOOL_BUDGET - state["tool_calls_used"]
        system = SystemMessage(prompts.build_system(manifest, left))
        msg, provider = router.agent_invoke([system, *_history(state)], tools)
        if msg.tool_calls and left <= 0:
            # Drop the calls: an AIMessage with unanswered tool calls breaks the next request.
            msg = AIMessage(prompts.OVER_BUDGET_TEXT)  # lead-in text is not an answer
        return {"messages": [msg], "provider_used": provider}

    def run_tools(state: AgentState) -> dict:
        writer = get_stream_writer()
        used = state["tool_calls_used"]
        facts, findings = dict(state["facts"]), list(state["findings"])
        assumptions, tool_data = list(state["assumptions"]), list(state["tool_data"])
        replies = []
        for call in state["messages"][-1].tool_calls:
            name, args = call["name"], dict(call["args"])
            if used >= config.TOOL_BUDGET:
                result = ToolResult(error={"code": "budget_exhausted",
                                           "message": "Tool budget used up. Answer now."})
            else:
                used += 1  # every call counts, even an unknown tool name
                if name not in by_name:
                    result = ToolResult(error={"code": "unknown_tool",
                                               "message": f"There is no tool named {name}."})
                    replies.append(ToolMessage(result.to_llm_json(), tool_call_id=call["id"],
                                               name=name))
                    continue
                if name == "rank_airports" and state["weights"]:
                    args["weights"] = state["weights"]  # sidebar sliders win over the model's
                writer({"type": "tool", "tool": name, "args": args})
                result = run_tool(by_name[name], args)
                if result.error is None:
                    facts |= {k: f.model_dump() for k, f in result.facts.items()}
                    findings += result.findings
                    assumptions += [a for a in result.assumptions if a not in assumptions]
                    tool_data.append({"tool": name, "args": args, **result.to_plain()})
            replies.append(ToolMessage(result.to_llm_json(), tool_call_id=call["id"], name=name))
        return {"messages": replies, "tool_calls_used": used, "facts": facts,
                "findings": findings, "assumptions": assumptions, "tool_data": tool_data}

    def ground(state: AgentState) -> dict:
        draft = state["messages"][-1].text
        facts = always | state["facts"]
        missing = missing_refs(draft, facts)
        if missing and state["ground_attempts"] < config.MAX_GROUND_FIXES:
            get_stream_writer()({"type": "stage", "stage": "fixing"})
            return {"messages": [HumanMessage(prompts.GROUND_FIX.format(refs=", ".join(missing)))],
                    "ground_attempts": state["ground_attempts"] + 1, "draft": None}
        return {"draft": asdict(render(draft, facts, manifest))}

    def check(state: AgentState) -> dict:
        get_stream_writer()({"type": "stage", "stage": "verifying"})
        draft = state["draft"]
        try:
            verdict, reviewer = review(
                router, question=state["question"], draft=draft,
                facts=always | state["facts"], findings=state["findings"],
                assumptions=state["assumptions"], exclude_provider=state["provider_used"])
        except AllProvidersFailed:
            return {"review_rounds": state["review_rounds"] + 1,
                    "verification": {"status": "unverified", "reviewer": None,
                                     "issues": [*state["review_issues"], prompts.REVIEWER_DOWN]}}
        rounds = state["review_rounds"] + 1
        issues = verdict.issues + [f"Unknown placeholder: {r}" for r in draft["unresolved"]]
        if verdict.approved and not draft["unresolved"]:
            return {"review_rounds": rounds,
                    "verification": {"status": "verified", "reviewer": reviewer, "issues": []}}
        if rounds < config.MAX_REVIEW_ROUNDS:
            feedback = "\n".join(f"- {i}" for i in issues) or "- The answer was not approved."
            return {"review_rounds": rounds, "draft": None, "review_issues": issues,
                    "messages": [HumanMessage(prompts.REVISE.format(issues=feedback))]}
        return {"review_rounds": rounds,
                "verification": {"status": "unverified", "reviewer": reviewer, "issues": issues}}

    def finalize(state: AgentState) -> dict:
        draft = state["draft"]
        parsed = parse_answer(draft["text"])
        allowed = set(draft["allowed_urls"])
        return {"answer": {
            "tldr": filter_output(parsed.tldr, allowed),
            "body": filter_output(parsed.body, allowed),
            "follow_ups": [filter_output(f, allowed) for f in parsed.follow_ups],
            "citations": draft["citations"], "verification": state["verification"],
            "assumptions": [filter_output(a, allowed) for a in state["assumptions"]],
            "findings": state["findings"],
            "tool_data": state["tool_data"], "provider": state["provider_used"],
            "period": period,
        }}

    g = StateGraph(AgentState)
    for name, node in [("agent", agent), ("tools", run_tools), ("ground", ground),
                       ("review", check), ("finalize", finalize)]:
        g.add_node(name, node)
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", lambda s: "tools" if s["messages"][-1].tool_calls
                            else "ground", ["tools", "ground"])
    g.add_edge("tools", "agent")
    g.add_conditional_edges("ground", lambda s: "review" if s["draft"] else "agent",
                            ["review", "agent"])
    g.add_conditional_edges("review", lambda s: "finalize" if s["verification"] else "agent",
                            ["finalize", "agent"])
    g.add_edge("finalize", END)
    return g.compile(checkpointer=checkpointer)
