import httpx
import openai
import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from airport_agent import config
from airport_agent.agent.graph import build_graph, make_checkpointer, run_config, turn_input
from airport_agent.agent.reviewer import Issue, Verdict
from airport_agent.agent.tools import build_tools
from airport_agent.data.faa_status import FaaStatusClient
from airport_agent.llm.resilience import AllProvidersFailed
from airport_agent.security.sanitize import InputRejected
from tests.test_faa_status import SAMPLE
from tests.test_router import REQ, router


def issue(problem: str) -> Issue:
    return Issue(rule=1, blocks=True, quote="It is busy", problem=problem)  # quotes FINAL


ERR = openai.APIStatusError("x", response=httpx.Response(503, request=REQ), body=None)
FINAL = ("TL;DR: {{BOS.name}} leads with a score of {{BOS.score_terminal}}.\n\nIt is busy. Data "
         "{{DATA.period_start}} to {{DATA.period_end}}. Assumptions: peers.\n\n"
         "FOLLOW-UPS:\n- Why BOS?\n- Compare BOS and BDL")
OFF_TOPIC = ("TL;DR: I only cover US airport investment questions.\n\nTry a ranking.\n\n"
             "FOLLOW-UPS:\n- Rank New England airports\n- Profile SFO")


def calls(*specs):
    return AIMessage("", tool_calls=[{"name": n, "args": a, "id": f"c{i}"}
                                     for i, (n, a) in enumerate(specs)])


RESOLVE = calls(("resolve_airports", {"query": "New England"}))
RANK = calls(("rank_airports", {"codes": ["BOS", "BDL", "PVD"], "focus": "terminal"}))


@pytest.fixture(scope="module")
def tools(store):
    return build_tools(store, FaaStatusClient(fetch=lambda: SAMPLE))


def run(store, tools, outcomes, question="Terminal candidates in New England?", saver=None,
        thread="t1", weights=None):
    graph = build_graph(router(outcomes), tools, saver or InMemorySaver(), store.manifest())
    return graph, graph.invoke(turn_input(question, weights), run_config(thread))


def test_happy_path_is_grounded_cited_and_verified(store, tools):
    _, s = run(store, tools, {"openai": [RESOLVE, RANK, AIMessage(FINAL)],
                              "anthropic": [Verdict(approved=True)]})
    a = s["answer"]
    assert f"{s['facts']['BOS.score_terminal']['value']:.1f}" in a["tldr"]
    assert "{{" not in a["tldr"] + a["body"]
    assert a["verification"] == {"status": "verified", "reviewer": "anthropic", "issues": [],
                                 "notes": []}
    assert a["follow_ups"] == ["Why BOS?", "Compare BOS and BDL"]
    assert {c["source"] for c in a["citations"]} >= {"computed", "t100"}
    assert [t["tool"] for t in a["tool_data"]] == ["resolve_airports", "rank_airports"]
    assert s["tool_calls_used"] == 2 and a["provider"] == "openai"
    assert a["period"] == "2025-08 to 2026-07"


def test_stream_emits_tool_and_stage_events(store, tools):
    graph = build_graph(router({"openai": [RANK, AIMessage(FINAL)],
                                "anthropic": [Verdict(approved=True)]}),
                        tools, InMemorySaver(), store.manifest())
    custom = [e for mode, e in graph.stream(turn_input("q"), run_config("s"),
                                            stream_mode=["updates", "custom"]) if mode == "custom"]
    assert {"type": "tool", "tool": "rank_airports",
            "args": {"codes": ["BOS", "BDL", "PVD"], "focus": "terminal"}} in custom
    assert {"type": "stage", "stage": "verifying"} in custom


def test_unknown_placeholder_gets_one_fix(store, tools):
    bad = FINAL.replace("{{BOS.score_terminal}}", "{{BOS.made_up}}")
    _, s = run(store, tools, {"openai": [RANK, AIMessage(bad), AIMessage(FINAL)],
                              "anthropic": [Verdict(approved=True)]})
    fixes = [m for m in s["messages"] if isinstance(m, HumanMessage) and "BOS.made_up" in m.content]
    assert len(fixes) == 1 and s["ground_attempts"] == 1
    assert s["answer"]["verification"]["status"] == "verified"


def test_reviewer_rejects_twice_gives_unverified(store, tools):
    _, s = run(store, tools, {"openai": [RANK, AIMessage(FINAL), AIMessage(FINAL)],
                              "anthropic": [Verdict(issues=[issue("too strong")]),
                                            Verdict(issues=[issue("still too strong")])]})
    assert s["review_rounds"] == 2
    assert s["answer"]["verification"] == {"status": "unverified", "reviewer": "anthropic",
                                           "issues": ['still too strong ("It is busy")'],
                                           "notes": []}


def test_agent_falls_back_and_reviewer_uses_another_provider(store, tools):
    _, s = run(store, tools, {"openai": [ERR, ERR, ERR], "anthropic": [AIMessage(OFF_TOPIC)],
                              "google": [Verdict(approved=True)]}, question="Best pizza?")
    assert s["answer"]["provider"] == "anthropic"
    assert s["answer"]["verification"]["reviewer"] == "google"


def test_all_agent_providers_down_raises(store, tools):
    with pytest.raises(AllProvidersFailed):
        run(store, tools, {"openai": [ERR, ERR, ERR], "anthropic": [ERR, ERR, ERR]})


def test_reviewer_down_gives_unverified_answer(store, tools):
    _, s = run(store, tools, {"openai": [AIMessage(OFF_TOPIC)], "anthropic": [ERR, ERR, ERR],
                              "google": [ERR, ERR, ERR]})
    v = s["answer"]["verification"]
    assert v["status"] == "unverified" and v["reviewer"] is None
    assert "unavailable" in v["issues"][0]


def test_tool_budget_is_enforced(store, tools, monkeypatch):
    monkeypatch.setattr(config, "TOOL_BUDGET", 2)
    _, s = run(store, tools, {"openai": [RESOLVE, RANK, RANK],
                              "anthropic": [Verdict(approved=True)]})
    assert s["tool_calls_used"] == 2
    assert s["answer"]["tldr"].startswith("I ran out of analysis steps")
    assert not s["messages"][-1].tool_calls


def test_unknown_tool_names_use_up_the_budget(store, tools, monkeypatch):
    monkeypatch.setattr(config, "TOOL_BUDGET", 2)
    fakes = [calls(("made_up_tool", {})) for _ in range(10)]
    _, s = run(store, tools, {"openai": fakes, "anthropic": [Verdict(approved=True)]})
    assert s["tool_calls_used"] == 2
    assert s["answer"]["tldr"].startswith("I ran out of analysis steps")


def test_assumptions_are_filtered_like_the_body(store, tools):
    evil = calls(("resolve_airports", {"query": "![x](https://evil.example/a)", "states": ["MA"]}))
    _, s = run(store, tools, {"openai": [evil, RANK, AIMessage(FINAL)],
                              "anthropic": [Verdict(approved=True)]})
    assert any("evil.example" in a for a in s["assumptions"])
    assert not any("evil.example" in a for a in s["answer"]["assumptions"])


def test_text_next_to_dropped_tool_calls_is_not_the_draft(store, tools, monkeypatch):
    monkeypatch.setattr(config, "TOOL_BUDGET", 1)
    lead_in = AIMessage("Let me check BDL too.", tool_calls=RANK.tool_calls)
    _, s = run(store, tools, {"openai": [RANK, lead_in], "anthropic": [Verdict(approved=True)]})
    assert s["answer"]["tldr"].startswith("I ran out of analysis steps")


def test_reviewer_down_keeps_earlier_issues(store, tools):
    _, s = run(store, tools, {"openai": [RANK, AIMessage(FINAL), AIMessage(FINAL)],
                              "anthropic": [Verdict(issues=[issue("too strong")]),
                                            ERR, ERR, ERR],
                              "google": [ERR, ERR, ERR]})
    v = s["answer"]["verification"]
    assert v["status"] == "unverified" and v["reviewer"] is None
    assert v["issues"][0] == 'too strong ("It is busy")' and "unavailable" in v["issues"][-1]
    assert s["review_rounds"] == 2


def test_user_weights_win_over_model_weights(store, tools):
    w = {"demand": 0.2, "growth": 0.5, "terminal": 0.1, "airside": 0.1, "scale": 0.1}
    model = calls(("rank_airports", {"codes": ["BOS", "BDL"], "focus": "terminal",
                                     "weights": {"demand": 1.0}}))
    _, s = run(store, tools, {"openai": [model, AIMessage(FINAL)],
                              "anthropic": [Verdict(approved=True)]}, weights=w)
    assert s["tool_data"][0]["args"]["weights"] == w


def test_sidebar_weights_are_injected_into_rank(store, tools):
    w = {"demand": 0.2, "growth": 0.5, "terminal": 0.1, "airside": 0.1, "scale": 0.1}
    _, s = run(store, tools, {"openai": [RANK, AIMessage(FINAL)],
                              "anthropic": [Verdict(approved=True)]}, weights=w)
    assert s["tool_data"][0]["args"]["weights"] == w


def test_memory_across_turns_with_sqlite_and_per_turn_reset(store, tools, tmp_path):
    saver = make_checkpointer(tmp_path / "chat.sqlite")
    outcomes = {"openai": [RANK, AIMessage(FINAL), AIMessage(OFF_TOPIC)],
                "anthropic": [Verdict(approved=True), Verdict(approved=True)]}
    graph = build_graph(router(outcomes), tools, saver, store.manifest())
    graph.invoke(turn_input("Terminal candidates in New England?"), run_config("t"))
    s = graph.invoke(turn_input("And pizza?"), run_config("t"))
    questions = [m.content for m in s["messages"] if isinstance(m, HumanMessage)]
    assert questions == ["Terminal candidates in New England?", "And pizza?"]
    assert s["tool_data"] == [] and s["tool_calls_used"] == 0 and s["review_rounds"] == 1


def test_turn_input_sanitizes():
    with pytest.raises(InputRejected):
        turn_input("   ")
    assert turn_input("hi\u200b there")["question"] == "hi there"


def test_profile_then_rank_keep_distinct_scores(store, tools):
    profile = calls(("airport_profile", {"code": "BOS"}))
    _, s = run(store, tools, {"openai": [profile, RANK, AIMessage(FINAL)],
                              "anthropic": [Verdict(approved=True)]})
    f = s["facts"]
    # the fixture ranks BOS #1 both ways; what matters is that neither call overwrote the other
    assert {"BOS.national_score", "BOS.national_rank", "BOS.score_terminal",
            "BOS.rank_terminal"} <= f.keys()
    assert "BOS.score" not in f


def test_reviewer_notes_do_not_block_and_round_two_sees_earlier_issues(store, tools):
    spy = {"openai": [RANK, AIMessage(FINAL), AIMessage(FINAL)],
           "anthropic": [Verdict(issues=[issue("score called large but finding says similar")]),
                         Verdict(notes=["could add the percentile"])]}
    _, s = run(store, tools, spy)
    assert s["answer"]["verification"] == {"status": "verified", "reviewer": "anthropic",
                                           "issues": [], "notes": ["could add the percentile"]}
