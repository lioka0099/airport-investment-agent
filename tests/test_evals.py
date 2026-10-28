from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from evals.run_evals import check, load_cases


def state(tools=(), tldr="BOS leads.", body="Body.", refs=("BOS.score",), unresolved=(),
          ranked=("BOS", "BDL", "PVD", "BTV"), focus="terminal"):
    tool_data = []
    for name in tools:
        args = {"focus": focus, "weights": {"growth": 0.5}} if name == "rank_airports" else {}
        data = {"ranked": [{"code": c} for c in ranked]} if name == "rank_airports" else {}
        tool_data.append({"tool": name, "args": args, "data": data})
    calls = [{"name": t, "args": {}, "id": str(i)} for i, t in enumerate(tools)]
    return {"answer": {"tldr": tldr, "body": body, "follow_ups": [], "tool_data": tool_data},
            "question": "q", "messages": [HumanMessage("q"), AIMessage("", tool_calls=calls)],
            "draft": {"refs": list(refs), "unresolved": list(unresolved)}}


def test_cases_file_is_valid():
    cases = load_cases()
    assert len(cases) >= 20 and len({c["id"] for c in cases}) == len(cases)


def test_passing_case():
    case = {"id": "x", "expect": {"tools": ["rank_airports"], "top3": ["BOS"], "focus": "terminal",
                                  "mentions": [["BOS", "Boston"]], "refs_any": ["BOS.score"],
                                  "args": {"rank_airports": ["weights"]}}}
    assert check(case, state(tools=["rank_airports"])) == []


def test_each_check_can_fail():
    case = {"id": "x", "expect": {"tools": ["route_mix"], "top3": ["SFO"], "focus": "airside",
                                  "mentions": ["Anchorage"], "refs_any": ["ANC.long_haul_pct"],
                                  "not_contains": ["leads"], "tools_any": ["live_status"]}}
    failures = check(case, state(tools=["rank_airports"]))
    assert len(failures) == 7


def test_refuse_and_grounding():
    assert check({"id": "x", "expect": {"refuse": True}}, state()) == []
    assert check({"id": "x", "expect": {"refuse": True}}, state(tools=["rank_airports"]))
    failures = check({"id": "x", "expect": {}}, state(tldr="{{BOS.x}}", unresolved=["BOS.x"]))
    assert any("ground" in f for f in failures)


def test_missing_answer_is_a_failure():
    assert check({"id": "x", "expect": {}}, {"answer": None}) == ["no answer"]


def test_version_stamp_identifies_prompt_and_code():
    from evals.run_evals import version_stamp
    s = version_stamp()
    assert len(s["prompt_version"]) == 12 and s["git_sha"] and isinstance(s["dirty"], bool)


def test_failed_tool_calls_count_as_called():
    msgs = [HumanMessage("Rank MA"), AIMessage("", tool_calls=[{"name": "rank_airports", "args": {},
                                                                "id": "1"}]),
            HumanMessage("Profile XQZ"),
            AIMessage("", tool_calls=[{"name": "airport_profile", "args": {}, "id": "2"}]),
            ToolMessage('{"error": "unknown airport"}', tool_call_id="2"), AIMessage("Not found.")]
    s = state() | {"messages": msgs, "question": "Profile XQZ"}
    assert check({"id": "x", "expect": {"tools": ["airport_profile"]}}, s) == []
    assert check({"id": "x", "expect": {"refuse": True}}, s)  # an earlier turn's call is ignored


def test_report_keeps_the_reviewers_reasons(tmp_path, monkeypatch):
    import json

    from evals import run_evals
    monkeypatch.setattr(run_evals, "REPORTS", tmp_path)
    results = [{"id": "pizza", "passed": True, "failures": [], "tools": [], "answer": "",
                "verification": "unverified", "review_issues": ["No period stated"],
                "seconds": 1.0}]
    md = run_evals.write_report(results, "anthropic")
    assert "**pizza**" in md.read_text() and "- No period stated" in md.read_text()
    saved = json.loads(md.with_suffix(".json").read_text())
    assert saved["results"][0]["review_issues"] == ["No period stated"]
