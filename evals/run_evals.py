"""Run eval cases on real LLMs and write a report.

    uv run python -m evals.run_evals --provider openai
    uv run python -m evals.run_evals --provider anthropic --only brief_sfo_unmet_demand
"""

import argparse
import hashlib
import json
import subprocess
import time
import uuid
from datetime import datetime
from pathlib import Path

import yaml

CASES = Path(__file__).parent / "cases.yaml"
REPORTS = Path(__file__).parent / "reports"
KNOWN_KEYS = {"tools", "tools_any", "top3", "focus", "args", "refs_any", "mentions", "refuse",
              "not_contains"}


def load_cases(path: Path = CASES) -> list[dict]:
    cases = yaml.safe_load(path.read_text())
    ids = [c["id"] for c in cases]
    assert len(ids) == len(set(ids)), "duplicate case ids"
    for c in cases:
        unknown = set(c.get("expect", {})) - KNOWN_KEYS
        assert not unknown, f"{c['id']}: unknown expect keys {unknown}"
    return cases


def called_tools(state: dict) -> list[str]:
    """Every tool the agent called in the last user turn, failed calls included. tool_data keeps
    only successful calls, so "looked up XQZ, got unknown airport" would look like no lookup."""
    msgs = state["messages"]
    start = max(i for i, m in enumerate(msgs)
                if m.type == "human" and m.content == state["question"])
    return [c["name"] for m in msgs[start:] if m.type == "ai" for c in m.tool_calls]


def check(case: dict, state: dict) -> list[str]:
    answer = state.get("answer")
    if not answer:
        return ["no answer"]
    exp, failures = case.get("expect", {}), []
    called = called_tools(state)
    last = {t["tool"]: t for t in answer["tool_data"]}  # last call of each tool
    text = f"{answer['tldr']}\n{answer['body']}"
    draft = state.get("draft") or {}

    if "{{" in text or draft.get("unresolved"):
        failures.append(f"grounding: unresolved {draft.get('unresolved')}")
    if missing := [t for t in exp.get("tools", []) if t not in called]:
        failures.append(f"tools not called: {missing}")
    if exp.get("tools_any") and not set(exp["tools_any"]) & set(called):
        failures.append(f"none of {exp['tools_any']} called")
    if exp.get("refuse") and called:
        failures.append(f"expected refusal, but called {called}")
    rank = last.get("rank_airports")
    if "top3" in exp:
        top = [r["code"] for r in (rank["data"].get("ranked", []) if rank else [])][:3]
        if not set(exp["top3"]) & set(top):
            failures.append(f"top3 {top} has none of {exp['top3']}")
    if "focus" in exp and (rank or {}).get("args", {}).get("focus") != exp["focus"]:
        failures.append(f"focus is not {exp['focus']}")
    for tool, keys in exp.get("args", {}).items():
        got = (last.get(tool) or {}).get("args", {})
        if missing := [k for k in keys if not got.get(k)]:
            failures.append(f"{tool} missing args {missing}")
    if exp.get("refs_any") and not set(exp["refs_any"]) & set(draft.get("refs", [])):
        failures.append(f"none of {exp['refs_any']} cited")
    lower = text.lower()
    for item in exp.get("mentions", []):
        options = item if isinstance(item, list) else [item]
        if not any(o.lower() in lower for o in options):
            failures.append(f"does not mention {options}")
    full = text + "\n" + "\n".join(answer["follow_ups"])
    if leaked := [s for s in exp.get("not_contains", []) if s in full]:
        failures.append(f"contains {leaked}")
    return failures


def run_case(graph, case: dict) -> dict:
    from airport_agent.agent.graph import run_config, turn_input

    thread, started = str(uuid.uuid4()), time.monotonic()
    try:
        for question in case["turns"]:
            state = graph.invoke(turn_input(question), run_config(thread))
        failures = check(case, state)
        answer = state["answer"] or {}
        review = answer.get("verification") or {}
        verification, issues = review.get("status"), review.get("issues", [])
        tools = called_tools(state)
        text = f"{answer.get('tldr', '')}\n{answer.get('body', '')}"  # JSON only, for triage
    except Exception as exc:  # one broken case must not stop the run
        failures, verification, tools = [f"error: {type(exc).__name__}: {exc}"], None, []
        issues = []
        text = ""
    return {"id": case["id"], "passed": not failures, "failures": failures, "tools": tools,
            "answer": text, "verification": verification, "review_issues": issues,
            "seconds": round(time.monotonic() - started, 1)}


def version_stamp() -> dict:
    """What produced this report: the prompt text and the code. A dirty tree means uncommitted
    changes, so the SHA alone would not reproduce the run; the prompt hash still identifies it."""
    from airport_agent.agent import prompts

    def git(*args: str) -> str:
        return subprocess.run(["git", *args], capture_output=True, text=True).stdout.strip()

    source = Path(prompts.__file__).read_bytes()
    return {"prompt_version": hashlib.sha256(source).hexdigest()[:12],
            "git_sha": git("rev-parse", "--short", "HEAD") or "unknown",
            "dirty": bool(git("status", "--porcelain", "--untracked-files=no"))}


def write_report(results: list[dict], provider: str) -> Path:
    REPORTS.mkdir(exist_ok=True)
    # time sorts the reports; the random suffix keeps parallel runs from overwriting each other
    stem = REPORTS / f"{datetime.now():%Y%m%d-%H%M%S}-{provider}-{uuid.uuid4().hex[:8]}"
    passed = sum(r["passed"] for r in results)
    verified = sum(r["verification"] == "verified" for r in results)
    mean_s = sum(r["seconds"] for r in results) / max(len(results), 1)
    stamp = version_stamp()
    lines = [f"# Eval report — agent provider: {provider}", "",
             f"prompt_version `{stamp['prompt_version']}` · git `{stamp['git_sha']}`"
             f"{' (uncommitted changes)' if stamp['dirty'] else ''}", "",
             f"Passed {passed}/{len(results)} · verified {verified}/{len(results)} · "
             f"mean {mean_s:.1f}s per case", "",
             "| case | pass | verified | seconds | tools | failures |", "|---|---|---|---|---|---|"]
    for r in results:
        lines.append(f"| {r['id']} | {'✅' if r['passed'] else '❌'} | {r['verification']} | "
                     f"{r['seconds']} | {', '.join(r['tools'])} | {'; '.join(r['failures'])} |")
    if unverified := [r for r in results if r["review_issues"]]:
        lines += ["", "## Why the reviewer did not verify"]
        for r in unverified:
            lines += ["", f"**{r['id']}**", *[f"- {i}" for i in r["review_issues"]]]
    stem.with_suffix(".md").write_text("\n".join(lines) + "\n")
    stem.with_suffix(".json").write_text(json.dumps({**stamp, "provider": provider,
                                                     "results": results}, indent=2))
    return stem.with_suffix(".md")


def main(argv=None) -> None:
    from langgraph.checkpoint.memory import InMemorySaver

    from airport_agent import config
    from airport_agent.agent.graph import build_graph
    from airport_agent.agent.tools import build_tools
    from airport_agent.data.faa_status import FaaStatusClient
    from airport_agent.data.store import get_store
    from airport_agent.llm.router import LLMRouter

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=[p for p, _ in config.AGENT_CHAIN], required=True)
    parser.add_argument("--only", help="run a single case id")
    args = parser.parse_args(argv)

    store = get_store()
    chain = [(p, m) for p, m in config.AGENT_CHAIN if p == args.provider]
    graph = build_graph(LLMRouter(agent_chain=chain), build_tools(store, FaaStatusClient()),
                        InMemorySaver(), store.manifest())
    cases = [c for c in load_cases() if not args.only or c["id"] == args.only]
    results = []
    for case in cases:
        result = run_case(graph, case)
        print(f"{'PASS' if result['passed'] else 'FAIL'} {case['id']} ({result['seconds']}s) "
              f"{'; '.join(result['failures'])}")
        results.append(result)
    print(f"\nReport: {write_report(results, args.provider)}")


if __name__ == "__main__":
    main()
