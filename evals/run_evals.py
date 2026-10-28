"""Run eval cases on real LLMs and write a report.

    uv run python -m evals.run_evals --provider openai
    uv run python -m evals.run_evals --provider anthropic --only brief_sfo_unmet_demand
    uv run --env-file .env python -m evals.run_evals --provider anthropic --langsmith

--langsmith runs the same cases as a LangSmith experiment on a dataset, so prompt versions can
be compared side by side there; without it, the report is written locally.
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
DATASET = "airport-agent-evals"
KNOWN_KEYS = {"tools", "tools_any", "top3", "focus", "args", "refs_any", "mentions", "refuse",
              "not_contains", "spoken_offer", "spoken_numbers"}


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
    called = state["called"] if "called" in state else called_tools(state)  # LangSmith: JSON
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
    if exp.get("spoken_offer") and not answer.get("spoken", "").rstrip().endswith("?"):
        failures.append("spoken answer does not end with an offer")
    if exp.get("spoken_numbers") and not any(ch.isdigit() for ch in answer.get("spoken", "")):
        failures.append("spoken answer has no numbers")
    return failures


def ask(graph, turns: list[str], by_voice: bool = False) -> dict:
    """Play the turns in one fresh conversation; returns the final graph state."""
    from airport_agent.agent.graph import run_config, turn_input

    thread = str(uuid.uuid4())
    for question in turns:
        state = graph.invoke(turn_input(question, by_voice=by_voice), run_config(thread))
    return state


def run_case(graph, case: dict) -> dict:
    started = time.monotonic()
    try:
        state = ask(graph, case["turns"], case.get("voice", False))
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


def sync_dataset(client, cases: list[dict]) -> None:
    """One example per case. A stable id per case id, so editing cases.yaml updates the example
    instead of adding a copy, and old experiments stay linked to it."""
    # ponytail: cases removed from cases.yaml stay in the dataset; runs only pick current ids
    if not client.has_dataset(dataset_name=DATASET):
        client.create_dataset(DATASET, description="Airport agent eval cases (evals/cases.yaml)")
    existing = {e.id for e in client.list_examples(dataset_name=DATASET)}
    for c in cases:
        eid = uuid.uuid5(uuid.NAMESPACE_URL, f"airport-agent-evals/{c['id']}")
        data = {"inputs": {"id": c["id"], "turns": c["turns"], "voice": c.get("voice", False)},
                "outputs": {"expect": c.get("expect", {})}}
        if eid in existing:
            client.update_example(eid, **data)
        else:
            client.create_example(dataset_name=DATASET, example_id=eid, **data)


def grade(outputs: dict, reference_outputs: dict) -> dict:
    """LangSmith evaluator: the same deterministic checks as the local report."""
    outputs = outputs or {"answer": None}  # the target raised
    failures = check({"expect": reference_outputs["expect"]}, outputs)
    review = (outputs.get("answer") or {}).get("verification") or {}
    return {"results": [
        {"key": "passed", "score": not failures, "comment": "; ".join(failures)},
        {"key": "verified", "score": review.get("status") == "verified",
         "comment": "; ".join(review.get("issues", []))}]}


def run_langsmith(graph, cases: list[dict], provider: str) -> None:
    from langsmith import Client, evaluate

    client = Client()
    sync_dataset(client, load_cases())  # always the full set; --only limits what runs
    ids = {c["id"] for c in cases}

    def target(inputs: dict) -> dict:  # JSON only: what check() needs, no message objects
        state = ask(graph, inputs["turns"], inputs.get("voice", False))
        return {"answer": state["answer"], "draft": state.get("draft"),
                "called": called_tools(state)}

    stamp = version_stamp()
    evaluate(target, data=[e for e in client.list_examples(dataset_name=DATASET)
                           if e.inputs["id"] in ids],
             evaluators=[grade], client=client, metadata={**stamp, "provider": provider},
             experiment_prefix=f"{provider}-{stamp['prompt_version']}")


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
    parser.add_argument("--langsmith", action="store_true",
                        help="run as a LangSmith experiment instead of a local report")
    args = parser.parse_args(argv)

    store = get_store()
    chain = [(p, m) for p, m in config.AGENT_CHAIN if p == args.provider]
    graph = build_graph(LLMRouter(agent_chain=chain), build_tools(store, FaaStatusClient()),
                        InMemorySaver(), store.manifest())
    cases = [c for c in load_cases() if not args.only or c["id"] == args.only]
    if args.langsmith:
        return run_langsmith(graph, cases, args.provider)
    results = []
    for case in cases:
        result = run_case(graph, case)
        print(f"{'PASS' if result['passed'] else 'FAIL'} {case['id']} ({result['seconds']}s) "
              f"{'; '.join(result['failures'])}")
        results.append(result)
    print(f"\nReport: {write_report(results, args.provider)}")


if __name__ == "__main__":
    main()
