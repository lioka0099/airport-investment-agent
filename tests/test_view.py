import pytest

from airport_agent.agent.tools import build_tools, run_tool
from airport_agent.data.faa_status import FaaStatusClient
from app import view


@pytest.fixture(scope="module")
def tool_data(store):
    tools = {t.name: t for t in build_tools(store, FaaStatusClient(fetch=lambda: "<x/>"))}
    calls = [("rank_airports", {"codes": ["BOS", "BDL", "PVD"]}),
             ("compare_airports", {"codes": ["LAX", "SNA"]}),
             ("airport_profile", {"code": "SFO"}),
             ("route_mix", {"code": "ANC"}),
             ("concentration_risk", {"code": "BOS"})]
    return [{"tool": n, "args": a, **run_tool(tools[n], a).to_plain()} for n, a in calls]


def test_status_label():
    assert view.status_label({"type": "tool", "tool": "rank_airports"}) == "Scoring airports…"
    assert view.status_label({"type": "stage", "stage": "verifying"}) == (
        "Double-checking the answer…")
    assert view.status_label({"type": "tool", "tool": "mystery"}) == "Working…"


def test_md_safe_escapes_dollars():
    assert view.md_safe("costs $5 or $6") == r"costs \$5 or \$6"


@pytest.mark.parametrize("i,mark", [(0, "bar"), (1, "bar"), (2, "line"), (3, "bar"), (4, "bar")])
def test_chart_for_each_tool(tool_data, i, mark):
    c = view.chart([tool_data[i]])
    assert c is not None and c.to_dict()["mark"]["type"] == mark


def test_chart_none_without_chartable_data():
    assert view.chart([]) is None
    assert view.chart([{"tool": "live_status", "args": {}, "data": {}}]) is None


def test_tables_titles(tool_data):
    titles = [t for t, _ in view.tables(tool_data)]
    assert titles == ["Ranking", "Comparison", "SFO monthly traffic", "ANC routes",
                      "BOS carrier shares"]


def test_confidence_is_worst(tool_data):
    rows = [{"tool": "rank_airports", "data": {"ranked": [{"confidence": "high"},
                                                          {"confidence": "low"}]}}]
    assert view.confidence(rows) == "low"
    assert view.confidence([]) is None


def test_badges_and_sources():
    assert view.verification_badge({"status": "verified", "reviewer": "anthropic", "issues": []}
                                   ).startswith("✓ Verified")
    assert "⚠️" in view.verification_badge({"status": "unverified", "reviewer": None,
                                            "issues": ["x"]})
    md = view.sources_md([{"n": 1, "source": "t100", "label": "BTS T-100", "url": "https://x",
                           "period": "2025-08 to 2026-07", "retrieved_at": "2026-10-28T08:00"}])
    assert md == "¹ [BTS T-100](https://x) — 2025-08 to 2026-07, retrieved 2026-10-28"


def test_steps_md(tool_data):
    md = view.steps_md(tool_data[:1])
    assert md.startswith("1. `rank_airports`") and "BOS" in md


def test_steps_md_neutralizes_backticks_in_args():
    evil = "`![x](https://evil.example/a)`"
    md = view.steps_md([{"tool": "resolve_airports", "args": {"query": evil}}])
    code_span = md.split("with ", 1)[1]
    assert code_span.startswith("`") and code_span.endswith("`")
    assert "`" not in code_span[1:-1]


def test_badge_issues_cannot_inject_links_or_latex():
    badge = view.verification_badge({"status": "unverified", "reviewer": "x",
                                     "issues": ["see ![x](https://evil.example/a) and $5"]})
    assert "evil.example" not in badge and r"\$5" in badge


def test_md_safe_does_not_double_escape():
    assert view.md_safe(r"\$5 and $6") == r"\$5 and \$6"


def test_confidence_ignores_unknown_levels():
    data = [{"tool": "rank_airports", "args": {},
             "data": {"ranked": [{"confidence": "n/a"}, {"confidence": "medium"}]}}]
    assert view.confidence(data) == "medium"


def test_reviewer_notes_are_filtered():
    notes = view.reviewer_notes({"notes": ["add ![x](https://evil.example/a) $5"]})
    assert "evil.example" not in notes[0] and r"\$5" in notes[0]
    assert view.reviewer_notes({"status": "verified"}) == []
