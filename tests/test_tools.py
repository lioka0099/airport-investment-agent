import json

import pytest
from langchain_core.tools import StructuredTool

from airport_agent.agent.tools import base_facts, build_tools, run_tool
from airport_agent.data.faa_status import FaaStatusClient
from tests.test_faa_status import SAMPLE


@pytest.fixture(scope="module")
def tools(store):
    return {t.name: t for t in build_tools(store, FaaStatusClient(fetch=lambda: SAMPLE))}


def call(tools, name, **args):
    return run_tool(tools[name], args)


def test_seven_tools(tools):
    assert set(tools) == {"resolve_airports", "rank_airports", "compare_airports",
                          "airport_profile", "route_mix", "concentration_risk", "live_status"}


@pytest.mark.parametrize("query,codes", [
    ("New England", ["BOS", "BDL", "PVD"]), ("California", ["LAX", "SFO", "SNA"]),
    ("LA", ["LAX"]), ("Santa Ana", ["SNA"]), ("anchorage", ["ANC"]), ("sfo", ["SFO"]),
    ("Logan", ["BOS"]),
])
def test_resolve(tools, query, codes):
    r = call(tools, "resolve_airports", query=query)
    assert [m["code"] for m in r.data["matches"]] == codes


def test_resolve_la_states_assumption_and_alternatives(tools):
    r = call(tools, "resolve_airports", query="LA")
    assert r.data["alternatives"] == ["SNA"]
    assert any("LAX" in a for a in r.assumptions)


def test_resolve_region_lists_small_airports(tools):
    r = call(tools, "resolve_airports", query="New England")
    assert r.data["small_airports"] == ["BTV"]


def test_resolve_custom_region_uses_llm_states(tools):
    r = call(tools, "resolve_airports", query="Great Lakes", states=["ma", "ct"])
    assert [m["code"] for m in r.data["matches"]] == ["BOS", "BDL"]
    assert any("not a built-in region" in a for a in r.assumptions)


def test_resolve_builtin_region_ignores_llm_states(tools):
    r = call(tools, "resolve_airports", query="New England", states=["CA"])
    assert r.data["small_airports"] == ["BTV"]


def test_resolve_rejects_bad_state_codes(tools):
    r = call(tools, "resolve_airports", query="Great Lakes", states=["XX"])
    assert r.error["code"] == "invalid_arguments"


def test_resolve_not_found(tools):
    assert call(tools, "resolve_airports", query="Atlantis").error["code"] == "not_found"


def test_rank_terminal_focus(tools):
    r = call(tools, "rank_airports", codes=["BOS", "BDL", "PVD", "BTV"], focus="terminal")
    assert [row["code"] for row in r.data["ranked"]] == ["BOS", "BDL"]
    assert "BOS.score" not in r.facts and "BOS.rank" not in r.facts
    assert r.facts["BOS.score_terminal"].unit == "score"
    assert r.facts["BOS.rank_terminal"].value == 1
    assert r.facts["BOS.rank_terminal_min"].unit == "rank"
    assert (r.facts["BOS.rank_terminal_min"].value <= r.facts["BOS.rank_terminal"].value
            <= r.facts["BOS.rank_terminal_max"].value)
    assert r.facts["BOS.terminal_pctl"].unit == "pctl"
    assert {e["code"] for e in r.data["excluded"]} == {"BTV", "PVD"}
    assert any(f["type"] == "rank" and f["airport"] == "BOS" and f["position"] == 1
               for f in r.findings)
    assert "WEIGHTS.demand" in r.facts


def test_rank_lowercase_codes_are_normalized(tools):
    assert call(tools, "rank_airports", codes=["bos", "sfo"]).error is None


def test_rank_invalid_code_returns_error_not_exception(tools):
    r = call(tools, "rank_airports", codes=["B0S"])
    assert r.error["code"] == "invalid_arguments"


def test_rank_bad_weights(tools):
    r = call(tools, "rank_airports", weights={"price": 1.0})
    assert r.error["code"] == "invalid_arguments"


def test_compare_lax_sna(tools):
    r = call(tools, "compare_airports", codes=["LAX", "SNA"])
    taxi = next(f for f in r.findings if f.get("metric") == "avg_taxi_out")
    assert taxi["leader"] == "LAX" and taxi["significance"] == "large"
    assert r.facts["LAX.avg_taxi_out"].value == pytest.approx(20)
    assert r.facts["LAX.avg_taxi_out_ratio_vs_sna"].value == pytest.approx(1.54)
    assert any("ongestion" in a for a in r.assumptions)


def test_compare_unknown_airport(tools):
    r = call(tools, "compare_airports", codes=["LAX", "LXA"])
    assert r.error["code"] == "unknown_airport"
    assert "LAX" in r.error["suggestions"]["LXA"]


def test_compare_needs_two(tools):
    assert call(tools, "compare_airports", codes=["LAX"]).error["code"] == "invalid_arguments"


def test_profile_sfo_unmet_demand(tools):
    r = call(tools, "airport_profile", code="SFO")
    assert r.facts["SFO.seats_gap"].value > 0
    assert r.facts["SFO.classification"].value == "Airside-constrained"
    assert r.facts["SFO.cause_nas_share"].value == pytest.approx(0.544, abs=1e-3)
    assert "SFO.national_rank" in r.facts
    assert any(f["type"] == "dominant_cause" and f["cause"] == "nas" for f in r.findings)
    assert len(r.data["trend"]) == 36


def test_route_mix_anc(tools):
    r = call(tools, "route_mix", code="ANC")
    assert r.facts["ANC.long_haul_pct"].value == pytest.approx(130 / 630, abs=1e-3)
    assert r.facts["ANC.long_haul_threshold_mi"].value == 2400
    assert "ANC.share_2400_plus" in r.facts
    assert any("domestic" in a.lower() for a in r.assumptions)


def test_route_mix_threshold_range(tools):
    r = call(tools, "route_mix", code="ANC", long_haul_threshold_mi=50)
    assert r.error["code"] == "invalid_arguments"


def test_rank_unknown_code_returns_suggestion(tools):
    r = call(tools, "rank_airports", codes=["LXA", "SFO"])
    assert r.error["code"] == "unknown_airport"
    assert "LAX" in r.error["suggestions"]["LXA"]


def test_compare_duplicate_codes_invalid(tools):
    r = call(tools, "compare_airports", codes=["LAX", "lax"])
    assert r.error["code"] == "invalid_arguments"


def test_run_tool_converts_exceptions():
    def boom(code: str) -> str:
        raise RuntimeError("x")

    tool = StructuredTool.from_function(func=boom, name="boom", description="d")
    assert run_tool(tool, {"code": "LAX"}).error["code"] == "tool_failed"


def test_concentration_bos(tools):
    r = call(tools, "concentration_risk", code="BOS")
    assert r.facts["BOS.hhi"].value == pytest.approx(2603.3, abs=0.5)
    assert r.facts["BOS.concentration_band"].value == "high"
    assert "BOS.share_b6" in r.facts


def test_live_status_no_free_text(tools):
    r = call(tools, "live_status", codes=["SNA", "SFO", "SEA"])
    assert "SNA.live_status" in r.facts
    blob = r.to_llm_json()
    assert "IGNORE" not in blob and "CLSD" not in blob
    assert r.data["airports"]["SEA"] == []


def test_base_facts(store):
    f = base_facts(store.manifest())
    assert f["CONFIG.lf_target"].value == 0.80
    assert f["DATA.period_end"].value == "2026-07"
    assert set(f) >= {"CONFIG.weight_demand", "CONFIG.peer_min_passengers", "DATA.period_start"}


def test_all_tool_payloads_are_json(tools):
    payload = json.loads(call(tools, "rank_airports").to_llm_json())
    assert payload["error"] is None


def test_profile_score_is_national_and_distinct_from_rank_score(tools):
    p = call(tools, "airport_profile", code="BOS")
    assert "BOS.score" not in p.facts and p.facts["BOS.national_score"].unit == "score"
    assert "BOS.national_rank" in p.facts
    lo, hi = p.facts["BOS.national_rank_min"].value, p.facts["BOS.national_rank_max"].value
    assert lo <= p.facts["BOS.national_rank"].value <= hi


def test_rank_reports_subset_size(tools):
    r = call(tools, "rank_airports", codes=["BOS", "BDL", "PVD"], focus="terminal")
    assert 1 <= r.facts["RANK.terminal_count"].value <= 3
