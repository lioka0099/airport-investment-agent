import pandas as pd
import pytest

from airport_agent.analytics import findings, metrics, scoring


@pytest.mark.parametrize("a,b,expected", [
    (100, 105, "similar"), (100, 130, "moderate"), (20, 13, "large"),
    (0, 0, "similar"), (5, 0, "large"), (5, -5, "large"), (-10, -10.5, "similar"),
])
def test_significance(a, b, expected):
    assert findings.significance(a, b) == expected


def test_compare_findings_lax_vs_sna(store):
    m = metrics.airport_metrics(store)
    out = findings.compare_findings(m.loc[["LAX", "SNA"]], ["avg_taxi_out", "load_factor"])
    taxi = next(f for f in out if f["metric"] == "avg_taxi_out")
    assert taxi == {"type": "comparison", "metric": "avg_taxi_out", "leader": "LAX", "other": "SNA",
                    "ratio": 1.54, "significance": "large"}
    lf = next(f for f in out if f["metric"] == "load_factor")
    assert lf["significance"] == "similar"


def test_compare_skips_missing_values():
    rows = pd.DataFrame({"x": [1.0, None]}, index=["AAA", "BBB"])
    assert findings.compare_findings(rows, ["x"]) == []


def test_no_ratio_across_zero_or_negative():
    rows = pd.DataFrame({"growth": [0.04, -0.06]}, index=["SNA", "LAX"])
    (f,) = findings.compare_findings(rows, ["growth"])
    assert f["leader"] == "SNA" and f["ratio"] is None and f["significance"] == "large"


def test_rank_findings(store):
    ranked = scoring.score_airports(metrics.airport_metrics(store)).ranked
    out = findings.rank_findings(ranked)
    bos = next(f for f in out if f["type"] == "rank" and f["airport"] == "BOS")
    assert bos["position"] == 1 and bos["of"] == 7 and bos["stable"] is True
    sfo = next(f for f in out if f["type"] == "classification" and f["airport"] == "SFO")
    assert sfo["value"] == "Airside-constrained"


def test_close_call_flagged():
    ranked = pd.DataFrame({"score": [80.0, 78.0], "rank": [1, 2], "rank_min": [1, 1],
                           "rank_max": [2, 2], "stable": [False, False],
                           "classification": ["Both", "Both"]}, index=["AAA", "BBB"])
    assert {"type": "close_call", "airports": ["AAA", "BBB"]} in findings.rank_findings(ranked)


def test_dominant_cause(store):
    f = findings.dominant_cause_finding("SFO", metrics.delay_cause_mix(store, "SFO"))
    assert f["cause"] == "nas" and f["share"] == pytest.approx(0.544, abs=1e-3)
    assert findings.dominant_cause_finding("ZZZ", {}) is None
