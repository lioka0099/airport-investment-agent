import numpy as np
import pytest

from airport_agent.analytics import metrics, scoring


@pytest.fixture(scope="module")
def m(store):
    return metrics.airport_metrics(store)


def test_normalize_weights_partial_override():
    w = scoring.normalize_weights({"scale": 0.0})
    assert w["scale"] == 0
    assert sum(w.values()) == pytest.approx(1)
    assert w["demand"] == pytest.approx(0.35 / 0.85)


@pytest.mark.parametrize(
    "bad", [{"price": 1}, {"demand": -1}, dict.fromkeys(scoring.COMPONENTS, 0)]
)
def test_normalize_weights_rejects_bad_input(bad):
    with pytest.raises(ValueError):
        scoring.normalize_weights(bad)


@pytest.mark.parametrize("t,a,expected", [
    (0.9, 0.9, "Both"), (0.7, 0.1, "Terminal-constrained"), (0.1, 0.66, "Airside-constrained"),
    (0.5, 0.5, "Unconstrained"), (None, 0.9, "Unknown (no On-Time data)"),
    (np.nan, np.nan, "Unknown (no On-Time data)"),
])
def test_classify(t, a, expected):
    assert scoring.classify(t, a) == expected


def test_confidence():
    assert scoring.confidence(12, 0.6) == "high"
    assert scoring.confidence(12, 0.3) == "medium"
    assert scoring.confidence(12, 0.1) == "low"
    assert scoring.confidence(9, 0.9) == "medium"
    assert scoring.confidence(12, None) == "low"


def test_fixture_classes(m):
    r = scoring.score_airports(m).ranked
    assert r.loc["BOS", "classification"] == "Both"
    assert r.loc["LAX", "classification"] == "Both"
    assert r.loc["BDL", "classification"] == "Terminal-constrained"
    assert r.loc["SFO", "classification"] == "Airside-constrained"
    for code in ("PVD", "SNA", "ANC"):
        assert r.loc[code, "classification"] == "Unconstrained"
    assert r.loc["ANC", "confidence"] == "low"
    assert "BTV" not in r.index


def test_overall_rank_and_stability(m):
    r = scoring.score_airports(m).ranked
    assert r.index[0] == "BOS"
    assert r.loc["BOS", "score"] == pytest.approx(83.2, abs=0.5)
    assert r.loc["BOS", "rank"] == 1 and bool(r.loc["BOS", "stable"])
    assert list(r["rank"]) == sorted(r["rank"])


def test_new_england_terminal_focus(m):
    ranking = scoring.score_airports(m, ["BOS", "BDL", "PVD", "BTV"], focus="terminal")
    assert list(ranking.ranked.index) == ["BOS", "BDL"]
    reasons = {e["code"]: e["reason"] for e in ranking.excluded}
    assert "peer threshold" in reasons["BTV"]
    assert "Unconstrained" in reasons["PVD"]


def test_weight_override_changes_ranking(m):
    only_scale = dict.fromkeys(scoring.COMPONENTS, 0) | {"scale": 1}
    r = scoring.score_airports(m, weights=only_scale).ranked
    assert r.index[0] == "LAX"


def test_missing_ontime_redistributes_weights(m):
    m2 = m.copy()
    m2.loc["SNA", ["avg_taxi_in", "avg_taxi_out", "nas_min_per_flight", "ontime_coverage"]] = np.nan
    r = scoring.score_airports(m2).ranked
    assert r.loc["SNA", "classification"] == "Unknown (no On-Time data)"
    assert r.loc["SNA", "confidence"] == "low"
    assert 0 <= r.loc["SNA", "score"] <= 100


def test_unknown_code_is_excluded(m):
    ranking = scoring.score_airports(m, ["BOS", "ZZZ"])
    assert list(ranking.ranked.index) == ["BOS"]
    assert ranking.excluded[0]["code"] == "ZZZ"


def test_invalid_focus(m):
    with pytest.raises(ValueError):
        scoring.score_airports(m, focus="runway")


def test_base_rank_within_sensitivity_range(m):
    m2 = m.copy()
    m2.loc["SNA", ["avg_taxi_in", "avg_taxi_out", "nas_min_per_flight", "ontime_coverage"]] = np.nan
    r = scoring.score_airports(m2).ranked
    assert ((r["rank_min"] <= r["rank"]) & (r["rank"] <= r["rank_max"])).all()


def test_ranked_is_sorted_best_first(m):
    # Contract: findings.rank_findings reads ranked.iloc[0] / iloc[1] as the top two.
    ranked = scoring.score_airports(m).ranked
    assert ranked["score"].is_monotonic_decreasing
    assert ranked["rank"].is_monotonic_increasing
