import pytest

from airport_agent.analytics import metrics


@pytest.fixture(scope="module")
def m(store):
    return metrics.airport_metrics(store)


def test_core_kpis(m):
    sfo = m.loc["SFO"]
    assert sfo["passengers_l12m"] == 24_000_000
    assert sfo["load_factor"] == pytest.approx(0.85, abs=1e-3)
    assert sfo["growth_yoy"] == pytest.approx(0.04, abs=1e-3)
    assert sfo["vs_2019"] == pytest.approx(24 / 27.6 - 1, abs=1e-3)
    assert sfo["avg_taxi_out"] == pytest.approx(19)
    assert sfo["avg_taxi_in"] == pytest.approx(7.2)
    assert sfo["nas_min_per_flight"] == pytest.approx(9)
    assert sfo["months_t100"] == 12


def test_seats_gap_brings_lf_to_target(m):
    sfo = m.loc["SFO"]
    expected = max(0, sfo["passengers_l12m"] / 0.80 - sfo["seats_l12m"])
    assert sfo["seats_gap"] == pytest.approx(expected)
    assert m.loc["PVD"]["seats_gap"] == 0  # LF 0.78 is already below target


def test_ontime_coverage(m):
    assert m.loc["ANC"]["ontime_coverage"] == pytest.approx(0.15, abs=0.01)
    assert m.loc["BOS"]["ontime_coverage"] == pytest.approx(0.7, abs=0.01)


def test_delay_cause_mix(store):
    mix = metrics.delay_cause_mix(store, "SFO")
    assert set(mix) == {"nas", "weather", "carrier", "late_aircraft", "security"}
    assert sum(mix.values()) == pytest.approx(1)
    assert mix["nas"] == pytest.approx(9 / 16.55, abs=1e-3)
    assert metrics.delay_cause_mix(store, "ZZZ") == {}


def test_route_mix_long_haul(store):
    rm = metrics.route_mix(store, "ANC", 2400)
    assert rm["total_departures"] == 630 * 12
    assert rm["long_haul_pct"] == pytest.approx(130 / 630, abs=1e-3)
    assert {d["dest"] for d in rm["long_haul_destinations"]} == {"ORD", "ATL", "DEN"}
    assert sum(rm["buckets"].values()) == pytest.approx(1)
    rm3000 = metrics.route_mix(store, "ANC", 3000)
    assert rm3000["long_haul_pct"] == pytest.approx(30 / 630, abs=1e-3)


def test_carrier_shares_hhi(store):
    cs = metrics.carrier_shares(store, "BOS")
    assert cs["hhi"] == pytest.approx(2603.3, abs=0.5)
    assert cs["band"] == "high"
    assert cs["top_carrier"] == "B6"


def test_monthly_trend(store):
    trend = metrics.monthly_trend(store, "SFO")
    assert list(trend.columns) == ["month", "passengers", "seats", "load_factor"]
    assert len(trend) == 36
