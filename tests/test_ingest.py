import pandas as pd
import pytest

from airport_agent.data import ingest
from airport_agent.data.store import Store
from tests.fixtures import L12M_MONTHS, build_snapshot


@pytest.fixture
def fake_sources(tmp_path):
    """Serve the synthetic fixture's tables through the fetcher interfaces."""
    src = build_snapshot(tmp_path / "src")
    t100 = pd.read_parquet(src / "t100_monthly.parquet")
    am = pd.read_parquet(src / "ontime_airport_month.parquet")
    routes = pd.read_parquet(src / "ontime_routes.parquet")
    airports = pd.read_parquet(src / "airports.parquet")

    raw_t100 = pd.DataFrame({
        "reporting_month": pd.to_datetime(t100["month"]).dt.strftime("%Y-%m-%dT00:00:00.000"),
        "origin_airport_code": t100["code"], "total_departures": t100["departures"].astype(str),
        "total_passengers": t100["passengers"].astype(str),
        "total_seats": t100["seats"].astype(str),
        "domestic_passengers": t100["domestic_passengers"].astype(str),
        "outbound_international_1": t100["intl_passengers"].astype(str),
    })
    raw_airports = pd.DataFrame({
        "type": "large_airport", "name": airports["name"], "latitude_deg": airports["lat"],
        "longitude_deg": airports["lon"], "iso_country": "US",
        "iso_region": "US-" + airports["state"], "municipality": airports["city"],
        "iata_code": airports["code"],
    })

    def fetch_ontime_month(year, month, workdir):
        if (year, month) == (2026, 1):
            raise RuntimeError("simulated download failure")
        period = pd.Period(f"{year}-{month:02d}")
        mask = pd.to_datetime(am["month"]).dt.to_period("M") == period
        rmask = pd.to_datetime(routes["month"]).dt.to_period("M") == period
        return am[mask].copy(), routes[rmask].copy()

    return dict(fetch_t100=lambda since: raw_t100, fetch_ontime_month=fetch_ontime_month,
                fetch_airports=lambda: raw_airports)


def test_run_writes_a_store_readable_snapshot(tmp_path, fake_sources):
    out = tmp_path / "snapshot"
    manifest = ingest.run(out, workdir=tmp_path / "work", **fake_sources)
    store = Store(out)
    assert manifest["l12m_end"] == L12M_MONTHS[-1].isoformat()
    assert manifest["l12m_start"] == L12M_MONTHS[0].isoformat()
    assert manifest["sources"]["ontime"]["missing_months"] == ["2026-01"]
    assert store.airport_exists("SFO")
    months = store.query("SELECT count(DISTINCT month) AS n FROM ontime_airport_month")["n"][0]
    assert months == 11


def test_failed_run_keeps_previous_snapshot(tmp_path, fake_sources):
    out = tmp_path / "snapshot"
    ingest.run(out, workdir=tmp_path / "work", **fake_sources)

    def broken():
        raise RuntimeError("OurAirports down")

    with pytest.raises(RuntimeError):
        ingest.run(out, workdir=tmp_path / "work", **{**fake_sources, "fetch_airports": broken})
    assert Store(out).airport_exists("SFO")
    assert not (tmp_path / "snapshot.tmp").exists()
