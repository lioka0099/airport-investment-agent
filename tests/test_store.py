import pytest

from airport_agent.data.store import Store


def test_query_with_params(store):
    df = store.query("SELECT name FROM airports WHERE code = ?", ["SFO"])
    assert df["name"].iloc[0] == "San Francisco Intl"


def test_airport_exists(store):
    assert store.airport_exists("BOS")
    assert not store.airport_exists("ZZZ")


def test_manifest_has_l12m_window(store):
    assert store.manifest()["l12m_end"] == "2026-07-01"


def test_missing_snapshot_gives_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="airport_agent.data.ingest"):
        Store(tmp_path)


def test_all_views_exist(store):
    for view in ("airports", "t100_monthly", "ontime_airport_month", "ontime_routes"):
        assert len(store.query(f"SELECT * FROM {view} LIMIT 1")) == 1
