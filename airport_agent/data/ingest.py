"""Offline ingest: pull public sources, validate, and write snapshot/ atomically.

Run: uv run python -m airport_agent.data.ingest
"""

import json
import logging
import shutil
import tempfile
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd

from airport_agent import config
from airport_agent.data.sources import ontime, ourairports, t100
from airport_agent.data.store import Store

log = logging.getLogger(__name__)
T100_SINCE = date(2019, 1, 1)  # CY2019 is the pre-pandemic baseline for vs_2019


def _shift(month: date, n: int) -> date:
    idx = month.year * 12 + month.month - 1 + n
    return date(idx // 12, idx % 12 + 1, 1)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _source(rows_kept: int, rows_dropped: int, period: str, notes: str, **extra) -> dict:
    return {"period": period, "retrieved_at": _now(), "rows_kept": int(rows_kept),
            "rows_dropped": int(rows_dropped), "notes": notes, **extra}


def run(snapshot_dir: Path = config.SNAPSHOT_DIR, *, fetch_t100=t100.fetch_t100,
        fetch_ontime_month=ontime.fetch_month, fetch_airports=ourairports.fetch_airports,
        workdir: Path | None = None) -> dict:
    snapshot_dir = Path(snapshot_dir)
    workdir = Path(workdir or tempfile.mkdtemp(prefix="ingest_"))

    t100_df, t100_dropped = t100.normalize_t100(fetch_t100(T100_SINCE))
    airports, airports_dropped = ourairports.normalize_airports(fetch_airports())

    end = max(t100_df["month"])
    start = _shift(end, -11)
    l12m = [_shift(start, i) for i in range(12)]

    am_parts, route_parts, missing = [], [], []
    for m in l12m:
        try:
            am, rt = fetch_ontime_month(m.year, m.month, workdir)
            am_parts.append(am)
            route_parts.append(rt)
            log.info("On-Time %s ok", m.strftime("%Y-%m"))
        except Exception as exc:  # one bad month must not abort the run
            log.warning("On-Time %s failed: %s", m.strftime("%Y-%m"), exc)
            missing.append(m.strftime("%Y-%m"))
    if not am_parts:
        raise RuntimeError("All On-Time months failed; keeping the previous snapshot")
    airport_month = pd.concat(am_parts, ignore_index=True)
    routes = pd.concat(route_parts, ignore_index=True)

    # Keep only airports that have T-100 traffic, and only traffic at known airports.
    airports = airports[airports["code"].isin(set(t100_df["code"]))].reset_index(drop=True)
    known = set(airports["code"])
    t100_keep = t100_df["code"].isin(known)
    am_keep = airport_month["code"].isin(known)
    rt_keep = routes["origin"].isin(known) & routes["dest"].isin(known)

    period = f"{start:%Y-%m} to {end:%Y-%m}"
    manifest = {
        "created_at": _now(),
        "l12m_start": start.isoformat(),
        "l12m_end": end.isoformat(),
        "sources": {
            "t100": _source(t100_keep.sum(), t100_dropped + (~t100_keep).sum(),
                            f"{T100_SINCE:%Y-%m} to {end:%Y-%m}",
                            "Origin-airport passengers, seats, departures; all US and foreign "
                            "carriers.", dataset="r495-tyji"),
            "ontime": _source(am_keep.sum(), (~am_keep).sum() + (~rt_keep).sum(), period,
                              "Domestic flights of reporting carriers only.",
                              missing_months=missing),
            "ourairports": _source(len(airports), airports_dropped, "current",
                                   "US airports with an IATA code and T-100 traffic."),
        },
    }

    tmp = snapshot_dir.with_name(snapshot_dir.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    try:
        airports.to_parquet(tmp / "airports.parquet", index=False)
        t100_df[t100_keep].to_parquet(tmp / "t100_monthly.parquet", index=False)
        airport_month[am_keep].to_parquet(tmp / "ontime_airport_month.parquet", index=False)
        routes[rt_keep].to_parquet(tmp / "ontime_routes.parquet", index=False)
        (tmp / "manifest.json").write_text(json.dumps(manifest, indent=2))
        Store(tmp)  # self-check: the app must be able to open what we wrote
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise

    # ponytail: two renames, not one atomic op; a crash between them leaves snapshot.old to restore
    old = snapshot_dir.with_name(snapshot_dir.name + ".old")
    shutil.rmtree(old, ignore_errors=True)
    if snapshot_dir.exists():
        snapshot_dir.rename(old)
    tmp.rename(snapshot_dir)
    shutil.rmtree(old, ignore_errors=True)
    return manifest


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    m = run()
    print(json.dumps({k: v for k, v in m.items() if k != "sources"}, indent=2))
    for key, src in m["sources"].items():
        print(f"{key}: kept={src['rows_kept']} dropped={src['rows_dropped']} "
              f"missing={src.get('missing_months', [])}")
