"""Deterministic synthetic snapshot. Numbers are chosen so tests can assert known outcomes."""

import json
from datetime import date
from pathlib import Path

import pandas as pd

# code: (name, city, state, lat, lon, pax_month, lf, growth, pax_2019_month,
#        taxi_out, taxi_in, nas_per_arr, ontime_share, del15_rate)
AIRPORTS = {
    "BOS": ("Boston Logan Intl", "Boston", "MA", 42.36, -71.01, 1_500_000, 0.86, 0.05, 1_450_000,
            18, 12, 6, 0.7, 0.22),
    "BDL": ("Bradley Intl", "Windsor Locks", "CT", 41.94, -72.68, 300_000, 0.80, 0.08, 290_000,
            14, 9, 2, 0.7, 0.18),
    "PVD": ("Rhode Island TF Green Intl", "Warwick", "RI", 41.73, -71.42, 180_000, 0.78, 0.02,
            190_000, 13, 6, 1.5, 0.7, 0.17),
    "BTV": ("Burlington Intl", "Burlington", "VT", 44.47, -73.15, 30_000, 0.75, 0.10, 28_000,
            12, 6, 1, 0.7, 0.15),
    "LAX": ("Los Angeles Intl", "Los Angeles", "CA", 33.94, -118.41, 3_000_000, 0.84, 0.03,
            3_400_000, 20, 13, 7, 0.6, 0.25),
    "SNA": ("John Wayne Airport", "Santa Ana", "CA", 33.68, -117.87, 450_000, 0.83, 0.06,
            450_000, 13, 7.5, 2, 0.7, 0.16),
    "SFO": ("San Francisco Intl", "San Francisco", "CA", 37.62, -122.38, 2_000_000, 0.85, 0.04,
            2_300_000, 19, 7.2, 9, 0.6, 0.27),
    "ANC": ("Ted Stevens Anchorage Intl", "Anchorage", "AK", 61.17, -149.99, 220_000, 0.80, 0.03,
            230_000, 12, 6, 1, 0.15, 0.12),
}

# (origin, dest, carrier, monthly flights, distance_mi)
ROUTES = [
    ("ANC", "SEA", "AS", 300, 1448), ("ANC", "ORD", "UA", 60, 2846), ("ANC", "ATL", "DL", 30, 3417),
    ("ANC", "FAI", "AS", 200, 261), ("ANC", "DEN", "UA", 40, 2405),
    ("BOS", "LAX", "DL", 400, 2611), ("BOS", "ORD", "UA", 500, 867), ("BOS", "DCA", "AA", 600, 399),
    ("BOS", "MCO", "B6", 700, 1121),
    ("SFO", "LAX", "UA", 900, 337), ("SFO", "JFK", "UA", 400, 2586), ("SFO", "SEA", "AS", 500, 679),
    ("LAX", "JFK", "AA", 600, 2475), ("LAX", "SFO", "UA", 900, 337), ("LAX", "LAS", "WN", 800, 236),
    ("SNA", "DEN", "WN", 300, 846), ("SNA", "SFO", "UA", 200, 372),
    ("BDL", "ORD", "UA", 150, 783), ("PVD", "BWI", "WN", 120, 328), ("BTV", "JFK", "B6", 40, 266),
]


def _months(start: date, n: int) -> list[date]:
    out, y, m = [], start.year, start.month
    for _ in range(n):
        out.append(date(y, m, 1))
        m += 1
        if m == 13:
            y, m = y + 1, 1
    return out


L12M_MONTHS = _months(date(2025, 8, 1), 12)
PRIOR_MONTHS = _months(date(2024, 8, 1), 12)
Y2019_MONTHS = _months(date(2019, 1, 1), 12)


def build_snapshot(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    airports, t100, ontime, routes = [], [], [], []
    for code, (name, city, state, lat, lon, pax, lf, growth, pax19,
               t_out, t_in, nas, share, del15) in AIRPORTS.items():
        airports.append(dict(code=code, name=name, city=city, state=state, lat=lat, lon=lon))
        for months, monthly in ((L12M_MONTHS, pax), (PRIOR_MONTHS, pax / (1 + growth)),
                                (Y2019_MONTHS, pax19)):
            for mo in months:
                seats = monthly / lf
                t100.append(dict(
                    code=code, month=mo, departures=int(seats / 150), passengers=int(monthly),
                    seats=int(seats), domestic_passengers=int(monthly * 0.8),
                    intl_passengers=int(monthly * 0.2)))
        departures = int(pax / lf / 150)
        dep = int(departures * share)
        operated = int(dep * 0.98)
        for mo in L12M_MONTHS:
            ontime.append(dict(
                code=code, month=mo, dep_flights=dep, dep_operated=operated,
                dep_del15_n=int(operated * del15), cancelled_n=dep - operated,
                taxi_out_sum=float(t_out * operated), taxi_out_n=operated,
                arr_flights=operated, taxi_in_sum=float(t_in * operated), taxi_in_n=operated,
                nas_min=float(nas * operated), weather_min=float(0.5 * operated),
                carrier_min=float(3 * operated), late_aircraft_min=float(4 * operated),
                security_min=float(0.05 * operated)))
    for origin, dest, carrier, flights, dist in ROUTES:
        for mo in L12M_MONTHS:
            routes.append(dict(origin=origin, dest=dest, carrier=carrier, month=mo,
                               flights=flights, distance_mi=float(dist)))

    pd.DataFrame(airports).to_parquet(out_dir / "airports.parquet", index=False)
    pd.DataFrame(t100).to_parquet(out_dir / "t100_monthly.parquet", index=False)
    pd.DataFrame(ontime).to_parquet(out_dir / "ontime_airport_month.parquet", index=False)
    pd.DataFrame(routes).to_parquet(out_dir / "ontime_routes.parquet", index=False)
    manifest = {
        "created_at": "2026-10-28T00:00:00Z",
        "l12m_start": "2025-08-01",
        "l12m_end": "2026-07-01",
        "sources": {
            key: {"period": "2025-08 to 2026-07", "retrieved_at": "2026-10-28T00:00:00Z",
                  "rows_kept": 1, "rows_dropped": 0, "notes": "synthetic test data"}
            for key in ("t100", "ontime", "ourairports")
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return out_dir
