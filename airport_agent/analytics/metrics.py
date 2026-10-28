"""Per-airport KPIs over the last 12 months (L12M) of the snapshot."""

from datetime import date

import numpy as np
import pandas as pd

from airport_agent import config
from airport_agent.data.store import Store

CAUSES = {"nas": "nas_min", "weather": "weather_min", "carrier": "carrier_min",
          "late_aircraft": "late_aircraft_min", "security": "security_min"}


def _shift(month: date, n: int) -> date:
    idx = month.year * 12 + month.month - 1 + n
    return date(idx // 12, idx % 12 + 1, 1)


def window(store: Store) -> dict[str, date]:
    m = store.manifest()
    start, end = date.fromisoformat(m["l12m_start"]), date.fromisoformat(m["l12m_end"])
    return {"start": start, "end": end, "prior_start": _shift(start, -12),
            "prior_end": _shift(start, -1)}


AIRPORT_SQL = """
WITH t AS (
    SELECT code,
           sum(passengers) FILTER (WHERE month BETWEEN $1 AND $2) AS passengers_l12m,
           sum(seats) FILTER (WHERE month BETWEEN $1 AND $2) AS seats_l12m,
           sum(departures) FILTER (WHERE month BETWEEN $1 AND $2) AS departures_l12m,
           sum(intl_passengers) FILTER (WHERE month BETWEEN $1 AND $2) AS intl_passengers_l12m,
           count(*) FILTER (WHERE month BETWEEN $1 AND $2) AS months_t100,
           sum(passengers) FILTER (WHERE month BETWEEN $3 AND $4) AS passengers_prior,
           count(*) FILTER (WHERE month BETWEEN $3 AND $4) AS months_prior,
           sum(passengers) FILTER (WHERE year(month) = 2019) AS passengers_2019
    FROM t100_monthly GROUP BY code
),
o AS (
    SELECT code,
           sum(dep_flights) AS ontime_departures,
           sum(taxi_out_sum) / nullif(sum(taxi_out_n), 0) AS avg_taxi_out,
           sum(taxi_in_sum) / nullif(sum(taxi_in_n), 0) AS avg_taxi_in,
           sum(dep_del15_n) / nullif(sum(dep_operated), 0) AS dep_del15_rate,
           sum(cancelled_n) / nullif(sum(dep_flights), 0) AS cancel_rate,
           sum(nas_min) / nullif(sum(arr_flights), 0) AS nas_min_per_flight
    FROM ontime_airport_month WHERE month BETWEEN $1 AND $2 GROUP BY code
)
SELECT a.code, a.name, a.city, a.state, t.* EXCLUDE (code), o.* EXCLUDE (code)
FROM airports a JOIN t USING (code) LEFT JOIN o USING (code)
WHERE t.passengers_l12m > 0
"""


def airport_metrics(store: Store) -> pd.DataFrame:
    w = window(store)
    df = store.query(AIRPORT_SQL, [w["start"], w["end"], w["prior_start"], w["prior_end"]])
    df = df.set_index("code")
    pax = df["passengers_l12m"].astype(float)
    df["load_factor"] = pax / df["seats_l12m"].replace(0, np.nan)
    full_prior = df["months_prior"] == 12  # growth needs a complete prior year
    df["growth_yoy"] = (pax / df["passengers_prior"].replace(0, np.nan) - 1).where(full_prior)
    df["vs_2019"] = pax / df["passengers_2019"].replace(0, np.nan) - 1
    df["seats_gap"] = (pax / config.LF_TARGET - df["seats_l12m"]).clip(lower=0)
    df["intl_share"] = df["intl_passengers_l12m"] / pax
    df["ontime_coverage"] = df["ontime_departures"] / df["departures_l12m"].replace(0, np.nan)
    df["log_passengers"] = np.log10(pax)
    return df


def delay_cause_mix(store: Store, code: str) -> dict[str, float]:
    w = window(store)
    cols = ", ".join(f"sum({c}) AS {k}" for k, c in CAUSES.items())
    row = store.query(f"SELECT {cols} FROM ontime_airport_month WHERE code = $1 "
                      "AND month BETWEEN $2 AND $3", [code, w["start"], w["end"]]).iloc[0]
    total = row.sum(skipna=True)
    if not total or np.isnan(total):
        return {}
    return {k: float(row[k] / total) for k in CAUSES}


def _route_rows(store: Store, code: str) -> pd.DataFrame:
    w = window(store)
    return store.query(
        "SELECT dest, carrier, sum(flights) AS flights, avg(distance_mi) AS distance_mi "
        "FROM ontime_routes WHERE origin = $1 AND month BETWEEN $2 AND $3 GROUP BY dest, carrier",
        [code, w["start"], w["end"]])


def _dest_list(df: pd.DataFrame) -> list[dict]:
    return [{"dest": r.dest, "flights": int(r.flights), "distance_mi": round(float(r.distance_mi))}
            for r in df.itertuples()]


def route_mix(store: Store, code: str, threshold_mi: int) -> dict:
    rows = _route_rows(store, code)
    by_dest = (rows.groupby("dest", as_index=False)
               .agg(flights=("flights", "sum"), distance_mi=("distance_mi", "mean"))
               .sort_values("flights", ascending=False))
    total = int(by_dest["flights"].sum())
    if total == 0:
        return {"total_departures": 0, "long_haul_threshold_mi": threshold_mi,
                "long_haul_departures": 0, "long_haul_pct": None, "buckets": {},
                "top_destinations": [], "long_haul_destinations": []}
    long_haul = by_dest[by_dest["distance_mi"] >= threshold_mi]
    buckets = {}
    for low, high, label in config.DISTANCE_BUCKETS:
        in_bucket = (by_dest["distance_mi"] >= low) & (
            by_dest["distance_mi"] < high if high is not None else True)
        buckets[label] = float(by_dest.loc[in_bucket, "flights"].sum() / total)
    return {
        "total_departures": total,
        "long_haul_threshold_mi": threshold_mi,
        "long_haul_departures": int(long_haul["flights"].sum()),
        "long_haul_pct": float(long_haul["flights"].sum() / total),
        "buckets": buckets,
        "top_destinations": _dest_list(by_dest.head(5)),
        "long_haul_destinations": _dest_list(long_haul),
    }


def carrier_shares(store: Store, code: str) -> dict:
    by_carrier = _route_rows(store, code).groupby("carrier")["flights"].sum()
    total = by_carrier.sum()
    if not total:
        return {"shares": {}, "hhi": None, "top_carrier": None, "top_share": None, "band": None}
    shares = (by_carrier / total).sort_values(ascending=False)
    hhi = float(((shares * 100) ** 2).sum())  # Herfindahl-Hirschman, 0-10,000
    band = "low" if hhi < config.HHI_LOW else "high" if hhi > config.HHI_HIGH else "moderate"
    return {"shares": {k: float(v) for k, v in shares.items()}, "hhi": hhi,
            "top_carrier": shares.index[0], "top_share": float(shares.iloc[0]), "band": band}


def monthly_trend(store: Store, code: str) -> pd.DataFrame:
    return store.query(
        "SELECT month, passengers, seats, passengers / nullif(seats, 0) AS load_factor "
        "FROM t100_monthly WHERE code = $1 ORDER BY month", [code])
