"""BTS Airline On-Time Performance monthly PREZIP files, aggregated to airport-month and route."""

import shutil
import zipfile
from pathlib import Path

import duckdb
import pandas as pd

from airport_agent.data.http import get_with_retry

URL = ("https://www.transtats.bts.gov/PREZIP/"
       "On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{year}_{month}.zip")

# Taxi-out is attributed to the Origin (departing from X), taxi-in and delay-cause minutes to
# the Dest (arriving at X). Sums are stored, not averages, so means stay flight-weighted.
AIRPORT_SQL = """
WITH f AS (
    SELECT make_date(CAST("Year" AS INT), CAST("Month" AS INT), 1) AS month, "Origin" AS origin,
           "Dest" AS dest, "Cancelled" AS cancelled, "Diverted" AS diverted,
           "DepDel15" AS dep_del15,
           "TaxiOut" AS taxi_out, "TaxiIn" AS taxi_in, "NASDelay" AS nas, "WeatherDelay" AS weather,
           "CarrierDelay" AS carrier, "LateAircraftDelay" AS late_aircraft,
           "SecurityDelay" AS security
    FROM read_csv(?, header = true)
),
dep AS (
    SELECT origin AS code, month, count(*) AS dep_flights,
           count(*) FILTER (WHERE cancelled = 0) AS dep_operated,
           count(*) FILTER (WHERE cancelled = 0 AND dep_del15 = 1) AS dep_del15_n,
           count(*) FILTER (WHERE cancelled = 1) AS cancelled_n,
           coalesce(sum(taxi_out) FILTER (WHERE taxi_out BETWEEN 0 AND 180), 0) AS taxi_out_sum,
           count(taxi_out) FILTER (WHERE taxi_out BETWEEN 0 AND 180) AS taxi_out_n
    FROM f GROUP BY ALL
),
arr AS (
    SELECT dest AS code, month,
           count(*) FILTER (WHERE cancelled = 0 AND diverted = 0) AS arr_flights,
           coalesce(sum(taxi_in) FILTER (WHERE taxi_in BETWEEN 0 AND 180), 0) AS taxi_in_sum,
           count(taxi_in) FILTER (WHERE taxi_in BETWEEN 0 AND 180) AS taxi_in_n,
           coalesce(sum(nas), 0) AS nas_min, coalesce(sum(weather), 0) AS weather_min,
           coalesce(sum(carrier), 0) AS carrier_min,
           coalesce(sum(late_aircraft), 0) AS late_aircraft_min,
           coalesce(sum(security), 0) AS security_min
    FROM f GROUP BY ALL
)
SELECT dep.*, arr.* EXCLUDE (code, month) FROM dep JOIN arr USING (code, month)
"""

ROUTES_SQL = """
SELECT "Origin" AS origin, "Dest" AS dest, "Reporting_Airline" AS carrier,
       make_date(CAST("Year" AS INT), CAST("Month" AS INT), 1) AS month,
       count(*) AS flights, avg("Distance") AS distance_mi
FROM read_csv(?, header = true) WHERE "Cancelled" = 0 GROUP BY ALL
"""


def _as_date(df: pd.DataFrame) -> pd.DataFrame:
    df["month"] = pd.to_datetime(df["month"]).dt.date
    return df


def aggregate_csv(csv_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    con = duckdb.connect()
    path = str(csv_path)
    airport_month = _as_date(con.execute(AIRPORT_SQL, [path]).df())
    routes = _as_date(con.execute(ROUTES_SQL, [path]).df())
    return airport_month, routes


def fetch_month(year: int, month: int, workdir: Path,
                get=get_with_retry) -> tuple[pd.DataFrame, pd.DataFrame]:
    month_dir = workdir / f"ontime_{year}_{month}"
    month_dir.mkdir(parents=True, exist_ok=True)
    try:
        zip_path = month_dir / "data.zip"
        zip_path.write_bytes(get(URL.format(year=year, month=month), timeout=300).content)
        with zipfile.ZipFile(zip_path) as zf:
            csv_name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
            zf.extract(csv_name, month_dir)
        return aggregate_csv(month_dir / csv_name)
    finally:
        shutil.rmtree(month_dir, ignore_errors=True)  # ~300 MB per month: never keep it
