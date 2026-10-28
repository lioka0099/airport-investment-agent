"""BTS T-100 Segment Summary by Origin Airport (Socrata dataset r495-tyji)."""

from datetime import date

import pandas as pd

from airport_agent.data.http import get_with_retry

URL = "https://data.bts.gov/resource/r495-tyji.json"
PAGE = 50_000
COLUMNS = {
    "origin_airport_code": "code",
    "reporting_month": "month",
    "total_departures": "departures",
    "total_passengers": "passengers",
    "total_seats": "seats",
    "domestic_passengers": "domestic_passengers",
    "outbound_international_1": "intl_passengers",
}
NUMERIC = ["departures", "passengers", "seats", "domestic_passengers", "intl_passengers"]
MAX_LOAD_FACTOR = 1.05  # above this the row is a reporting error


def fetch_t100(since: date, get=get_with_retry) -> pd.DataFrame:
    pages, offset = [], 0
    while True:
        params = {
            "$select": ",".join(COLUMNS),
            "$where": f"reporting_month >= '{since.isoformat()}T00:00:00'",  # our constant
            "$order": ":id",
            "$limit": PAGE,
            "$offset": offset,
        }
        rows = get(URL, params).json()
        pages.append(pd.DataFrame(rows))
        if len(rows) < PAGE:
            return pd.concat(pages, ignore_index=True)
        offset += PAGE


def normalize_t100(raw: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    df = raw[list(COLUMNS)].rename(columns=COLUMNS)
    df["code"] = df["code"].astype(str).str.strip().str.upper()
    df["month"] = pd.to_datetime(df["month"], errors="coerce").dt.date
    for col in NUMERIC:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    valid = (
        df["code"].str.fullmatch(r"[A-Z]{3}")
        & df["month"].notna()
        & df[NUMERIC].notna().all(axis=1)
        & (df[NUMERIC] >= 0).all(axis=1)
        & (df["passengers"] <= df["seats"] * MAX_LOAD_FACTOR)
    )
    df = df[valid].copy()
    df[NUMERIC] = df[NUMERIC].astype("int64")
    # Several carriers report the same airport-month as separate rows in some periods.
    df = df.groupby(["code", "month"], as_index=False)[NUMERIC].sum()
    return df, int((~valid).sum())
