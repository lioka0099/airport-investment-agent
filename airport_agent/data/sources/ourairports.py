"""OurAirports airports.csv: names, cities, states and coordinates for US airports."""

import io

import pandas as pd

from airport_agent.data.http import get_with_retry

URL = "https://davidmegginson.github.io/ourairports-data/airports.csv"
KEEP_TYPES = {"large_airport", "medium_airport", "small_airport"}
US_COUNTRIES = {"US", "PR", "VI", "GU"}  # territories report to BTS like states


def fetch_airports(get=get_with_retry) -> pd.DataFrame:
    return pd.read_csv(io.StringIO(get(URL).text), keep_default_na=False)


def normalize_airports(raw: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    code = raw["iata_code"].astype(str).str.strip().str.upper()
    valid = (
        raw["iso_country"].isin(US_COUNTRIES)
        & raw["type"].isin(KEEP_TYPES)
        & code.str.fullmatch(r"[A-Z]{3}")
    )
    kept = raw[valid]
    state = kept["iso_region"].str.split("-").str[1].where(kept["iso_country"] == "US",
                                                           kept["iso_country"])
    df = pd.DataFrame({
        "code": code[valid],
        "name": kept["name"],
        "city": kept["municipality"],
        "state": state,
        "lat": pd.to_numeric(kept["latitude_deg"]),
        "lon": pd.to_numeric(kept["longitude_deg"]),
    }).drop_duplicates("code").reset_index(drop=True)
    return df, int(len(raw) - len(df))
