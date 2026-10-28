from datetime import date

import pandas as pd

from airport_agent.data.sources import ontime, ourairports, t100

ONTIME_HEADER = ("Year,Month,Origin,Dest,Reporting_Airline,Cancelled,Diverted,DepDel15,TaxiOut,"
                 "TaxiIn,Distance,CarrierDelay,WeatherDelay,NASDelay,SecurityDelay,LateAircraftDelay")


def test_normalize_t100_types_and_drops():
    raw = pd.DataFrame([
        {"reporting_month": "2026-07-01T00:00:00.000", "origin_airport_code": "SFO",
         "total_departures": "100", "total_passengers": "15000", "total_seats": "18000",
         "domestic_passengers": "12000", "outbound_international_1": "3000"},
        {"reporting_month": "2026-07-01T00:00:00.000", "origin_airport_code": "1X",  # bad code
         "total_departures": "1", "total_passengers": "1", "total_seats": "1",
         "domestic_passengers": "1", "outbound_international_1": "0"},
        {"reporting_month": "2026-07-01T00:00:00.000", "origin_airport_code": "BOS",  # pax >> seats
         "total_departures": "1", "total_passengers": "900", "total_seats": "100",
         "domestic_passengers": "900", "outbound_international_1": "0"},
    ])
    df, dropped = t100.normalize_t100(raw)
    assert dropped == 2
    row = df.iloc[0]
    assert row["code"] == "SFO" and row["month"] == date(2026, 7, 1)
    assert row["passengers"] == 15000 and row["intl_passengers"] == 3000


def test_aggregate_csv_attributes_taxi_out_to_origin_and_taxi_in_to_dest(tmp_path):
    csv = tmp_path / "m.csv"
    csv.write_text("\n".join([
        ONTIME_HEADER,
        "2026,7,SFO,LAX,UA,0,0,1,30,5,337,0,0,20,0,0",
        "2026,7,LAX,SFO,UA,0,0,0,10,8,337,0,0,0,0,0",
        "2026,7,SFO,LAX,UA,1,0,,,,337,,,,,",
    ]))
    airport_month, routes = ontime.aggregate_csv(csv)
    sfo = airport_month.set_index("code").loc["SFO"]
    assert sfo["dep_flights"] == 2 and sfo["dep_operated"] == 1 and sfo["cancelled_n"] == 1
    assert sfo["taxi_out_sum"] == 30 and sfo["taxi_out_n"] == 1   # from Origin = SFO
    assert sfo["taxi_in_sum"] == 8 and sfo["taxi_in_n"] == 1      # from Dest = SFO
    lax = airport_month.set_index("code").loc["LAX"]
    assert lax["nas_min"] == 20                                    # cause minutes -> destination
    assert sfo["month"] == date(2026, 7, 1)
    by_route = routes.set_index(["origin", "dest"])
    assert by_route.loc[("SFO", "LAX"), "flights"] == 1  # cancelled excluded


def test_normalize_airports_keeps_us_with_iata():
    raw = pd.DataFrame([
        {"type": "large_airport", "name": "San Francisco Intl", "latitude_deg": 37.6,
         "longitude_deg": -122.4, "iso_country": "US", "iso_region": "US-CA",
         "municipality": "San Francisco", "iata_code": "SFO"},
        {"type": "heliport", "name": "Pad", "latitude_deg": 1, "longitude_deg": 1,
         "iso_country": "US", "iso_region": "US-CA", "municipality": "X", "iata_code": "HHH"},
        {"type": "large_airport", "name": "Toronto", "latitude_deg": 43, "longitude_deg": -79,
         "iso_country": "CA", "iso_region": "CA-ON", "municipality": "Toronto", "iata_code": "YYZ"},
        {"type": "medium_airport", "name": "San Juan", "latitude_deg": 18.4, "longitude_deg": -66,
         "iso_country": "PR", "iso_region": "PR-U-A", "municipality": "San Juan",
         "iata_code": "SJU"},
    ])
    df, dropped = ourairports.normalize_airports(raw)
    assert list(df["code"]) == ["SFO", "SJU"]
    assert list(df["state"]) == ["CA", "PR"]
    assert dropped == 2
