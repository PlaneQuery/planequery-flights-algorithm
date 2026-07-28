from datetime import datetime

import polars as pl

from data_engineering.flights.flight_type import Flight, flights_to_flights_df
from flights.flights_match_adsb import add_adsb_match_column, flights_match_adsb


START = datetime(2026, 3, 1)


def test_early_takeoff_with_missing_initial_coverage_is_unknown(monkeypatch):
    import flights.flights_match_adsb as match_module

    class FakeAirportLookup:
        def get_airport_coordinates(self, airport_ident):
            return (0.0, 0.0)

    monkeypatch.setattr(match_module, "airport_lookup", FakeAirportLookup())
    flights = flights_to_flights_df(
        [
            Flight(
                icao="abc123",
                takeoff_time=datetime(2026, 3, 1, 1),
                landing_time=datetime(2026, 3, 1, 2),
                takeoff_airport_ident="KAAA",
                landing_airport_ident="KBBB",
            )
        ]
    )
    adsb = pl.DataFrame(
        {
            "icao": ["abc123", "abc123"],
            "time": [datetime(2026, 3, 1, 0, 30), datetime(2026, 3, 1, 2)],
            "lat": [10.0, 0.0],
            "lon": [10.0, 0.0],
        }
    )

    assert flights_match_adsb(flights, adsb, START) == "unknown"


def test_add_adsb_match_column_preserves_unknown_as_null(monkeypatch):
    import flights.flights_match_adsb as match_module

    flights = pl.DataFrame(
        {
            "icao": ["unknown", "matched"],
            "takeoff_time": [
                datetime(2026, 3, 1, 1),
                datetime(2026, 3, 1, 10),
            ],
        }
    )
    adsb = pl.DataFrame(
        {
            "icao": ["unknown", "matched"],
            "time": [START, START],
        }
    )

    def fake_match(df_flights, df_adsb, start_dt):
        return "unknown" if df_flights.get_column("icao").item() == "unknown" else True

    monkeypatch.setattr(match_module, "flights_match_adsb", fake_match)

    result = add_adsb_match_column(flights, adsb, START).sort("icao")

    assert result.get_column("adsb_matched").to_list() == [True, None]
