from datetime import date, datetime, timedelta

import numpy as np
import polars as pl

from airports.airport_lookup import Airport, Airport_Types
from data_engineering.adsb.adsb_messages_types import AdsbMessage
from data_engineering.flights.flight_type import Flight
from flights_ai_model_airport.features import (
    TrainingDataCache,
    build_training_data,
    create_flight_airport_features,
    extract_from_messages,
)


def test_extract_from_messages_is_invariant_to_input_order():
    start = datetime(2026, 3, 1, 10, 0)
    messages = [
        AdsbMessage(
            time=start + timedelta(minutes=offset),
            icao="abc123",
            lat=float(offset),
            lon=-float(offset),
            baro_altitude_ft=offset * 1_000,
            track_deg=float(offset * 10),
        )
        for offset in (10, 0, 20)
    ]

    assert extract_from_messages(messages, "takeoff") == (
        start,
        0.0,
        -0.0,
        0,
        0.0,
    )
    assert extract_from_messages(messages, "landing") == (
        start + timedelta(minutes=20),
        20.0,
        -20.0,
        20_000,
        200.0,
    )


def test_extract_from_messages_keeps_position_and_motion_at_same_timestamp():
    start = datetime(2026, 3, 1, 10, 0)
    messages = [
        AdsbMessage(
            time=start,
            icao="abc123",
            lat=1.0,
            lon=-1.0,
            baro_altitude_ft=None,
            track_deg=10.0,
        ),
        AdsbMessage(
            time=start + timedelta(minutes=1),
            icao="abc123",
            lat=2.0,
            lon=-2.0,
            baro_altitude_ft=1_000,
            track_deg=20.0,
        ),
    ]

    assert extract_from_messages(messages, "takeoff") == (
        start + timedelta(minutes=1),
        2.0,
        -2.0,
        1_000,
        20.0,
    )


def test_candidate_features_keep_existing_nearest_airport(monkeypatch):
    import flights_ai_model_airport.features as features_module

    nearby = Airport(
        ident="KAAA",
        iata="AAA",
        lat=0.0,
        lon=0.0,
        elevation_ft=0,
        type=Airport_Types.SMALL_AIRPORT,
    )
    baseline = Airport(
        ident="KBBB",
        iata="BBB",
        lat=2.0,
        lon=2.0,
        elevation_ft=0,
        type=Airport_Types.SMALL_AIRPORT,
    )

    class FakeAirportLookup:
        def getAirportsWithinRadius(self, lat, lon, radius_km):
            return [nearby]

        def get_Airport_from_airport_ident(self, ident):
            return baseline if ident == baseline.ident else None

    monkeypatch.setattr(features_module, "airport_lookup", FakeAirportLookup())
    flight = Flight(
        icao="abc123",
        takeoff_time=datetime(2026, 3, 1, 10, 0),
        landing_time=datetime(2026, 3, 1, 11, 0),
        takeoff_airport_ident=baseline.ident,
        landing_airport_ident=nearby.ident,
    )

    rows = create_flight_airport_features(
        lat=0.0,
        lon=0.0,
        baro_altitude_ft=1_000,
        track=90.0,
        flight=flight,
        endpoint="takeoff",
    )

    assert [airport_ident for _features, _label, airport_ident in rows] == [
        nearby.ident,
        baseline.ident,
    ]
    assert [label for _features, label, _airport_ident in rows] == [0, 1]


def test_build_training_data_can_limit_flights_across_dates(monkeypatch):
    import flights_ai_model_airport.features as features_module

    training_dates = [date(2026, 6, 1), date(2026, 6, 2)]

    def fake_get_training_flights(dt):
        day = dt.day
        return pl.DataFrame(
            {
                "icao": [f"icao-{day}-{i}" for i in (2, 0, 1)],
                "takeoff_time": [
                    datetime(2026, 6, day, hour)
                    for hour in (12, 10, 11)
                ],
            }
        )

    batches = []

    def fake_process_flights(df_flights, _df_adsb, _endpoint):
        batches.append(df_flights.get_column("takeoff_time").to_list())
        num_flights = len(df_flights)
        return (
            np.zeros((num_flights, 1)),
            np.zeros(num_flights),
            np.array(df_flights.get_column("flight_id").to_list()),
            np.array(["KTEST"] * num_flights),
        )

    monkeypatch.setattr(
        features_module,
        "get_training_flights",
        fake_get_training_flights,
    )
    monkeypatch.setattr(features_module, "read_adsb", lambda *_args, **_kwargs: pl.DataFrame())
    monkeypatch.setattr(features_module, "process_flights", fake_process_flights)

    X, y, groups, airport_idents = build_training_data(
        training_dates,
        max_flights=4,
    )

    assert len(X) == len(y) == len(groups) == len(airport_idents) == 4
    assert [len(batch) for batch in batches] == [3, 1]
    assert batches[0] == sorted(batches[0])
    assert batches[1] == [datetime(2026, 6, 2, 10)]


def test_training_data_cache_reuses_flights_adsb_and_endpoint_features(monkeypatch):
    import flights_ai_model_airport.features as features_module

    training_date = date(2026, 6, 1)
    calls = {"flights": 0, "adsb": 0, "features": 0}

    def fake_get_training_flights(_dt):
        calls["flights"] += 1
        return pl.DataFrame(
            {
                "icao": ["icao-2", "icao-0", "icao-1"],
                "takeoff_time": [
                    datetime(2026, 6, 1, hour)
                    for hour in (12, 10, 11)
                ],
            }
        )

    def fake_read_adsb(_dt, *, icaos):
        calls["adsb"] += 1
        assert set(icaos) == {"icao-0", "icao-1", "icao-2"}
        return pl.DataFrame()

    def fake_process_flights(df_flights, _df_adsb, _endpoint):
        calls["features"] += 1
        df_flights = features_module.add_flight_id_col(df_flights)
        num_flights = len(df_flights)
        return (
            np.zeros((num_flights, 1)),
            np.zeros(num_flights),
            np.array(df_flights.get_column("flight_id").to_list()),
            np.array(["KTEST"] * num_flights),
        )

    monkeypatch.setattr(
        features_module,
        "get_training_flights",
        fake_get_training_flights,
    )
    monkeypatch.setattr(features_module, "read_adsb", fake_read_adsb)
    monkeypatch.setattr(features_module, "process_flights", fake_process_flights)

    cache = TrainingDataCache()
    first_result = build_training_data(
        [training_date],
        max_flights=2,
        cache=cache,
    )
    second_result = build_training_data(
        [training_date],
        max_flights=3,
        cache=cache,
    )

    assert len(first_result[0]) == 2
    assert len(second_result[0]) == 3
    assert calls == {"flights": 1, "adsb": 1, "features": 1}
