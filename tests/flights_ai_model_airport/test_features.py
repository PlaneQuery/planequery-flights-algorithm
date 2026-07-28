from datetime import datetime, timedelta

from airports.airport_lookup import Airport, Airport_Types
from data_engineering.adsb.adsb_messages_types import AdsbMessage
from data_engineering.flights.flight_type import Flight
from flights_ai_model_airport.features import (
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
