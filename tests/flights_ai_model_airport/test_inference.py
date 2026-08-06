from datetime import date
from pathlib import Path

import polars as pl
from polars.testing import assert_frame_equal

import data_engineering.adsb.read_adsb as read_adsb_module
import flights_ai_model_airport.inference as inference_module


def test_run_inference_uses_supplied_flights(monkeypatch):
    run_date = date(2026, 2, 1)
    supplied_flights = pl.DataFrame({"icao": ["abc123", "def456", "abc123"]})
    supplied_adsb = pl.DataFrame({"icao": ["abc123"]})
    expected = pl.DataFrame({"result": [1]})
    loaded_model = object()
    captured_icaos = []

    def fail_if_all_flights_are_loaded(_run_date):
        raise AssertionError("get_flights should not be called")

    def fake_read_adsb(actual_date, *, icaos):
        assert actual_date == run_date
        captured_icaos.extend(icaos)
        return supplied_adsb

    def fake_inference(actual_flights, actual_model, actual_adsb):
        assert_frame_equal(actual_flights, supplied_flights)
        assert actual_model is loaded_model
        assert_frame_equal(actual_adsb, supplied_adsb)
        return expected

    monkeypatch.setattr(inference_module, "get_flights", fail_if_all_flights_are_loaded)
    monkeypatch.setattr(read_adsb_module, "read_adsb", fake_read_adsb)
    monkeypatch.setattr(inference_module, "load_model", lambda _path: loaded_model)
    monkeypatch.setattr(inference_module, "inference", fake_inference)

    actual = inference_module.run_inference(
        run_date,
        Path("model.pkl"),
        df_flights=supplied_flights,
    )

    assert set(captured_icaos) == {"abc123", "def456"}
    assert_frame_equal(actual, expected)
