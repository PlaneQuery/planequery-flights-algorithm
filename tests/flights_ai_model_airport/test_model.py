import numpy as np

import flights_ai_model_airport.model as model_module


def test_model_training_supports_one_flight_group(monkeypatch):
    fitted = {}

    class FakeModel:
        def fit(self, X, y, **kwargs):
            fitted["X"] = X
            fitted["y"] = y
            fitted["kwargs"] = kwargs
            return self

    fake_model = FakeModel()
    monkeypatch.setattr(
        model_module.lgb,
        "LGBMClassifier",
        lambda **_kwargs: fake_model,
    )

    X = np.array([[0.0], [1.0], [2.0]])
    y = np.array([1, 0, 0])
    groups = np.array(["flight-1", "flight-1", "flight-1"])
    airport_idents = np.array(["KAAA", "KBBB", "KCCC"])

    result = model_module.model_training(
        X,
        y,
        groups,
        airport_idents,
        endpoint="takeoff",
    )

    assert result is fake_model
    np.testing.assert_array_equal(fitted["X"], X)
    np.testing.assert_array_equal(fitted["y"], y)
    assert fitted["kwargs"] == {}
