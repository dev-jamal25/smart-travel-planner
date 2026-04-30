"""Tests for the classifier schema, service, and endpoint.

None of these tests require the real joblib artifact on disk.
The endpoint test overrides get_classifier_service via dependency injection.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
from pydantic import ValidationError

from app.dependencies import get_classifier_service
from app.main import app
from app.schemas.classifier import ClassifierPredictionRequest
from app.services.classifier_service import ClassifierService

_VALID = {
    "country": "France",
    "continent": "Europe",
    "destination_type": "Cultural",
    "avg_cost_usd_per_day": 150.0,
    "best_season": "Spring",
    "avg_rating": 4.5,
    "annual_visitors_m": 10.0,
    "unesco_site": True,
}


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------


def test_schema_accepts_valid_input() -> None:
    req = ClassifierPredictionRequest(**_VALID)
    assert req.country == "France"
    assert req.unesco_site is True


def test_schema_rejects_negative_cost() -> None:
    with pytest.raises(ValidationError):
        ClassifierPredictionRequest(**{**_VALID, "avg_cost_usd_per_day": -1.0})


def test_schema_rejects_rating_above_5() -> None:
    with pytest.raises(ValidationError):
        ClassifierPredictionRequest(**{**_VALID, "avg_rating": 5.1})


def test_schema_rejects_rating_below_0() -> None:
    with pytest.raises(ValidationError):
        ClassifierPredictionRequest(**{**_VALID, "avg_rating": -0.1})


def test_schema_rejects_empty_country() -> None:
    with pytest.raises(ValidationError):
        ClassifierPredictionRequest(**{**_VALID, "country": ""})


def test_schema_rejects_empty_best_season() -> None:
    with pytest.raises(ValidationError):
        ClassifierPredictionRequest(**{**_VALID, "best_season": ""})


def test_schema_rejects_negative_visitors() -> None:
    with pytest.raises(ValidationError):
        ClassifierPredictionRequest(**{**_VALID, "annual_visitors_m": -0.1})


# ---------------------------------------------------------------------------
# Service unit tests (no HTTP, no real model)
# ---------------------------------------------------------------------------


def _make_fake_model(label: str = "Adventure", proba: list[float] | None = None) -> MagicMock:
    fake = MagicMock()
    fake.predict.return_value = np.array([label])
    if proba is not None:
        fake.predict_proba.return_value = np.array([proba])
    else:
        del fake.predict_proba  # remove attribute so hasattr returns False
    return fake


def test_service_returns_travel_style_and_confidence() -> None:
    fake_model = _make_fake_model("Adventure", [0.1, 0.7, 0.05, 0.05, 0.05, 0.05])
    service = ClassifierService(model=fake_model)
    result = service.predict(ClassifierPredictionRequest(**_VALID))
    assert result.travel_style == "Adventure"
    assert result.confidence == pytest.approx(0.7)


def test_service_confidence_is_none_when_no_predict_proba() -> None:
    fake_model = _make_fake_model("Culture")
    service = ClassifierService(model=fake_model)
    result = service.predict(ClassifierPredictionRequest(**_VALID))
    assert result.travel_style == "Culture"
    assert result.confidence is None


def test_service_maps_columns_to_training_names() -> None:
    """Verify DataFrame sent to the model uses exact training column names."""
    captured: dict[str, object] = {}

    def fake_predict(df):  # type: ignore[no-untyped-def]
        captured["df"] = df
        return np.array(["Luxury"])

    fake_model = MagicMock(spec=["predict"])
    fake_model.predict.side_effect = fake_predict

    ClassifierService(model=fake_model).predict(ClassifierPredictionRequest(**_VALID))

    import pandas as pd

    df = captured["df"]
    assert isinstance(df, pd.DataFrame)
    assert "Country" in df.columns
    assert "Avg Cost (USD/day)" in df.columns
    assert "Annual Visitors (M)" in df.columns
    assert "UNESCO Site" in df.columns
    assert df["UNESCO Site"].iloc[0] == "Yes"


def test_service_maps_unesco_false_to_no() -> None:
    captured: dict[str, object] = {}

    def fake_predict(df):  # type: ignore[no-untyped-def]
        captured["df"] = df
        return np.array(["Budget"])

    fake_model = MagicMock(spec=["predict"])
    fake_model.predict.side_effect = fake_predict

    ClassifierService(model=fake_model).predict(
        ClassifierPredictionRequest(**{**_VALID, "unesco_site": False})
    )

    import pandas as pd

    df = captured["df"]
    assert isinstance(df, pd.DataFrame)
    assert df["UNESCO Site"].iloc[0] == "No"


# ---------------------------------------------------------------------------
# Endpoint tests (dependency override, no real model)
# ---------------------------------------------------------------------------


@pytest.fixture()
def fake_classifier_service() -> ClassifierService:
    fake_model = _make_fake_model("Adventure", [0.1, 0.7, 0.05, 0.05, 0.05, 0.05])
    return ClassifierService(model=fake_model)


@pytest.fixture()
def classifier_client(client, fake_classifier_service: ClassifierService):  # type: ignore[no-untyped-def]
    """Extend the base client fixture with a classifier service override."""
    app.dependency_overrides[get_classifier_service] = lambda: fake_classifier_service
    yield client
    app.dependency_overrides.pop(get_classifier_service, None)


def test_predict_endpoint_success(classifier_client) -> None:  # type: ignore[no-untyped-def]
    response = classifier_client.post("/classifier/predict", json=_VALID)
    assert response.status_code == 200
    data = response.json()
    assert data["travel_style"] == "Adventure"
    assert data["confidence"] == pytest.approx(0.7)


def test_predict_endpoint_returns_422_on_invalid_rating(classifier_client) -> None:  # type: ignore[no-untyped-def]
    response = classifier_client.post(
        "/classifier/predict",
        json={**_VALID, "avg_rating": 99.0},
    )
    assert response.status_code == 422


def test_predict_endpoint_returns_422_on_empty_country(classifier_client) -> None:  # type: ignore[no-untyped-def]
    response = classifier_client.post(
        "/classifier/predict",
        json={**_VALID, "country": ""},
    )
    assert response.status_code == 422
