from __future__ import annotations

from typing import Any

import pandas as pd
import structlog

from app.schemas.classifier import ClassifierPredictionRequest, ClassifierPredictionResponse

logger = structlog.get_logger(__name__)

# Maps API-friendly snake_case field names to the exact column names the
# trained sklearn pipeline expects. Keep in sync with model_metadata.json.
_COLUMN_MAP: dict[str, str] = {
    "country": "Country",
    "continent": "Continent",
    "destination_type": "Type",
    "avg_cost_usd_per_day": "Avg Cost (USD/day)",
    "best_season": "Best Season",
    "avg_rating": "Avg Rating",
    "annual_visitors_m": "Annual Visitors (M)",
}


class ClassifierService:
    def __init__(self, model: Any) -> None:
        self._model = model

    def predict(self, request: ClassifierPredictionRequest) -> ClassifierPredictionResponse:
        # UNESCO Site was encoded as a categorical string in training data
        unesco_value = "Yes" if request.unesco_site else "No"

        row: dict[str, Any] = {
            "Country": request.country,
            "Continent": request.continent,
            "Type": request.destination_type,
            "Avg Cost (USD/day)": request.avg_cost_usd_per_day,
            "Best Season": request.best_season,
            "Avg Rating": request.avg_rating,
            "Annual Visitors (M)": request.annual_visitors_m,
            "UNESCO Site": unesco_value,
        }

        df = pd.DataFrame([row])
        travel_style = str(self._model.predict(df)[0])

        confidence: float | None = None
        if hasattr(self._model, "predict_proba"):
            proba = self._model.predict_proba(df)[0]
            confidence = float(max(proba))

        logger.debug(
            "classifier.predict",
            travel_style=travel_style,
            confidence=confidence,
        )
        return ClassifierPredictionResponse(travel_style=travel_style, confidence=confidence)
