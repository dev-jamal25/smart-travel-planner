from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field


class ClassifierPredictionRequest(BaseModel):
    country: Annotated[str, Field(min_length=1)]
    continent: Annotated[str, Field(min_length=1)]
    destination_type: Annotated[str, Field(min_length=1)]
    avg_cost_usd_per_day: Annotated[float, Field(ge=0)]
    best_season: Annotated[str, Field(min_length=1)]
    avg_rating: Annotated[float, Field(ge=0, le=5)]
    annual_visitors_m: Annotated[float, Field(ge=0)]
    unesco_site: bool


class ClassifierPredictionResponse(BaseModel):
    travel_style: str
    confidence: float | None
