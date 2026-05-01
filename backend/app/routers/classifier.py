from __future__ import annotations

from typing import Annotated

import structlog
from fastapi import APIRouter, Depends

from app.dependencies import get_classifier_service, get_current_user
from app.schemas.auth import CurrentUser
from app.schemas.classifier import ClassifierPredictionRequest, ClassifierPredictionResponse
from app.services.classifier_service import ClassifierService

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/classifier", tags=["classifier"])


@router.post("/predict", response_model=ClassifierPredictionResponse)
async def predict_travel_style(
    body: ClassifierPredictionRequest,
    classifier: Annotated[ClassifierService, Depends(get_classifier_service)],
    _current_user: Annotated[CurrentUser, Depends(get_current_user)],
) -> ClassifierPredictionResponse:
    logger.info(
        "classifier.predict.start",
        country=body.country,
        destination_type=body.destination_type,
        user_id=str(_current_user.user_id),
    )
    result = classifier.predict(body)
    logger.info(
        "classifier.predict.success",
        travel_style=result.travel_style,
        confidence=result.confidence,
        user_id=str(_current_user.user_id),
    )
    return result
