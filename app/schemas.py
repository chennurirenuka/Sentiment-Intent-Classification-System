from typing import List

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    texts: List[str] = Field(
        ...,
        min_length=1,
        examples=[["My VPN is not working and this is frustrating"]],
    )


class SinglePredictionResponse(BaseModel):
    text: str
    clean_text: str
    intent: str
    intent_confidence: float
    sentiment: str
    sentiment_confidence: float
    priority: str
    routing_team: str


class PredictionResponse(BaseModel):
    predictions: List[SinglePredictionResponse]
