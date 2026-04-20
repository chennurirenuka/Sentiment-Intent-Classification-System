from dataclasses import dataclass
from typing import Dict, List

import joblib
import numpy as np

from app.config import (
    INTENT_LABEL_ENCODER_PATH,
    INTENT_MODEL_PATH,
    SENTIMENT_LABEL_ENCODER_PATH,
    SENTIMENT_MODEL_PATH,
    TFIDF_VECTORIZER_PATH,
)
from app.logger import get_logger
from app.preprocessing import batch_clean_text
from app.rules import routing_metadata

logger = get_logger(__name__)


@dataclass
class ModelArtifacts:
    vectorizer: object
    intent_model: object
    sentiment_model: object
    intent_encoder: object
    sentiment_encoder: object


class Predictor:
    def __init__(self) -> None:
        self.artifacts = self._load_artifacts()

    @staticmethod
    def _load_artifacts() -> ModelArtifacts:
        logger.info("Loading vectorizer, models, and label encoders")
        return ModelArtifacts(
            vectorizer=joblib.load(TFIDF_VECTORIZER_PATH),
            intent_model=joblib.load(INTENT_MODEL_PATH),
            sentiment_model=joblib.load(SENTIMENT_MODEL_PATH),
            intent_encoder=joblib.load(INTENT_LABEL_ENCODER_PATH),
            sentiment_encoder=joblib.load(SENTIMENT_LABEL_ENCODER_PATH),
        )

    def predict(self, texts: List[str]) -> List[Dict]:
        cleaned = batch_clean_text(texts)
        X = self.artifacts.vectorizer.transform(cleaned)

        intent_preds = self.artifacts.intent_model.predict(X)
        sentiment_preds = self.artifacts.sentiment_model.predict(X)

        intent_probs = self.artifacts.intent_model.predict_proba(X)
        sentiment_probs = self.artifacts.sentiment_model.predict_proba(X)

        results = []
        for i, raw_text in enumerate(texts):
            intent_label = self.artifacts.intent_encoder.inverse_transform([intent_preds[i]])[0]
            sentiment_label = self.artifacts.sentiment_encoder.inverse_transform([sentiment_preds[i]])[0]
            intent_confidence = float(np.max(intent_probs[i]))
            sentiment_confidence = float(np.max(sentiment_probs[i]))
            routing = routing_metadata(intent_label, sentiment_label, intent_confidence)

            results.append(
                {
                    "text": raw_text,
                    "clean_text": cleaned[i],
                    "intent": intent_label,
                    "intent_confidence": round(intent_confidence, 4),
                    "sentiment": sentiment_label,
                    "sentiment_confidence": round(sentiment_confidence, 4),
                    **routing,
                }
            )

        logger.info("Generated predictions for %s text(s)", len(texts))
        return results
