from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
VECTORIZERS_DIR = ARTIFACTS_DIR / "vectorizers"
ENCODERS_DIR = ARTIFACTS_DIR / "label_encoders"
REPORTS_DIR = ARTIFACTS_DIR / "reports"
LOGS_DIR = BASE_DIR / "logs"

for path in [
    DATA_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    ARTIFACTS_DIR,
    MODELS_DIR,
    VECTORIZERS_DIR,
    ENCODERS_DIR,
    REPORTS_DIR,
    LOGS_DIR,
]:
    path.mkdir(parents=True, exist_ok=True)

INTENT_MODEL_PATH = MODELS_DIR / "intent_logistic_regression.joblib"
SENTIMENT_MODEL_PATH = MODELS_DIR / "sentiment_logistic_regression.joblib"
INTENT_XGB_MODEL_PATH = MODELS_DIR / "intent_xgboost.joblib"
SENTIMENT_XGB_MODEL_PATH = MODELS_DIR / "sentiment_xgboost.joblib"
TFIDF_VECTORIZER_PATH = VECTORIZERS_DIR / "tfidf_vectorizer.joblib"
INTENT_LABEL_ENCODER_PATH = ENCODERS_DIR / "intent_label_encoder.joblib"
SENTIMENT_LABEL_ENCODER_PATH = ENCODERS_DIR / "sentiment_label_encoder.joblib"
METRICS_PATH = REPORTS_DIR / "metrics.json"

RANDOM_STATE = 42
MAX_FEATURES = 20000
TEST_SIZE = 0.2
CONFIDENCE_THRESHOLD = 0.55

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000
