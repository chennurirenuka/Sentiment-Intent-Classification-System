from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from app.inference import Predictor
from app.logger import get_logger
from app.schemas import PredictionRequest, PredictionResponse

logger = get_logger(__name__)
predictor: Predictor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    logger.info("Starting application")
    predictor = Predictor()
    yield
    logger.info("Shutting down application")


app = FastAPI(
    title="Sentiment & Intent Classification API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def health_check():
    return {"status": "ok", "message": "API is running"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        if predictor is None:
            raise RuntimeError("Predictor is not initialized")
        outputs = predictor.predict(request.texts)
        return {"predictions": outputs}
    except Exception as exc:
        logger.exception("Prediction failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
