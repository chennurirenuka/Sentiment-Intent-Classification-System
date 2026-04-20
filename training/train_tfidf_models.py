import json

import joblib
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from app.config import (
    INTENT_LABEL_ENCODER_PATH,
    INTENT_MODEL_PATH,
    INTENT_XGB_MODEL_PATH,
    MAX_FEATURES,
    METRICS_PATH,
    RANDOM_STATE,
    RAW_DATA_DIR,
    SENTIMENT_LABEL_ENCODER_PATH,
    SENTIMENT_MODEL_PATH,
    SENTIMENT_XGB_MODEL_PATH,
    TEST_SIZE,
    TFIDF_VECTORIZER_PATH,
)
from app.features import build_tfidf_vectorizer
from app.logger import get_logger
from training.utils import encode_labels, load_dataset

logger = get_logger(__name__)


def _train_and_evaluate_model(model, X_train, y_train, X_test, y_test, label_names):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Force sklearn to use the full class list even if some classes
    # are missing in this particular test split
    label_ids = list(range(len(label_names)))

    report = classification_report(
        y_test,
        y_pred,
        labels=label_ids,
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_test,
        y_pred,
        labels=label_ids,
    ).tolist()

    return model, report, weighted_f1, matrix


def train_single_task(
    X_train_text,
    X_test_text,
    y_train,
    y_test,
    task_name,
    label_names,
    linear_model_path,
    xgb_model_path,
    vectorizer=None,
    fit_vectorizer=False,
):
    if fit_vectorizer:
        X_train = vectorizer.fit_transform(X_train_text)
        X_test = vectorizer.transform(X_test_text)
        joblib.dump(vectorizer, TFIDF_VECTORIZER_PATH)
    else:
        X_train = vectorizer.transform(X_train_text)
        X_test = vectorizer.transform(X_test_text)

    ros = RandomOverSampler(random_state=RANDOM_STATE)
    X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)

    lr_model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    lr_model, report_lr, weighted_f1_lr, matrix_lr = _train_and_evaluate_model(
        lr_model,
        X_train_resampled,
        y_train_resampled,
        X_test,
        y_test,
        label_names,
    )
    joblib.dump(lr_model, linear_model_path)

    xgb_model = XGBClassifier(
        objective="multi:softprob",
        num_class=len(label_names),
        eval_metric="mlogloss",
        n_estimators=300,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=RANDOM_STATE,
    )
    xgb_model, report_xgb, weighted_f1_xgb, matrix_xgb = _train_and_evaluate_model(
        xgb_model,
        X_train_resampled,
        y_train_resampled,
        X_test,
        y_test,
        label_names,
    )
    joblib.dump(xgb_model, xgb_model_path)

    return {
        f"{task_name}_logistic_regression": {
            "weighted_f1": weighted_f1_lr,
            "classification_report": report_lr,
            "confusion_matrix": matrix_lr,
        },
        f"{task_name}_xgboost": {
            "weighted_f1": weighted_f1_xgb,
            "classification_report": report_xgb,
            "confusion_matrix": matrix_xgb,
        },
    }


def main():
    csv_path = RAW_DATA_DIR / "tickets.csv"
    df = load_dataset(str(csv_path))
    logger.info("Loaded dataset with %s rows", len(df))

    intent_encoder, y_intent = encode_labels(df["intent"], INTENT_LABEL_ENCODER_PATH)
    sentiment_encoder, y_sentiment = encode_labels(df["sentiment"], SENTIMENT_LABEL_ENCODER_PATH)

    # Use only intent for stratification on small datasets
    intent_counts = pd.Series(y_intent).value_counts()
    can_stratify = intent_counts.min() >= 2

    if can_stratify:
        logger.info("Using stratified split based on intent labels")
    else:
        logger.warning("Skipping stratified split because at least one intent class has fewer than 2 samples")

    (
        X_train_text,
        X_test_text,
        y_intent_train,
        y_intent_test,
        y_sent_train,
        y_sent_test,
    ) = train_test_split(
        df["clean_text"],
        y_intent,
        y_sentiment,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_intent if can_stratify else None,
    )

    vectorizer = build_tfidf_vectorizer(max_features=MAX_FEATURES)

    intent_metrics = train_single_task(
        X_train_text=X_train_text,
        X_test_text=X_test_text,
        y_train=y_intent_train,
        y_test=y_intent_test,
        task_name="intent",
        label_names=list(intent_encoder.classes_),
        linear_model_path=INTENT_MODEL_PATH,
        xgb_model_path=INTENT_XGB_MODEL_PATH,
        vectorizer=vectorizer,
        fit_vectorizer=True,
    )

    loaded_vectorizer = joblib.load(TFIDF_VECTORIZER_PATH)

    sentiment_metrics = train_single_task(
        X_train_text=X_train_text,
        X_test_text=X_test_text,
        y_train=y_sent_train,
        y_test=y_sent_test,
        task_name="sentiment",
        label_names=list(sentiment_encoder.classes_),
        linear_model_path=SENTIMENT_MODEL_PATH,
        xgb_model_path=SENTIMENT_XGB_MODEL_PATH,
        vectorizer=loaded_vectorizer,
        fit_vectorizer=False,
    )

    metrics = {**intent_metrics, **sentiment_metrics}

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    summary_df = pd.DataFrame(
        {
            "model": list(metrics.keys()),
            "weighted_f1": [metrics[m]["weighted_f1"] for m in metrics],
        }
    )

    print("Training completed successfully.")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()