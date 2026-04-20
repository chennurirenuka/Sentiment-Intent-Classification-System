from pathlib import Path
from typing import Tuple

import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from app.preprocessing import batch_clean_text


def load_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    required_columns = {"text", "intent", "sentiment"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")

    df = df.dropna(subset=["text", "intent", "sentiment"]).copy()
    df["text"] = df["text"].astype(str).str.strip()
    df["intent"] = df["intent"].astype(str).str.strip()
    df["sentiment"] = df["sentiment"].astype(str).str.strip()
    df = df[df["text"] != ""].copy()

    df["clean_text"] = batch_clean_text(df["text"].tolist())
    return df


def encode_labels(series: pd.Series, save_path: Path) -> Tuple[LabelEncoder, pd.Series]:
    encoder = LabelEncoder()
    encoded = encoder.fit_transform(series)
    joblib.dump(encoder, save_path)
    return encoder, encoded
