import numpy as np
import evaluate
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from app.logger import get_logger
from training.utils import load_dataset

logger = get_logger(__name__)

MODEL_NAME = "distilbert-base-uncased"


def tokenize_function(examples, tokenizer):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=128,
    )


def compute_metrics(eval_pred):
    metric_f1 = evaluate.load("f1")
    metric_acc = evaluate.load("accuracy")

    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    return {
        "accuracy": metric_acc.compute(predictions=preds, references=labels)["accuracy"],
        "weighted_f1": metric_f1.compute(predictions=preds, references=labels, average="weighted")["f1"],
    }


def train_intent_transformer(csv_path="data/raw/tickets.csv"):
    df = load_dataset(csv_path)

    label_map = {label: idx for idx, label in enumerate(sorted(df["intent"].unique()))}
    df["label"] = df["intent"].map(label_map)

    train_df, test_df = train_test_split(
        df[["text", "label"]],
        test_size=0.2,
        random_state=42,
        stratify=df["label"],
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
    test_ds = Dataset.from_pandas(test_df.reset_index(drop=True))

    train_ds = train_ds.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    test_ds = test_ds.map(lambda x: tokenize_function(x, tokenizer), batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label_map),
    )

    args = TrainingArguments(
        output_dir="artifacts/models/intent_transformer",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="artifacts/reports/intent_transformer_logs",
        load_best_model_at_end=True,
        metric_for_best_model="weighted_f1",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    logger.info("Starting transformer training")
    trainer.train()
    trainer.save_model("artifacts/models/intent_transformer")
    tokenizer.save_pretrained("artifacts/models/intent_transformer")
    logger.info("Transformer model saved successfully")


if __name__ == "__main__":
    train_intent_transformer()
