# Sentiment & Intent Classification System

A production-style NLP project for:
- intent classification
- sentiment classification
- priority derivation
- routing team prediction
- FastAPI deployment

## Features
- shared text preprocessing
- TF-IDF vectorizer
- Logistic Regression baseline
- XGBoost comparison model
- optional DistilBERT training path
- oversampling for class imbalance
- evaluation reports
- business rule layer
- FastAPI inference API
- structured logging
- basic test file

## Project structure

```text
sentiment_intent_system/
├── app/
├── training/
├── artifacts/
├── data/
├── logs/
├── notebooks/
├── tests/
├── requirements.txt
├── README.md
└── run.py
```

## Setup

```bash
python -m venv .venv
```

### Windows
```bash
.venv\Scripts\activate
```

### Linux / Mac
```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Train the TF-IDF models:

```bash
python -m training.train_tfidf_models
```

Run the API:

```bash
python run.py
```

Open Swagger UI:

```text
http://127.0.0.1:8000/docs
```

## Example request

```json
{
  "texts": [
    "I am unable to login after password reset and this is very frustrating",
    "Please add export to excel feature",
    "Thanks, the issue was resolved quickly"
  ]
}
```
