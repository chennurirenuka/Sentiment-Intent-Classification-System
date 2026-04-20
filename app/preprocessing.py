import re
from typing import List

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def _safe_download(resource_path: str, resource_name: str) -> None:
    try:
        nltk.data.find(resource_path)
    except LookupError:
        nltk.download(resource_name, quiet=True)


_safe_download("corpora/stopwords", "stopwords")
_safe_download("corpora/wordnet", "wordnet")
_safe_download("corpora/omw-1.4", "omw-1.4")

STOPWORDS = set(stopwords.words("english"))
IMPORTANT_WORDS = {"not", "no", "never", "cannot", "cant"}
STOPWORDS = STOPWORDS - IMPORTANT_WORDS
LEMMATIZER = WordNetLemmatizer()


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = text.lower().strip()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = text.split()
    tokens = [LEMMATIZER.lemmatize(token) for token in tokens if token not in STOPWORDS]

    return " ".join(tokens)


def batch_clean_text(texts: List[str]) -> List[str]:
    return [clean_text(text) for text in texts]
