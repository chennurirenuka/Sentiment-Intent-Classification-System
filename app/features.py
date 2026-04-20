from sklearn.feature_extraction.text import TfidfVectorizer


def build_tfidf_vectorizer(max_features: int = 20000) -> TfidfVectorizer:
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
        sublinear_tf=True,
    )
