import numpy as np
from typing import Any


def extract_features_single(
    text: str,
    vectorizer: Any,
) -> np.ndarray:
    return vectorizer.transform([text]).toarray()


def extract_features_batch(
    texts,
    vectorizer: Any,
    fit: bool = True,
) -> np.ndarray:
    if fit:
        return vectorizer.fit_transform(texts).toarray()
    return vectorizer.transform(texts).toarray()
